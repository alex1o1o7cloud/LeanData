import Mathlib

namespace smallest_next_divisor_after_221_l1076_107656

theorem smallest_next_divisor_after_221 (m : ℕ) (h1 : m ≥ 1000 ∧ m < 10000) 
  (h2 : m % 2 = 0) (h3 : m % 221 = 0) : 
  ∃ (d : ℕ), d ∣ m ∧ d > 221 ∧ d ≤ 442 ∧ ∀ (x : ℕ), x ∣ m → x > 221 → x ≥ d :=
by sorry

end smallest_next_divisor_after_221_l1076_107656


namespace quadratic_equation_with_absolute_roots_l1076_107628

theorem quadratic_equation_with_absolute_roots 
  (x₁ x₂ m : ℝ) 
  (h₁ : x₁ > 0) 
  (h₂ : x₂ < 0) 
  (h₃ : ∃ (original_eq : ℝ → Prop), original_eq x₁ ∧ original_eq x₂) :
  ∃ (new_eq : ℝ → Prop), 
    new_eq (|x₁|) ∧ 
    new_eq (|x₂|) ∧ 
    ∀ x, new_eq x ↔ x^2 - (1 - 4*m)/x + 2 = 0 :=
sorry

end quadratic_equation_with_absolute_roots_l1076_107628


namespace michaels_bunnies_l1076_107685

theorem michaels_bunnies (total_pets : ℕ) (dog_percent : ℚ) (cat_percent : ℚ) 
  (h1 : total_pets = 36)
  (h2 : dog_percent = 25 / 100)
  (h3 : cat_percent = 50 / 100)
  (h4 : dog_percent + cat_percent < 1) :
  (1 - dog_percent - cat_percent) * total_pets = 9 := by
  sorry

end michaels_bunnies_l1076_107685


namespace arithmetic_calculation_l1076_107639

theorem arithmetic_calculation : 5 * (9 / 3) + 7 * 4 - 36 / 4 = 34 := by
  sorry

end arithmetic_calculation_l1076_107639


namespace last_two_digits_of_sum_of_factorials_15_l1076_107668

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_of_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem last_two_digits_of_sum_of_factorials_15 :
  sum_of_factorials 15 % 100 = 13 := by sorry

end last_two_digits_of_sum_of_factorials_15_l1076_107668


namespace edge_bound_l1076_107696

/-- A simple graph with no 4-cycles -/
structure NoCycleFourGraph where
  -- The vertex set
  V : Type
  -- The edge relation
  E : V → V → Prop
  -- Symmetry of edges
  symm : ∀ u v, E u v → E v u
  -- No self-loops
  irrefl : ∀ v, ¬E v v
  -- No 4-cycles
  no_four_cycle : ∀ a b c d, E a b → E b c → E c d → E d a → (a = c ∨ b = d)

/-- The number of vertices in a graph -/
def num_vertices (G : NoCycleFourGraph) : ℕ := sorry

/-- The number of edges in a graph -/
def num_edges (G : NoCycleFourGraph) : ℕ := sorry

/-- The main theorem -/
theorem edge_bound (G : NoCycleFourGraph) :
  let n := num_vertices G
  let m := num_edges G
  m ≤ (n / 4) * (1 + Real.sqrt (4 * n - 3)) := by sorry

end edge_bound_l1076_107696


namespace picks_theorem_irregular_polygon_area_l1076_107623

/-- Pick's Theorem for a polygon on a lattice -/
theorem picks_theorem (B I : ℕ) (A : ℚ) : A = I + B / 2 - 1 →
  B = 10 → I = 12 → A = 16 := by
  sorry

/-- The area of the irregular polygon -/
theorem irregular_polygon_area : ∃ A : ℚ, A = 16 := by
  sorry

end picks_theorem_irregular_polygon_area_l1076_107623


namespace tangent_sum_equality_l1076_107609

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the triangle
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

-- Define the tangent line
def TangentLine (p : ℝ × ℝ) (c : Circle) : ℝ × ℝ → Prop := sorry

-- State that the circles are tangent
def CirclesTangent (c1 c2 : Circle) : Prop := sorry

-- State that the triangle is equilateral
def IsEquilateral (t : Triangle) : Prop := sorry

-- State that the triangle is inscribed in the larger circle
def Inscribed (t : Triangle) (c : Circle) : Prop := sorry

-- Define the length of a tangent line
def TangentLength (p : ℝ × ℝ) (c : Circle) : ℝ := sorry

-- Main theorem
theorem tangent_sum_equality 
  (c1 c2 : Circle) 
  (t : Triangle) 
  (h1 : CirclesTangent c1 c2) 
  (h2 : IsEquilateral t) 
  (h3 : Inscribed t c1) :
  ∃ (i j k : Fin 3), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    TangentLength (t.vertices i) c2 = 
    TangentLength (t.vertices j) c2 + TangentLength (t.vertices k) c2 :=
sorry

end tangent_sum_equality_l1076_107609


namespace range_of_a_l1076_107600

/-- Given functions f and g, prove the range of a -/
theorem range_of_a (a : ℝ) : 
  (∀ x₁ : ℝ, ∃ x₂ : ℝ, |2*x₁ - a| + |2*x₁ + 3| = |2*x₂ - 3| + 2) →
  (a ≥ -1 ∨ a ≤ -5) :=
by sorry

end range_of_a_l1076_107600


namespace x_value_proof_l1076_107653

theorem x_value_proof (x y : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) : x = 20 := by
  sorry

end x_value_proof_l1076_107653


namespace two_boxes_in_case_l1076_107635

/-- The number of boxes in a case, given the total number of blocks and blocks per box -/
def boxes_in_case (total_blocks : ℕ) (blocks_per_box : ℕ) : ℕ :=
  total_blocks / blocks_per_box

/-- Theorem: There are 2 boxes in a case when there are 12 blocks in total and 6 blocks per box -/
theorem two_boxes_in_case :
  boxes_in_case 12 6 = 2 := by
  sorry

end two_boxes_in_case_l1076_107635


namespace percentage_increase_is_20_percent_l1076_107611

/-- Represents the number of units in each building --/
structure BuildingUnits where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Calculates the percentage increase from the second to the third building --/
def percentageIncrease (units : BuildingUnits) : ℚ :=
  (units.third - units.second : ℚ) / units.second * 100

/-- The main theorem stating the percentage increase is 20% --/
theorem percentage_increase_is_20_percent 
  (total : ℕ) 
  (h1 : total = 7520) 
  (units : BuildingUnits) 
  (h2 : units.first = 4000) 
  (h3 : units.second = 2 * units.first / 5) 
  (h4 : total = units.first + units.second + units.third) : 
  percentageIncrease units = 20 := by
  sorry

end percentage_increase_is_20_percent_l1076_107611


namespace candy_remainder_l1076_107687

theorem candy_remainder : (31254389 : ℕ) % 6 = 5 := by sorry

end candy_remainder_l1076_107687


namespace exactly_one_pass_probability_l1076_107680

theorem exactly_one_pass_probability (p : ℝ) (hp : p = 1 / 2) :
  let q := 1 - p
  p * q + q * p = 1 / 2 := by sorry

end exactly_one_pass_probability_l1076_107680


namespace max_point_of_product_l1076_107616

/-- Linear function f(x) -/
def f (x : ℝ) : ℝ := 2 * x + 2

/-- Linear function g(x) -/
def g (x : ℝ) : ℝ := -x - 3

/-- Product function h(x) = f(x) * g(x) -/
def h (x : ℝ) : ℝ := f x * g x

theorem max_point_of_product (x : ℝ) :
  f (-1) = 0 ∧ f 0 = 2 ∧ g 3 = 0 ∧ g 0 = -3 →
  ∃ (max_x : ℝ), max_x = -2 ∧ ∀ y, h y ≤ h max_x :=
sorry

end max_point_of_product_l1076_107616


namespace complex_number_quadrant_l1076_107603

theorem complex_number_quadrant (z : ℂ) (h : z * Complex.I = -2 + Complex.I) :
  0 < z.re ∧ 0 < z.im := by
  sorry

end complex_number_quadrant_l1076_107603


namespace alcohol_dilution_l1076_107632

/-- Proves that mixing 50 ml of 30% alcohol after-shave lotion with 30 ml of pure water
    results in a solution with 18.75% alcohol content. -/
theorem alcohol_dilution (initial_volume : ℝ) (initial_percentage : ℝ) (water_volume : ℝ)
  (h1 : initial_volume = 50)
  (h2 : initial_percentage = 30)
  (h3 : water_volume = 30) :
  let alcohol_volume : ℝ := initial_volume * (initial_percentage / 100)
  let total_volume : ℝ := initial_volume + water_volume
  let new_percentage : ℝ := (alcohol_volume / total_volume) * 100
  new_percentage = 18.75 := by
sorry

end alcohol_dilution_l1076_107632


namespace five_digit_divisible_by_nine_l1076_107626

theorem five_digit_divisible_by_nine :
  ∀ x : ℕ, x < 10 →
  (738 * 10 + x) * 10 + 5 ≡ 0 [MOD 9] ↔ x = 4 := by
  sorry

end five_digit_divisible_by_nine_l1076_107626


namespace frank_bakes_two_trays_per_day_l1076_107650

/-- The number of days Frank bakes cookies -/
def days : ℕ := 6

/-- The number of cookies Frank eats per day -/
def frankEatsPerDay : ℕ := 1

/-- The number of cookies Ted eats on the sixth day -/
def tedEats : ℕ := 4

/-- The number of cookies each tray makes -/
def cookiesPerTray : ℕ := 12

/-- The number of cookies left when Ted leaves -/
def cookiesLeft : ℕ := 134

/-- The number of trays Frank bakes per day -/
def traysPerDay : ℕ := 2

theorem frank_bakes_two_trays_per_day :
  traysPerDay * cookiesPerTray * days - 
  (frankEatsPerDay * days + tedEats) = cookiesLeft := by
  sorry

end frank_bakes_two_trays_per_day_l1076_107650


namespace yellow_or_blue_consecutive_rolls_l1076_107683

/-- A die with 12 sides and specific color distribution -/
structure Die :=
  (sides : Nat)
  (red : Nat)
  (yellow : Nat)
  (blue : Nat)
  (green : Nat)
  (total_eq : sides = red + yellow + blue + green)

/-- The probability of an event occurring -/
def probability (favorable : Nat) (total : Nat) : ℚ :=
  ↑favorable / ↑total

/-- The probability of two independent events both occurring -/
def probability_both (p1 : ℚ) (p2 : ℚ) : ℚ := p1 * p2

theorem yellow_or_blue_consecutive_rolls (d : Die) 
  (h : d.sides = 12 ∧ d.red = 5 ∧ d.yellow = 4 ∧ d.blue = 2 ∧ d.green = 1) : 
  probability_both 
    (probability (d.yellow + d.blue) d.sides) 
    (probability (d.yellow + d.blue) d.sides) = 1/4 := by
  sorry

end yellow_or_blue_consecutive_rolls_l1076_107683


namespace some_number_value_l1076_107691

theorem some_number_value (x : ℝ) :
  1 / 2 + ((2 / 3 * x) + 4) - 8 / 16 = 4.25 → x = 0.375 := by
  sorry

end some_number_value_l1076_107691


namespace smallest_debt_theorem_l1076_107636

/-- The value of a pig in dollars -/
def pig_value : ℕ := 250

/-- The value of a goat in dollars -/
def goat_value : ℕ := 175

/-- The value of a sheep in dollars -/
def sheep_value : ℕ := 125

/-- The smallest positive debt that can be resolved -/
def smallest_resolvable_debt : ℕ := 25

theorem smallest_debt_theorem :
  (∃ (p g s : ℤ), smallest_resolvable_debt = pig_value * p + goat_value * g + sheep_value * s) ∧
  (∀ (d : ℕ), d > 0 ∧ d < smallest_resolvable_debt →
    ¬∃ (p g s : ℤ), d = pig_value * p + goat_value * g + sheep_value * s) :=
by sorry

end smallest_debt_theorem_l1076_107636


namespace largest_d_inequality_d_satisfies_inequality_d_is_largest_l1076_107670

theorem largest_d_inequality (d : ℝ) : 
  (d > 0 ∧ 
   ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → 
   Real.sqrt (x^2 + y^2) + d * |x - y| ≤ Real.sqrt (2 * (x + y))) → 
  d ≤ 1 / Real.sqrt 2 :=
by sorry

theorem d_satisfies_inequality : 
  ∀ (x y : ℝ), x ≥ 0 → y ≥ 0 → 
  Real.sqrt (x^2 + y^2) + (1 / Real.sqrt 2) * |x - y| ≤ Real.sqrt (2 * (x + y)) :=
by sorry

theorem d_is_largest : 
  ∀ (d : ℝ), d > 1 / Real.sqrt 2 → 
  ∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ 
  Real.sqrt (x^2 + y^2) + d * |x - y| > Real.sqrt (2 * (x + y)) :=
by sorry

end largest_d_inequality_d_satisfies_inequality_d_is_largest_l1076_107670


namespace no_integer_pairs_l1076_107615

theorem no_integer_pairs : ¬∃ (x y : ℤ), 0 < x ∧ x < y ∧ Real.sqrt 2500 = Real.sqrt x + 2 * Real.sqrt y := by
  sorry

end no_integer_pairs_l1076_107615


namespace bracelet_cost_calculation_josh_bracelet_cost_l1076_107641

theorem bracelet_cost_calculation (bracelet_price : ℝ) (num_bracelets : ℕ) (cookie_cost : ℝ) (money_left : ℝ) : ℝ :=
  let total_earned := bracelet_price * num_bracelets
  let total_after_cookies := cookie_cost + money_left
  let supply_cost := (total_earned - total_after_cookies) / num_bracelets
  supply_cost

theorem josh_bracelet_cost :
  bracelet_cost_calculation 1.5 12 3 3 = 1 := by sorry

end bracelet_cost_calculation_josh_bracelet_cost_l1076_107641


namespace bisection_arbitrary_precision_l1076_107655

/-- Represents a continuous function on a closed interval -/
def ContinuousFunction (a b : ℝ) := ℝ → ℝ

/-- Represents the bisection method applied to a function -/
def BisectionMethod (f : ContinuousFunction a b) (ε : ℝ) : ℝ := sorry

/-- Theorem stating that the bisection method can achieve arbitrary precision -/
theorem bisection_arbitrary_precision 
  (f : ContinuousFunction a b) 
  (h₁ : a < b) 
  (h₂ : f a * f b ≤ 0) 
  (ε : ℝ) 
  (h₃ : ε > 0) :
  ∃ x : ℝ, |f x| < ε ∧ x ∈ Set.Icc a b :=
sorry

end bisection_arbitrary_precision_l1076_107655


namespace book_pages_calculation_l1076_107607

/-- Represents the number of pages read in a book over a week -/
def BookPages : ℕ → ℕ → ℕ → ℕ → ℕ := λ d1 d2 d3 d4 =>
  d1 * 30 + d2 * 50 + d4

theorem book_pages_calculation :
  BookPages 2 4 1 70 = 330 := by
  sorry

end book_pages_calculation_l1076_107607


namespace greatest_divisor_with_remainders_l1076_107699

theorem greatest_divisor_with_remainders : ∃ (n : ℕ), 
  n > 0 ∧
  n = Nat.gcd (1557 - 7) (Nat.gcd (2037 - 5) (2765 - 9)) ∧
  1557 % n = 7 ∧
  2037 % n = 5 ∧
  2765 % n = 9 ∧
  ∀ (m : ℕ), m > n → 
    (1557 % m = 7 ∧ 2037 % m = 5 ∧ 2765 % m = 9) → False :=
by sorry

end greatest_divisor_with_remainders_l1076_107699


namespace f_properties_l1076_107613

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 4 + 2 * Real.sin x * Real.cos x - Real.cos x ^ 4

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∀ (x : ℝ), f x ≥ -2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi / 2) → ∀ (y : ℝ), y ∈ Set.Icc 0 (Real.pi / 2) → x < y → f x < f y) ∧
  (∀ (x : ℝ), x ∈ Set.Ioc (Real.pi / 2) Real.pi → ∀ (y : ℝ), y ∈ Set.Ioc (Real.pi / 2) Real.pi → x < y → f x > f y) :=
by sorry

end f_properties_l1076_107613


namespace product_218_5_base9_l1076_107689

/-- Convert a base-9 number to base-10 --/
def base9ToBase10 (n : ℕ) : ℕ := sorry

/-- Convert a base-10 number to base-9 --/
def base10ToBase9 (n : ℕ) : ℕ := sorry

/-- Multiply two base-9 numbers and return the result in base-9 --/
def multiplyBase9 (a b : ℕ) : ℕ :=
  base10ToBase9 (base9ToBase10 a * base9ToBase10 b)

theorem product_218_5_base9 :
  multiplyBase9 218 5 = 1204 := by sorry

end product_218_5_base9_l1076_107689


namespace sport_drink_water_amount_l1076_107604

/-- Represents the composition of a sport drink -/
structure SportDrink where
  flavoringRatio : ℚ
  cornSyrupRatio : ℚ
  waterRatio : ℚ
  cornSyrupOunces : ℚ

/-- Calculates the amount of water in a sport drink -/
def waterAmount (drink : SportDrink) : ℚ :=
  (drink.waterRatio / drink.cornSyrupRatio) * drink.cornSyrupOunces

/-- Theorem stating the amount of water in the sport drink -/
theorem sport_drink_water_amount 
  (drink : SportDrink)
  (h1 : drink.flavoringRatio = 1)
  (h2 : drink.cornSyrupRatio = 4)
  (h3 : drink.waterRatio = 60)
  (h4 : drink.cornSyrupOunces = 7) :
  waterAmount drink = 105 := by
  sorry

#check sport_drink_water_amount

end sport_drink_water_amount_l1076_107604


namespace triangle_inequalities_l1076_107637

-- Define the points and lengths
variables (P Q R S : ℝ × ℝ) (a b c : ℝ)

-- Define the conditions
def collinear (P Q R S : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ t₃ : ℝ, 0 < t₁ ∧ t₁ < t₂ ∧ t₂ < t₃ ∧
  Q = P + t₁ • (S - P) ∧
  R = P + t₂ • (S - P) ∧
  S = P + t₃ • (S - P)

def segment_lengths (P Q R S : ℝ × ℝ) (a b c : ℝ) : Prop :=
  dist P Q = a ∧ dist P R = b ∧ dist P S = c

def can_form_triangle (a b c : ℝ) : Prop :=
  a + (b - a) > c - b ∧
  (b - a) + (c - b) > a ∧
  a + (c - b) > b - a

-- State the theorem
theorem triangle_inequalities
  (h_collinear : collinear P Q R S)
  (h_lengths : segment_lengths P Q R S a b c)
  (h_triangle : can_form_triangle a b c) :
  a < c / 3 ∧ b < 2 * a + c :=
sorry

end triangle_inequalities_l1076_107637


namespace no_three_digit_odd_sum_30_l1076_107624

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem no_three_digit_odd_sum_30 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 30 ∧ n % 2 = 1 :=
sorry

end no_three_digit_odd_sum_30_l1076_107624


namespace eldest_child_age_l1076_107652

-- Define the number of children
def num_children : ℕ := 8

-- Define the age difference between consecutive children
def age_difference : ℕ := 3

-- Define the total sum of ages
def total_age_sum : ℕ := 100

-- Theorem statement
theorem eldest_child_age :
  ∃ (youngest_age : ℕ),
    (youngest_age + (num_children - 1) * age_difference) +
    (youngest_age + (num_children - 2) * age_difference) +
    (youngest_age + (num_children - 3) * age_difference) +
    (youngest_age + (num_children - 4) * age_difference) +
    (youngest_age + (num_children - 5) * age_difference) +
    (youngest_age + (num_children - 6) * age_difference) +
    (youngest_age + (num_children - 7) * age_difference) +
    youngest_age = total_age_sum →
    youngest_age + (num_children - 1) * age_difference = 23 :=
by
  sorry

end eldest_child_age_l1076_107652


namespace sufficient_but_not_necessary_l1076_107663

theorem sufficient_but_not_necessary (p q : Prop) :
  -- Part 1: Sufficient condition
  ((p ∧ q) → ¬(¬p)) ∧
  -- Part 2: Not necessary condition
  ∃ (r : Prop), (¬(¬r) ∧ ¬(r ∧ q)) :=
by sorry

end sufficient_but_not_necessary_l1076_107663


namespace arithmetic_contains_geometric_l1076_107618

/-- An arithmetic sequence of positive real numbers -/
def arithmetic_sequence (a₀ d : ℝ) (n : ℕ) : ℝ := a₀ + n • d

/-- A geometric sequence of real numbers -/
def geometric_sequence (b₀ q : ℝ) (n : ℕ) : ℝ := b₀ * q^n

/-- Theorem: If an infinite arithmetic sequence of positive real numbers contains two different
    powers of an integer greater than 1, then it contains an infinite geometric sequence -/
theorem arithmetic_contains_geometric
  (a₀ d : ℝ) (a : ℕ) (h_a : a > 1) 
  (h_pos : ∀ n, arithmetic_sequence a₀ d n > 0)
  (m n : ℕ) (h_mn : m ≠ n)
  (h_power_m : ∃ k₁, arithmetic_sequence a₀ d k₁ = a^m)
  (h_power_n : ∃ k₂, arithmetic_sequence a₀ d k₂ = a^n) :
  ∃ b₀ q : ℝ, ∀ k, ∃ l, arithmetic_sequence a₀ d l = geometric_sequence b₀ q k :=
sorry

end arithmetic_contains_geometric_l1076_107618


namespace sphere_volume_from_cylinder_and_cone_l1076_107669

/-- The volume of a sphere given specific conditions involving a cylinder and cone -/
theorem sphere_volume_from_cylinder_and_cone 
  (r : ℝ) -- radius of the sphere
  (h : ℝ) -- height of the cylinder and cone
  (M : ℝ) -- volume of the cylinder
  (h_eq : h = 2 * r) -- height is twice the radius
  (M_eq : M = π * r^2 * h) -- volume formula for cylinder
  (V_cone : ℝ := (1/3) * π * r^2 * h) -- volume of the cone
  (C : ℝ) -- volume of the sphere
  (vol_eq : M - V_cone = C) -- combined volume equals sphere volume
  : C = (8/3) * π * r^3 := by
  sorry

end sphere_volume_from_cylinder_and_cone_l1076_107669


namespace ralphs_socks_l1076_107649

theorem ralphs_socks (x y z : ℕ) : 
  x + y + z = 12 →  -- Total pairs of socks
  x + 3*y + 4*z = 24 →  -- Total cost
  x ≥ 1 →  -- At least one pair of $1 socks
  y ≥ 1 →  -- At least one pair of $3 socks
  z ≥ 1 →  -- At least one pair of $4 socks
  x = 7  -- Number of $1 socks Ralph bought
  := by sorry

end ralphs_socks_l1076_107649


namespace complex_modulus_one_l1076_107634

theorem complex_modulus_one (z : ℂ) (h : (1 + z) / (1 - z) = Complex.I) : Complex.abs z = 1 := by
  sorry

end complex_modulus_one_l1076_107634


namespace cube_volume_from_face_perimeter_l1076_107622

theorem cube_volume_from_face_perimeter (face_perimeter : ℝ) (h : face_perimeter = 28) :
  (face_perimeter / 4) ^ 3 = 343 := by
  sorry

#check cube_volume_from_face_perimeter

end cube_volume_from_face_perimeter_l1076_107622


namespace lucy_additional_distance_l1076_107621

/-- The length of the field in kilometers -/
def field_length : ℚ := 24

/-- The fraction of the field that Mary ran -/
def mary_fraction : ℚ := 3/8

/-- The fraction of Mary's distance that Edna ran -/
def edna_fraction : ℚ := 2/3

/-- The fraction of Edna's distance that Lucy ran -/
def lucy_fraction : ℚ := 5/6

/-- Mary's running distance in kilometers -/
def mary_distance : ℚ := field_length * mary_fraction

/-- Edna's running distance in kilometers -/
def edna_distance : ℚ := mary_distance * edna_fraction

/-- Lucy's running distance in kilometers -/
def lucy_distance : ℚ := edna_distance * lucy_fraction

theorem lucy_additional_distance :
  mary_distance - lucy_distance = 4 := by
  sorry

end lucy_additional_distance_l1076_107621


namespace parallel_lines_a_value_l1076_107629

/-- Two lines in the plane are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The problem statement -/
theorem parallel_lines_a_value :
  (∀ x y, y = x ↔ 2 * x + a * y = 1) → a = -2 :=
by sorry

end parallel_lines_a_value_l1076_107629


namespace distance_between_mum_and_turbo_l1076_107677

/-- The distance between Usain's mum and Turbo when Usain has run 100 meters -/
theorem distance_between_mum_and_turbo (usain_speed mum_speed turbo_speed : ℝ) : 
  usain_speed = 2 * mum_speed →
  mum_speed = 5 * turbo_speed →
  usain_speed > 0 →
  (100 / usain_speed) * mum_speed - (100 / usain_speed) * turbo_speed = 40 := by
  sorry

end distance_between_mum_and_turbo_l1076_107677


namespace inverse_f_58_l1076_107612

def f (x : ℝ) : ℝ := 2 * x^3 + 4

theorem inverse_f_58 : f⁻¹ 58 = 3 := by sorry

end inverse_f_58_l1076_107612


namespace Q_equals_G_l1076_107698

-- Define the sets Q and G
def Q : Set ℝ := {y | ∃ x, y = x^2 + 1}
def G : Set ℝ := {x | x ≥ 1}

-- Theorem statement
theorem Q_equals_G : Q = G := by sorry

end Q_equals_G_l1076_107698


namespace rectangle_probability_in_n_gon_l1076_107693

theorem rectangle_probability_in_n_gon (n : ℕ) (h1 : Even n) (h2 : n > 4) :
  let P := (3 : ℚ) / ((n - 1) * (n - 3))
  P = (Nat.choose (n / 2) 2 : ℚ) / (Nat.choose n 4 : ℚ) :=
by sorry

end rectangle_probability_in_n_gon_l1076_107693


namespace circle_equation_with_hyperbola_asymptotes_as_tangents_l1076_107667

/-- The standard equation of a circle with center (0,5) and tangents that are the asymptotes of the hyperbola x^2 - y^2 = 1 -/
theorem circle_equation_with_hyperbola_asymptotes_as_tangents :
  ∃ (r : ℝ),
    (∀ (x y : ℝ), x^2 + (y - 5)^2 = r^2 ↔
      (∃ (t : ℝ), (x = t ∧ y = t + 5) ∨ (x = -t ∧ y = -t + 5))) ∧
    r^2 = 16 :=
by sorry

end circle_equation_with_hyperbola_asymptotes_as_tangents_l1076_107667


namespace even_function_sum_l1076_107686

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

theorem even_function_sum (f : ℝ → ℝ) (h_even : is_even_function f) (h_value : f (-1) = 2) :
  f (-1) + f 1 = 4 := by
  sorry

end even_function_sum_l1076_107686


namespace max_gaming_average_l1076_107681

theorem max_gaming_average (wednesday_hours : ℝ) (thursday_hours : ℝ) (tom_hours : ℝ) (fred_hours : ℝ) (additional_time : ℝ) :
  wednesday_hours = 2 →
  thursday_hours = 2 →
  tom_hours = 4 →
  fred_hours = 6 →
  additional_time = 0.5 →
  let total_hours := wednesday_hours + thursday_hours + max tom_hours fred_hours + additional_time
  let days := 3
  let average_hours := total_hours / days
  average_hours = 3.5 := by
sorry

end max_gaming_average_l1076_107681


namespace four_correct_statements_l1076_107631

theorem four_correct_statements : 
  (∀ x : ℝ, Irrational x → ¬ (∃ q : ℚ, x = ↑q)) ∧ 
  ({x : ℝ | x^2 = 4} = {2, -2}) ∧
  ({x : ℝ | x^3 = x} = {-1, 0, 1}) ∧
  (∀ x : ℝ, ∃! p : ℝ, p = x) := by
  sorry

end four_correct_statements_l1076_107631


namespace two_inequalities_for_real_numbers_l1076_107694

theorem two_inequalities_for_real_numbers (a b c : ℝ) : 
  (a * b / c^2 + b * c / a^2 + a * c / b^2 ≥ a / c + b / a + c / b) ∧
  (a^2 / b^2 + b^2 / c^2 + c^2 / a^2 ≥ a / b + b / c + c / a) := by
  sorry

end two_inequalities_for_real_numbers_l1076_107694


namespace half_volume_convex_hull_cube_l1076_107684

theorem half_volume_convex_hull_cube : ∃ a : ℝ, 0 < a ∧ a < 1 ∧ 
  2 * (a^3 + (1-a)^3) = 3/4 := by
  sorry

end half_volume_convex_hull_cube_l1076_107684


namespace figure_to_square_l1076_107697

/-- A figure that can be cut into three parts -/
structure Figure where
  area : ℕ

/-- Proves that a figure with an area of 57 unit squares can be assembled into a square -/
theorem figure_to_square (f : Figure) (h : f.area = 57) : 
  ∃ (s : ℝ), s^2 = f.area := by
  sorry

end figure_to_square_l1076_107697


namespace smallest_divisor_l1076_107627

theorem smallest_divisor (N D : ℕ) (q1 q2 k : ℕ) : 
  N = D * q1 + 75 →
  N = 37 * q2 + 1 →
  D > 75 →
  D = 37 * k + 38 →
  112 ≤ D :=
by sorry

end smallest_divisor_l1076_107627


namespace smallest_prime_divisor_of_sum_l1076_107682

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : ℕ), p.Prime ∧ p ∣ (3^19 + 11^13) ∧ ∀ (q : ℕ), q.Prime → q ∣ (3^19 + 11^13) → p ≤ q :=
by
  sorry

end smallest_prime_divisor_of_sum_l1076_107682


namespace hyperbola_equation_l1076_107661

-- Define the eccentricity
def e : ℝ := 2

-- Define the ellipse parameters
def a_ellipse : ℝ := 4
def b_ellipse : ℝ := 3

-- Define the hyperbola equations
def horizontal_hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 48 = 1
def vertical_hyperbola (x y : ℝ) : Prop := y^2 / 9 - x^2 / 27 = 1

-- Theorem statement
theorem hyperbola_equation :
  ∀ x y : ℝ,
  (x^2 / a_ellipse^2 + y^2 / b_ellipse^2 = 1) →
  (∃ a b : ℝ, (a = a_ellipse ∧ b^2 = a^2 * (e^2 - 1)) ∨ (a = b_ellipse ∧ b^2 = a^2 * (e^2 - 1))) →
  (horizontal_hyperbola x y ∨ vertical_hyperbola x y) :=
by sorry

end hyperbola_equation_l1076_107661


namespace digital_city_activities_l1076_107640

-- Define the concept of a digital city
structure DigitalCity where
  is_part_of_digital_earth : Bool

-- Define possible activities in a digital city
inductive DigitalCityActivity
  | DistanceEducation
  | OnlineShopping
  | OnlineMedicalAdvice

-- Define a function that checks if an activity is enabled in a digital city
def is_enabled (city : DigitalCity) (activity : DigitalCityActivity) : Prop :=
  city.is_part_of_digital_earth

-- Theorem stating that digital cities enable specific activities
theorem digital_city_activities (city : DigitalCity) 
  (h : city.is_part_of_digital_earth = true) : 
  (is_enabled city DigitalCityActivity.DistanceEducation) ∧
  (is_enabled city DigitalCityActivity.OnlineShopping) ∧
  (is_enabled city DigitalCityActivity.OnlineMedicalAdvice) :=
by
  sorry


end digital_city_activities_l1076_107640


namespace leopard_arrangement_count_l1076_107606

/-- The number of snow leopards -/
def total_leopards : ℕ := 9

/-- The number of leopards with special placement requirements -/
def special_leopards : ℕ := 3

/-- The number of ways to arrange the shortest two leopards at the ends -/
def shortest_arrangements : ℕ := 2

/-- The number of ways to place the tallest leopard in the middle -/
def tallest_arrangement : ℕ := 1

/-- The number of remaining leopards to be arranged -/
def remaining_leopards : ℕ := total_leopards - special_leopards

/-- Theorem: The number of ways to arrange the leopards is 1440 -/
theorem leopard_arrangement_count : 
  shortest_arrangements * tallest_arrangement * Nat.factorial remaining_leopards = 1440 := by
  sorry

end leopard_arrangement_count_l1076_107606


namespace pure_imaginary_solutions_l1076_107625

theorem pure_imaginary_solutions :
  let f : ℂ → ℂ := λ x => x^4 - 5*x^3 + 10*x^2 - 50*x - 75
  ∀ x : ℂ, (∃ k : ℝ, x = k * I) → (f x = 0 ↔ x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10) :=
by sorry

end pure_imaginary_solutions_l1076_107625


namespace gcd_of_squares_plus_one_l1076_107619

theorem gcd_of_squares_plus_one (n : ℕ+) : 
  Nat.gcd (n.val^2 + 1) ((n.val + 1)^2 + 1) = 1 :=
by sorry

end gcd_of_squares_plus_one_l1076_107619


namespace fenced_field_area_l1076_107647

/-- A rectangular field with specific fencing requirements -/
structure FencedField where
  length : ℝ
  width : ℝ
  uncovered_side : ℝ
  fencing : ℝ
  uncovered_side_eq : uncovered_side = 20
  fencing_eq : uncovered_side + 2 * width = fencing
  fencing_length : fencing = 88

/-- The area of a rectangular field -/
def field_area (f : FencedField) : ℝ :=
  f.length * f.width

/-- Theorem stating that a field with the given specifications has an area of 680 square feet -/
theorem fenced_field_area (f : FencedField) : field_area f = 680 := by
  sorry

end fenced_field_area_l1076_107647


namespace problem_statement_l1076_107679

theorem problem_statement : (3150 - 3030)^2 / 144 = 100 := by
  sorry

end problem_statement_l1076_107679


namespace consecutive_points_distance_l1076_107644

/-- Given five consecutive points on a straight line, if certain distance conditions are met,
    then the distance between the last two points is 4. -/
theorem consecutive_points_distance (a b c d e : ℝ) : 
  (b - a) + (c - b) + (d - c) + (e - d) = (e - a)  -- Points are consecutive on a line
  → (c - b) = 2 * (d - c)  -- bc = 2 cd
  → (b - a) = 5  -- ab = 5
  → (c - a) = 11  -- ac = 11
  → (e - a) = 18  -- ae = 18
  → (e - d) = 4  -- de = 4
:= by sorry

end consecutive_points_distance_l1076_107644


namespace pure_imaginary_complex_l1076_107658

/-- Given that z = (a^2 - 4) + (a + 2)i is a pure imaginary number,
    prove that (a + i^2015) / (1 + 2i) = -i -/
theorem pure_imaginary_complex (a : ℝ) :
  (Complex.I : ℂ)^2015 = -Complex.I →
  (a^2 - 4 : ℂ) = 0 →
  (a + 2 : ℂ) ≠ 0 →
  (a + Complex.I^2015) / (1 + 2 * Complex.I) = -Complex.I :=
by sorry

end pure_imaginary_complex_l1076_107658


namespace sequence_problem_l1076_107645

def arithmetic_sequence (a b : ℕ) (n : ℕ) : ℕ := a + b * (n - 1)

def geometric_sequence (b a : ℕ) (n : ℕ) : ℕ := b * a^(n - 1)

theorem sequence_problem (a b : ℕ) 
  (h1 : a > 1) 
  (h2 : b > 1) 
  (h3 : a < b) 
  (h4 : b * a < arithmetic_sequence a b 3) 
  (h5 : ∀ n : ℕ, n > 0 → ∃ m : ℕ, m > 0 ∧ geometric_sequence b a n = arithmetic_sequence a b m + 3) :
  a = 2 ∧ ∀ n : ℕ, arithmetic_sequence a b n = 5 * n - 3 :=
sorry

end sequence_problem_l1076_107645


namespace chloe_boxes_l1076_107659

/-- The number of scarves in each box -/
def scarves_per_box : ℕ := 2

/-- The number of mittens in each box -/
def mittens_per_box : ℕ := 6

/-- The total number of winter clothing pieces Chloe has -/
def total_clothing : ℕ := 32

/-- The number of boxes Chloe found -/
def boxes : ℕ := total_clothing / (scarves_per_box + mittens_per_box)

theorem chloe_boxes : boxes = 4 := by
  sorry

end chloe_boxes_l1076_107659


namespace passengers_in_first_class_l1076_107671

theorem passengers_in_first_class (total_passengers : ℕ) 
  (women_percentage : ℚ) (men_percentage : ℚ)
  (women_first_class_percentage : ℚ) (men_first_class_percentage : ℚ)
  (h1 : total_passengers = 300)
  (h2 : women_percentage = 1/2)
  (h3 : men_percentage = 1/2)
  (h4 : women_first_class_percentage = 1/5)
  (h5 : men_first_class_percentage = 3/20) :
  ⌈(total_passengers : ℚ) * women_percentage * women_first_class_percentage + 
   (total_passengers : ℚ) * men_percentage * men_first_class_percentage⌉ = 53 :=
by sorry

end passengers_in_first_class_l1076_107671


namespace union_of_sets_l1076_107665

def A (a : ℝ) : Set ℝ := {-1, a}
def B (a b : ℝ) : Set ℝ := {2^a, b}

theorem union_of_sets (a b : ℝ) :
  (A a) ∩ (B a b) = {1} → (A a) ∪ (B a b) = {-1, 1, 2} := by
  sorry

end union_of_sets_l1076_107665


namespace count_multiples_of_five_l1076_107662

def d₁ (a : ℕ) : ℕ := a^2 + 2^a + a * 2^((a + 1)/2)
def d₂ (a : ℕ) : ℕ := a^2 + 2^a - a * 2^((a + 1)/2)

theorem count_multiples_of_five :
  ∃ (S : Finset ℕ), S.card = 101 ∧
    (∀ a ∈ S, 1 ≤ a ∧ a ≤ 251 ∧ (d₁ a * d₂ a) % 5 = 0) ∧
    (∀ a : ℕ, 1 ≤ a ∧ a ≤ 251 ∧ (d₁ a * d₂ a) % 5 = 0 → a ∈ S) :=
by sorry

end count_multiples_of_five_l1076_107662


namespace factorization_1_factorization_2_factorization_3_l1076_107654

-- Problem 1
theorem factorization_1 (a : ℝ) : 3*a^3 - 6*a^2 + 3*a = 3*a*(a - 1)^2 := by sorry

-- Problem 2
theorem factorization_2 (a b x y : ℝ) : a^2*(x - y) + b^2*(y - x) = (x - y)*(a - b)*(a + b) := by sorry

-- Problem 3
theorem factorization_3 (a b : ℝ) : 16*(a + b)^2 - 9*(a - b)^2 = (a + 7*b)*(7*a + b) := by sorry

end factorization_1_factorization_2_factorization_3_l1076_107654


namespace number_of_arrangements_l1076_107664

/-- The number of applicants --/
def num_applicants : ℕ := 5

/-- The number of positions to be filled --/
def num_positions : ℕ := 3

/-- The number of students to be selected --/
def num_selected : ℕ := 3

/-- Function to calculate the number of arrangements --/
def calculate_arrangements (n : ℕ) (r : ℕ) : ℕ :=
  if r > n then 0
  else Nat.choose n r

/-- Theorem stating the number of arrangements --/
theorem number_of_arrangements :
  calculate_arrangements num_applicants num_selected +
  calculate_arrangements (num_applicants - 1) num_selected = 16 :=
sorry

end number_of_arrangements_l1076_107664


namespace accountant_total_amount_l1076_107617

/-- Calculates the total amount given to the accountant for festival allowance --/
def festival_allowance_total (staff_count : ℕ) (daily_rate : ℕ) (days : ℕ) (petty_cash : ℕ) : ℕ :=
  staff_count * daily_rate * days + petty_cash

/-- Theorem stating the total amount given to the accountant --/
theorem accountant_total_amount :
  festival_allowance_total 20 100 30 1000 = 61000 := by
  sorry

end accountant_total_amount_l1076_107617


namespace fixed_point_of_line_l1076_107675

/-- The line equation mx - y + 2m + 1 = 0 passes through the point (-2, 1) for all values of m. -/
theorem fixed_point_of_line (m : ℝ) : m * (-2) - 1 + 2 * m + 1 = 0 := by
  sorry

end fixed_point_of_line_l1076_107675


namespace parabola_shift_l1076_107646

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (h : ℝ) : Parabola :=
  { a := p.a,
    b := p.b - 2 * p.a * h,
    c := p.c + p.a * h^2 + p.b * h }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (v : ℝ) : Parabola :=
  { a := p.a,
    b := p.b,
    c := p.c + v }

/-- The original parabola y = x^2 + 2 -/
def original : Parabola :=
  { a := 1,
    b := 0,
    c := 2 }

theorem parabola_shift :
  let p1 := shift_horizontal original 1
  let p2 := shift_vertical p1 (-1)
  p2 = { a := 1, b := 2, c := 1 } :=
by sorry

end parabola_shift_l1076_107646


namespace student_polynomial_correction_l1076_107602

/-- Given a polynomial P(x) that satisfies P(x) - 3x^2 = x^2 - 4x + 1,
    prove that P(x) * (-3x^2) = -12x^4 + 12x^3 - 3x^2 -/
theorem student_polynomial_correction (P : ℝ → ℝ) :
  (∀ x, P x - 3 * x^2 = x^2 - 4 * x + 1) →
  (∀ x, P x * (-3 * x^2) = -12 * x^4 + 12 * x^3 - 3 * x^2) :=
by sorry

end student_polynomial_correction_l1076_107602


namespace sequence_sum_l1076_107642

theorem sequence_sum (A B C D E F G H : ℤ) : 
  C = 3 ∧ 
  A + B + C = 27 ∧
  B + C + D = 27 ∧
  C + D + E = 27 ∧
  D + E + F = 27 ∧
  E + F + G = 27 ∧
  F + G + H = 27 →
  A + H = 27 := by sorry

end sequence_sum_l1076_107642


namespace age_difference_l1076_107657

theorem age_difference (A B C : ℕ) (h1 : C = A - 18) : A + B - (B + C) = 18 := by
  sorry

end age_difference_l1076_107657


namespace mortgage_payment_l1076_107638

theorem mortgage_payment (P : ℝ) : 
  (P * (1 - 3^10) / (1 - 3) = 2952400) → P = 100 := by
  sorry

end mortgage_payment_l1076_107638


namespace second_rate_is_five_percent_l1076_107674

-- Define the total amount, first part, and interest rates
def total_amount : ℚ := 3200
def first_part : ℚ := 800
def first_rate : ℚ := 3 / 100
def total_interest : ℚ := 144

-- Define the second part
def second_part : ℚ := total_amount - first_part

-- Define the interest from the first part
def interest_first : ℚ := first_part * first_rate

-- Define the interest from the second part
def interest_second : ℚ := total_interest - interest_first

-- Define the interest rate of the second part
def second_rate : ℚ := interest_second / second_part

-- Theorem to prove
theorem second_rate_is_five_percent : second_rate = 5 / 100 := by
  sorry

end second_rate_is_five_percent_l1076_107674


namespace multiplication_sum_l1076_107620

theorem multiplication_sum (a b : ℕ) : 
  a ≤ 9 → b ≤ 9 → (30 * a + a) * (10 * b + 4) = 126 → a + b = 7 := by
  sorry

end multiplication_sum_l1076_107620


namespace ellipse_equation_l1076_107690

/-- An ellipse with center at the origin, foci on the x-axis, and point P(2, √3) on the ellipse. 
    The distances |PF₁|, |F₁F₂|, and |PF₂| form an arithmetic sequence. -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_positive : 0 < b ∧ b < a
  h_foci : c^2 = a^2 - b^2
  h_point_on_ellipse : 4 / a^2 + 3 / b^2 = 1
  h_arithmetic_sequence : ∃ (d : ℝ), |2 - c| + d = 2*c ∧ 2*c + d = |2 + c|

/-- The equation of the ellipse is x²/8 + y²/6 = 1 -/
theorem ellipse_equation (e : Ellipse) : e.a^2 = 8 ∧ e.b^2 = 6 := by
  sorry

end ellipse_equation_l1076_107690


namespace triangle_height_inequality_l1076_107630

/-- For a triangle with side lengths a ≤ b ≤ c, heights h_a, h_b, h_c, 
    circumradius R, and semiperimeter p, the following inequality holds. -/
theorem triangle_height_inequality 
  (a b c : ℝ) (h_a h_b h_c R p : ℝ) 
  (h_order : a ≤ b ∧ b ≤ c) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ p > 0) 
  (h_heights : h_a = 2 * (p - a) * (p - b) * (p - c) / (a * b * c) ∧ 
               h_b = 2 * (p - a) * (p - b) * (p - c) / (a * b * c) ∧ 
               h_c = 2 * (p - a) * (p - b) * (p - c) / (a * b * c))
  (h_semiperimeter : p = (a + b + c) / 2)
  (h_circumradius : R = a * b * c / (4 * (p - a) * (p - b) * (p - c))) :
  h_a + h_b + h_c ≤ 3 * b * (a^2 + a*c + c^2) / (4 * p * R) :=
sorry

end triangle_height_inequality_l1076_107630


namespace family_can_purchase_in_fourth_month_l1076_107610

/-- Represents the family's financial situation and purchase plan -/
structure Family where
  monthly_income : ℕ
  monthly_expenses : ℕ
  initial_savings : ℕ
  furniture_cost : ℕ

/-- Calculates the month when the family can make the purchase -/
def purchase_month (f : Family) : ℕ :=
  let monthly_savings := f.monthly_income - f.monthly_expenses
  let additional_required := f.furniture_cost - f.initial_savings
  (additional_required + monthly_savings - 1) / monthly_savings + 1

/-- Theorem stating that the family can make the purchase in the 4th month -/
theorem family_can_purchase_in_fourth_month (f : Family) 
  (h1 : f.monthly_income = 150000)
  (h2 : f.monthly_expenses = 115000)
  (h3 : f.initial_savings = 45000)
  (h4 : f.furniture_cost = 127000) :
  purchase_month f = 4 := by
  sorry

#eval purchase_month { 
  monthly_income := 150000, 
  monthly_expenses := 115000, 
  initial_savings := 45000, 
  furniture_cost := 127000 
}

end family_can_purchase_in_fourth_month_l1076_107610


namespace quadratic_inequality_implications_l1076_107651

theorem quadratic_inequality_implications 
  (a b c t : ℝ) 
  (h1 : t > 1) 
  (h2 : ∀ x : ℝ, ax^2 + b*x + c > 0 ↔ 1 < x ∧ x < t) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a*x₁^2 + (a-b)*x₁ - c = 0 ∧ a*x₂^2 + (a-b)*x₂ - c = 0) ∧
  (∀ x₁ x₂ : ℝ, a*x₁^2 + (a-b)*x₁ - c = 0 → a*x₂^2 + (a-b)*x₂ - c = 0 → |x₂ - x₁| > Real.sqrt 13) :=
by sorry

end quadratic_inequality_implications_l1076_107651


namespace ultra_savings_interest_theorem_l1076_107688

/-- Represents the Ultra Savings Account investment scenario -/
structure UltraSavingsAccount where
  principal : ℝ
  rate : ℝ
  years : ℕ

/-- Calculates the final balance after compound interest -/
def finalBalance (account : UltraSavingsAccount) : ℝ :=
  account.principal * (1 + account.rate) ^ account.years

/-- Calculates the interest earned -/
def interestEarned (account : UltraSavingsAccount) : ℝ :=
  finalBalance account - account.principal

/-- Theorem stating that the interest earned is approximately $328.49 -/
theorem ultra_savings_interest_theorem (account : UltraSavingsAccount) 
  (h1 : account.principal = 1500)
  (h2 : account.rate = 0.02)
  (h3 : account.years = 10) : 
  ∃ ε > 0, |interestEarned account - 328.49| < ε :=
sorry

end ultra_savings_interest_theorem_l1076_107688


namespace cubic_sum_inequality_l1076_107608

theorem cubic_sum_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a^3 + b^3 + a + b ≥ 4*a*b := by
  sorry

end cubic_sum_inequality_l1076_107608


namespace frog_count_l1076_107605

theorem frog_count : ∀ (N : ℕ), 
  (∃ (T : ℝ), T > 0 ∧
    50 * (0.3 * T / 50) ≤ 0.43 * T / (N - 94) ∧
    0.43 * T / (N - 94) ≤ 44 * (0.27 * T / 44) ∧
    N > 94)
  → N = 165 := by
sorry

end frog_count_l1076_107605


namespace unique_seating_arrangement_l1076_107633

-- Define the types of representatives
inductive Representative
| Martian
| Venusian
| Earthling

-- Define the seating arrangement as a function from chair number to representative
def SeatingArrangement := Fin 10 → Representative

-- Define the rules for valid seating arrangements
def is_valid_arrangement (arr : SeatingArrangement) : Prop :=
  -- Martian must occupy chair 1
  arr 0 = Representative.Martian ∧
  -- Earthling must occupy chair 10
  arr 9 = Representative.Earthling ∧
  -- Representatives must be arranged in clockwise order: Martian, Venusian, Earthling, repeating
  (∀ i : Fin 10, arr i = Representative.Martian → arr ((i + 1) % 10) = Representative.Venusian) ∧
  (∀ i : Fin 10, arr i = Representative.Venusian → arr ((i + 1) % 10) = Representative.Earthling) ∧
  (∀ i : Fin 10, arr i = Representative.Earthling → arr ((i + 1) % 10) = Representative.Martian)

-- Theorem stating that there is exactly one valid seating arrangement
theorem unique_seating_arrangement :
  ∃! arr : SeatingArrangement, is_valid_arrangement arr :=
sorry

end unique_seating_arrangement_l1076_107633


namespace distance_between_shores_is_600_l1076_107678

/-- Represents the distance between two shores --/
def distance_between_shores : ℝ := sorry

/-- Represents the distance of the first meeting point from shore A --/
def first_meeting_distance : ℝ := 500

/-- Represents the distance of the second meeting point from shore B --/
def second_meeting_distance : ℝ := 300

/-- Theorem stating that the distance between shores A and B is 600 yards --/
theorem distance_between_shores_is_600 :
  distance_between_shores = 600 :=
sorry

end distance_between_shores_is_600_l1076_107678


namespace expression_values_l1076_107601

theorem expression_values (m n : ℕ) (h : m * n ≠ 1) :
  let expr := (m^2 + m*n + n^2) / (m*n - 1)
  expr ∈ ({0, 4, 7} : Set ℕ) :=
sorry

end expression_values_l1076_107601


namespace sequence_to_zero_l1076_107666

/-- A transformation that applies |x - α| to each element of a sequence -/
def transform (s : List ℝ) (α : ℝ) : List ℝ :=
  s.map (fun x => |x - α|)

/-- Predicate to check if all elements in a list are zero -/
def all_zero (s : List ℝ) : Prop :=
  s.all (fun x => x = 0)

theorem sequence_to_zero (n : ℕ) :
  ∀ (s : List ℝ), s.length = n →
  (∃ (transformations : List ℝ),
    transformations.length = n ∧
    all_zero (transformations.foldl transform s)) ∧
  (∀ (transformations : List ℝ),
    transformations.length < n →
    ¬ all_zero (transformations.foldl transform s)) :=
sorry

end sequence_to_zero_l1076_107666


namespace translation_equivalence_l1076_107695

noncomputable def original_function (x : ℝ) : ℝ :=
  Real.sin (2 * x + Real.pi / 6)

noncomputable def translated_function (x : ℝ) : ℝ :=
  original_function (x + Real.pi / 6)

theorem translation_equivalence :
  ∀ x : ℝ, translated_function x = Real.cos (2 * x) := by
  sorry

end translation_equivalence_l1076_107695


namespace triangle_third_side_exists_l1076_107676

theorem triangle_third_side_exists : ∃ x : ℕ, 
  3 ≤ x ∧ x ≤ 7 ∧ 
  (x + 3 > 5) ∧ (x + 5 > 3) ∧ (3 + 5 > x) ∧
  (x > 5 - 3) ∧ (x < 5 + 3) := by
  sorry

end triangle_third_side_exists_l1076_107676


namespace min_slope_and_sum_reciprocals_l1076_107672

noncomputable section

def f (x : ℝ) := x^3 - x^2 + (2 * Real.sqrt 2 - 3) * x + 3 - 2 * Real.sqrt 2

def f' (x : ℝ) := 3 * x^2 - 2 * x + 2 * Real.sqrt 2 - 3

theorem min_slope_and_sum_reciprocals :
  (∃ (x_min : ℝ), ∀ (x : ℝ), f' x_min ≤ f' x ∧ f' x_min = 2 * Real.sqrt 2 - 10 / 3) ∧
  (∃ (x₁ x₂ x₃ : ℝ), f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ 
    1 / f' x₁ + 1 / f' x₂ + 1 / f' x₃ = 0) := by
  sorry

end

end min_slope_and_sum_reciprocals_l1076_107672


namespace sum_of_two_numbers_l1076_107673

theorem sum_of_two_numbers (s l : ℝ) : 
  s = 3.5 →
  l = 3 * s →
  s + l = 14 :=
by sorry

end sum_of_two_numbers_l1076_107673


namespace complex_expression_equality_l1076_107614

theorem complex_expression_equality : ∀ (i : ℂ), i^2 = -1 →
  (2 + i) / (1 - i) - (1 - i) = -1/2 + 5/2 * i := by
  sorry

end complex_expression_equality_l1076_107614


namespace c_rent_share_l1076_107692

/-- Represents the usage of the pasture by a person -/
structure Usage where
  oxen : ℕ
  months : ℕ

/-- Calculates the ox-months for a given usage -/
def oxMonths (u : Usage) : ℕ := u.oxen * u.months

/-- Represents the rental situation of the pasture -/
structure PastureRental where
  a : Usage
  b : Usage
  c : Usage
  totalRent : ℕ

/-- Calculates the total ox-months for all users -/
def totalOxMonths (r : PastureRental) : ℕ :=
  oxMonths r.a + oxMonths r.b + oxMonths r.c

/-- Calculates the share of rent for a given usage -/
def rentShare (r : PastureRental) (u : Usage) : ℚ :=
  (oxMonths u : ℚ) / (totalOxMonths r : ℚ) * (r.totalRent : ℚ)

theorem c_rent_share (r : PastureRental) : 
  r.a = { oxen := 10, months := 7 } →
  r.b = { oxen := 12, months := 5 } →
  r.c = { oxen := 15, months := 3 } →
  r.totalRent = 245 →
  rentShare r r.c = 63 := by
  sorry

end c_rent_share_l1076_107692


namespace specific_trapezoid_dimensions_l1076_107648

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  -- The area of the trapezoid
  area : ℝ
  -- The height of the trapezoid
  height : ℝ
  -- The length of one parallel side (shorter)
  base_short : ℝ
  -- The length of the other parallel side (longer)
  base_long : ℝ
  -- The length of the non-parallel sides (legs)
  leg : ℝ
  -- The trapezoid is isosceles
  isosceles : True
  -- The lines containing the legs intersect at a right angle
  right_angle_intersection : True
  -- The area is calculated correctly
  area_eq : area = (base_short + base_long) * height / 2

/-- Theorem about a specific isosceles trapezoid -/
theorem specific_trapezoid_dimensions :
  ∃ t : IsoscelesTrapezoid,
    t.area = 12 ∧
    t.height = 2 ∧
    t.base_short = 4 ∧
    t.base_long = 8 ∧
    t.leg = 2 * Real.sqrt 2 := by
  sorry

end specific_trapezoid_dimensions_l1076_107648


namespace men_in_room_l1076_107660

theorem men_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  (initial_men : ℚ) / initial_women = 4 / 5 →
  2 * (initial_women - 3) = 24 →
  initial_men + 2 = 14 := by
sorry

end men_in_room_l1076_107660


namespace read_book_in_six_days_book_structure_l1076_107643

/-- The number of days required to read a book -/
def days_to_read (total_pages : ℕ) (pages_per_day : ℕ) : ℕ :=
  total_pages / pages_per_day

/-- Theorem: It takes 6 days to read a 612-page book at 102 pages per day -/
theorem read_book_in_six_days :
  days_to_read 612 102 = 6 := by
  sorry

/-- The book has 24 chapters with pages equally distributed -/
def pages_per_chapter (total_pages : ℕ) (num_chapters : ℕ) : ℕ :=
  total_pages / num_chapters

/-- The book has 612 pages and 24 chapters -/
theorem book_structure :
  pages_per_chapter 612 24 = 612 / 24 := by
  sorry

end read_book_in_six_days_book_structure_l1076_107643
