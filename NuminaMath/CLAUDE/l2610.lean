import Mathlib

namespace power_product_equals_l2610_261078

theorem power_product_equals : (3 : ℕ)^6 * (4 : ℕ)^6 = 2985984 := by sorry

end power_product_equals_l2610_261078


namespace discount_rates_sum_l2610_261037

/-- The discount rate for Fox jeans -/
def fox_discount_rate : ℝ := sorry

/-- The discount rate for Pony jeans -/
def pony_discount_rate : ℝ := 0.1

/-- The regular price of Fox jeans -/
def fox_regular_price : ℝ := 15

/-- The regular price of Pony jeans -/
def pony_regular_price : ℝ := 18

/-- The number of Fox jeans purchased -/
def fox_quantity : ℕ := 3

/-- The number of Pony jeans purchased -/
def pony_quantity : ℕ := 2

/-- The total savings from the purchase -/
def total_savings : ℝ := 9

theorem discount_rates_sum :
  fox_discount_rate + pony_discount_rate = 0.22 :=
by
  have h1 : fox_quantity * fox_regular_price * fox_discount_rate +
            pony_quantity * pony_regular_price * pony_discount_rate = total_savings :=
    by sorry
  sorry

end discount_rates_sum_l2610_261037


namespace unique_solution_l2610_261069

def is_valid_number (α β : ℕ) : Prop :=
  0 ≤ α ∧ α ≤ 9 ∧ 0 ≤ β ∧ β ≤ 9

def number_value (α β : ℕ) : ℕ :=
  62000000 + α * 10000 + β * 1000 + 427

theorem unique_solution (α β : ℕ) :
  is_valid_number α β →
  (number_value α β) % 99 = 0 →
  α = 2 ∧ β = 4 := by
  sorry

end unique_solution_l2610_261069


namespace solve_triangle_l2610_261007

noncomputable def triangle_problem (A B C : ℝ) (a b c : ℝ) : Prop :=
  let S := (1/2) * a * b * Real.sin C
  (a = 3) ∧
  (Real.cos A = Real.sqrt 6 / 3) ∧
  (B = A + Real.pi / 2) →
  (b = 3 * Real.sqrt 2) ∧
  (S = (3/2) * Real.sqrt 2)

theorem solve_triangle : ∀ (A B C : ℝ) (a b c : ℝ),
  triangle_problem A B C a b c :=
by
  sorry

end solve_triangle_l2610_261007


namespace sphere_with_cylindrical_hole_volume_l2610_261081

theorem sphere_with_cylindrical_hole_volume :
  let R : ℝ := Real.sqrt 3
  let sphere_volume := (4 / 3) * Real.pi * R^3
  let cylinder_radius := R / 2
  let cylinder_height := R * Real.sqrt 3
  let cylinder_volume := Real.pi * cylinder_radius^2 * cylinder_height
  let spherical_cap_height := R * (2 - Real.sqrt 3) / 2
  let spherical_cap_volume := (Real.pi * spherical_cap_height^2 * (3 * R - spherical_cap_height)) / 3
  let remaining_volume := sphere_volume - cylinder_volume - 2 * spherical_cap_volume
  remaining_volume = (9 * Real.pi) / 2 := by
sorry


end sphere_with_cylindrical_hole_volume_l2610_261081


namespace polynomial_equality_l2610_261012

theorem polynomial_equality (x : ℝ) (h : 3 * x^3 - x = 1) :
  9 * x^4 + 12 * x^3 - 3 * x^2 - 7 * x + 2001 = 2005 := by
  sorry

end polynomial_equality_l2610_261012


namespace smallest_result_l2610_261061

def S : Finset ℕ := {4, 5, 7, 11, 13, 17}

theorem smallest_result (a b c : ℕ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∃ (x y z : ℕ), x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  (x + y) * z = 48 ∧ (a + b) * c ≥ 48 := by
  sorry

end smallest_result_l2610_261061


namespace ones_digit_of_triple_4567_l2610_261044

def triple_number (n : ℕ) : ℕ := 3 * n

def ones_digit (n : ℕ) : ℕ := n % 10

theorem ones_digit_of_triple_4567 :
  ones_digit (triple_number 4567) = 1 := by sorry

end ones_digit_of_triple_4567_l2610_261044


namespace unique_bagel_count_l2610_261051

def is_valid_purchase (bagels : ℕ) : Prop :=
  ∃ (muffins : ℕ),
    bagels + muffins = 7 ∧
    (90 * bagels + 40 * muffins) % 150 = 0

theorem unique_bagel_count : ∃! b : ℕ, is_valid_purchase b ∧ b = 4 := by
  sorry

end unique_bagel_count_l2610_261051


namespace complex_fraction_equality_l2610_261011

theorem complex_fraction_equality (x y : ℂ) 
  (h : (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = 1) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 41/20 := by
  sorry

end complex_fraction_equality_l2610_261011


namespace paint_mixture_intensity_l2610_261038

theorem paint_mixture_intensity 
  (original_intensity : ℝ) 
  (added_intensity : ℝ) 
  (fraction_replaced : ℝ) :
  original_intensity = 0.5 →
  added_intensity = 0.2 →
  fraction_replaced = 2/3 →
  let new_intensity := (1 - fraction_replaced) * original_intensity + fraction_replaced * added_intensity
  new_intensity = 0.3 := by
sorry


end paint_mixture_intensity_l2610_261038


namespace overlap_rectangle_area_l2610_261083

theorem overlap_rectangle_area : 
  let rect1_width : ℝ := 8
  let rect1_height : ℝ := 10
  let rect2_width : ℝ := 9
  let rect2_height : ℝ := 12
  let overlap_area : ℝ := 37
  let rect1_area : ℝ := rect1_width * rect1_height
  let rect2_area : ℝ := rect2_width * rect2_height
  let grey_area : ℝ := rect2_area - (rect1_area - overlap_area)
  grey_area = 65 := by
sorry

end overlap_rectangle_area_l2610_261083


namespace banana_production_l2610_261074

theorem banana_production (x : ℕ) : 
  x + 10 * x = 99000 → x = 9000 := by
  sorry

end banana_production_l2610_261074


namespace partial_fraction_decomposition_l2610_261060

theorem partial_fraction_decomposition (N₁ N₂ : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (60 * x - 46) / (x^2 - 5*x + 6) = N₁ / (x - 2) + N₂ / (x - 3)) →
  N₁ * N₂ = -1036 := by
sorry

end partial_fraction_decomposition_l2610_261060


namespace negation_is_true_l2610_261098

theorem negation_is_true : 
  (∀ x : ℝ, x^2 ≥ 1 → (x ≤ -1 ∨ x ≥ 1)) := by sorry

end negation_is_true_l2610_261098


namespace mrs_snyder_pink_cookies_l2610_261088

/-- The total number of cookies Mrs. Snyder made -/
def total_cookies : ℕ := 86

/-- The number of red cookies Mrs. Snyder made -/
def red_cookies : ℕ := 36

/-- The number of pink cookies Mrs. Snyder made -/
def pink_cookies : ℕ := total_cookies - red_cookies

theorem mrs_snyder_pink_cookies : pink_cookies = 50 := by
  sorry

end mrs_snyder_pink_cookies_l2610_261088


namespace b_equals_seven_l2610_261032

-- Define the functions f and F
def f (a : ℝ) (x : ℝ) : ℝ := x - a

def F (x y : ℝ) : ℝ := y^2 + x

-- Define b as F(3, f(4))
def b (a : ℝ) : ℝ := F 3 (f a 4)

-- Theorem to prove
theorem b_equals_seven (a : ℝ) : b a = 7 := by
  sorry

end b_equals_seven_l2610_261032


namespace polynomial_division_quotient_l2610_261006

theorem polynomial_division_quotient : 
  ∀ (x : ℝ), (7 * x^3 + 3 * x^2 - 5 * x - 8) = (x + 2) * (7 * x^2 - 11 * x + 17) + (-42) := by
  sorry

end polynomial_division_quotient_l2610_261006


namespace first_group_size_l2610_261028

/-- The number of days it takes the first group to complete the work -/
def first_group_days : ℕ := 30

/-- The number of men in the second group -/
def second_group_men : ℕ := 25

/-- The number of days it takes the second group to complete the work -/
def second_group_days : ℕ := 24

/-- The number of men in the first group -/
def first_group_men : ℕ := first_group_days * second_group_men * second_group_days / first_group_days

theorem first_group_size :
  first_group_men = 20 :=
by sorry

end first_group_size_l2610_261028


namespace second_largest_part_l2610_261062

theorem second_largest_part (total : ℚ) (a b c d : ℚ) : 
  total = 120 → 
  a + b + c + d = total →
  a / 3 = b / 2 →
  a / 3 = c / 4 →
  a / 3 = d / 5 →
  (max b (min c d)) = 240 / 7 := by
  sorry

end second_largest_part_l2610_261062


namespace thermodynamic_cycle_efficiency_l2610_261025

/-- Represents a thermodynamic cycle with three stages -/
structure ThermodynamicCycle where
  P₀ : ℝ
  ρ₀ : ℝ
  stage1_isochoric : ℝ → ℝ → Prop
  stage2_isobaric : ℝ → ℝ → Prop
  stage3_return : ℝ → ℝ → Prop

/-- Efficiency of a thermodynamic cycle -/
def efficiency (cycle : ThermodynamicCycle) : ℝ := sorry

/-- Maximum possible efficiency for given temperature range -/
def max_efficiency (T_min T_max : ℝ) : ℝ := sorry

/-- Theorem stating the efficiency of the described thermodynamic cycle -/
theorem thermodynamic_cycle_efficiency (cycle : ThermodynamicCycle) 
  (h1 : cycle.stage1_isochoric (3 * cycle.P₀) cycle.P₀)
  (h2 : cycle.stage2_isobaric cycle.ρ₀ (3 * cycle.ρ₀))
  (h3 : cycle.stage3_return 1 1)
  (h4 : ∃ T_min T_max, efficiency cycle = (1 / 8) * max_efficiency T_min T_max) :
  efficiency cycle = 1 / 12 := by
  sorry

end thermodynamic_cycle_efficiency_l2610_261025


namespace distance_by_sea_l2610_261077

/-- The distance traveled by sea is the difference between the total distance and the distance by land -/
theorem distance_by_sea (total_distance land_distance : ℕ) (h1 : total_distance = 601) (h2 : land_distance = 451) :
  total_distance - land_distance = 150 := by
  sorry

end distance_by_sea_l2610_261077


namespace product_of_fractions_l2610_261070

theorem product_of_fractions : 
  (4 / 2) * (8 / 4) * (9 / 3) * (18 / 6) * (16 / 8) * (24 / 12) * (30 / 15) * (36 / 18) = 576 := by
  sorry

end product_of_fractions_l2610_261070


namespace madelines_score_l2610_261042

theorem madelines_score (madeline_mistakes : ℕ) (leo_mistakes : ℕ) (brent_score : ℕ) (brent_mistakes : ℕ) 
  (h1 : madeline_mistakes = 2)
  (h2 : madeline_mistakes * 2 = leo_mistakes)
  (h3 : brent_score = 25)
  (h4 : brent_mistakes = leo_mistakes + 1) :
  30 - madeline_mistakes = 28 := by
  sorry

end madelines_score_l2610_261042


namespace complex_power_to_rectangular_l2610_261065

theorem complex_power_to_rectangular : 
  (3 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^4 = 
    Complex.mk (-40.5) (40.5 * Real.sqrt 3) := by sorry

end complex_power_to_rectangular_l2610_261065


namespace child_sold_seven_apples_l2610_261068

/-- Represents the number of apples sold by a child given the initial conditions and final count -/
def apples_sold (num_children : ℕ) (apples_per_child : ℕ) (eating_children : ℕ) (apples_eaten_each : ℕ) (apples_left : ℕ) : ℕ :=
  num_children * apples_per_child - eating_children * apples_eaten_each - apples_left

/-- Theorem stating that given the conditions in the problem, the child sold 7 apples -/
theorem child_sold_seven_apples :
  apples_sold 5 15 2 4 60 = 7 := by
  sorry

end child_sold_seven_apples_l2610_261068


namespace triangle_abc_properties_l2610_261024

theorem triangle_abc_properties (a b c A B C m : ℝ) :
  0 < A → A ≤ 2 * Real.pi / 3 →
  a > 0 → b > 0 → c > 0 →
  A + B + C = Real.pi →
  a^2 + b^2 - c^2 = Real.sqrt 3 * a * b →
  m = 2 * (Real.cos (A / 2))^2 - Real.sin B - 1 →
  (C = Real.pi / 6 ∧ -1 ≤ m ∧ m < 1/2) :=
by sorry

end triangle_abc_properties_l2610_261024


namespace exists_even_b_for_odd_n_l2610_261082

def operation (p : ℕ × ℕ) : ℕ × ℕ :=
  if p.1 % 2 = 0 then (p.1 / 2, p.2 + p.1 / 2)
  else (p.1 + p.2 / 2, p.2 / 2)

def applyOperationNTimes (p : ℕ × ℕ) (n : ℕ) : ℕ × ℕ :=
  match n with
  | 0 => p
  | m + 1 => operation (applyOperationNTimes p m)

theorem exists_even_b_for_odd_n (n : ℕ) (h_odd : n % 2 = 1) (h_gt_1 : n > 1) :
  ∃ b : ℕ, b % 2 = 0 ∧ b < n ∧ ∃ k : ℕ, applyOperationNTimes (n, b) k = (b, n) := by
  sorry

end exists_even_b_for_odd_n_l2610_261082


namespace sum_of_roots_is_six_l2610_261027

-- Define the quadratic polynomials
def f (a b x : ℝ) : ℝ := x^2 + a*x + b
def g (c d x : ℝ) : ℝ := x^2 + c*x + d

-- State the theorem
theorem sum_of_roots_is_six 
  (a b c d : ℝ) 
  (hf : ∃ r₁ r₂ : ℝ, ∀ x, f a b x = (x - r₁) * (x - r₂))
  (hg : ∃ s₁ s₂ : ℝ, ∀ x, g c d x = (x - s₁) * (x - s₂))
  (h_eq1 : f a b 1 = g c d 2)
  (h_eq2 : g c d 1 = f a b 2) :
  ∃ r₁ r₂ s₁ s₂ : ℝ, r₁ + r₂ + s₁ + s₂ = 6 :=
sorry

end sum_of_roots_is_six_l2610_261027


namespace base_problem_l2610_261087

theorem base_problem (b : ℕ) : (3 * b + 1)^2 = b^3 + 2 * b + 1 → b = 10 := by
  sorry

end base_problem_l2610_261087


namespace pizza_slices_theorem_l2610_261093

/-- Represents the types of pizzas available --/
inductive PizzaType
  | Small
  | Medium
  | Large

/-- Returns the number of slices for a given pizza type --/
def slicesPerPizza (pt : PizzaType) : Nat :=
  match pt with
  | .Small => 6
  | .Medium => 8
  | .Large => 12

/-- Calculates the total number of slices for a given number of pizzas of a specific type --/
def totalSlices (pt : PizzaType) (count : Nat) : Nat :=
  (slicesPerPizza pt) * count

/-- Represents the order of pizzas --/
structure PizzaOrder where
  small : Nat
  medium : Nat
  large : Nat
  total : Nat

theorem pizza_slices_theorem (order : PizzaOrder)
  (h1 : order.small = 4)
  (h2 : order.medium = 5)
  (h3 : order.total = 15)
  (h4 : order.large = order.total - order.small - order.medium) :
  totalSlices .Small order.small +
  totalSlices .Medium order.medium +
  totalSlices .Large order.large = 136 := by
    sorry

#check pizza_slices_theorem

end pizza_slices_theorem_l2610_261093


namespace game_probability_result_l2610_261001

def game_probability (total_rounds : ℕ) 
                     (alex_prob : ℚ) 
                     (mel_chelsea_ratio : ℚ) : ℚ :=
  let chelsea_prob := (1 - alex_prob) / (1 + mel_chelsea_ratio)
  let mel_prob := chelsea_prob * mel_chelsea_ratio
  let specific_sequence_prob := alex_prob^4 * mel_prob^2 * chelsea_prob
  let arrangements := (Nat.factorial total_rounds) / 
                      ((Nat.factorial 4) * (Nat.factorial 2) * (Nat.factorial 1))
  arrangements * specific_sequence_prob

theorem game_probability_result : 
  game_probability 7 (1/2) 2 = 35/288 := by sorry

end game_probability_result_l2610_261001


namespace remainder_equality_l2610_261063

theorem remainder_equality (a b c : ℤ) (hc : c ≠ 0) :
  c ∣ (a - b) → a ≡ b [ZMOD c] :=
by sorry

end remainder_equality_l2610_261063


namespace brick_width_calculation_l2610_261030

/-- Proves that the width of each brick is 11.25 cm given the wall and brick dimensions and the number of bricks needed. -/
theorem brick_width_calculation (brick_length : ℝ) (brick_height : ℝ) (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ) (num_bricks : ℕ) :
  brick_length = 25 →
  brick_height = 6 →
  wall_length = 750 →
  wall_width = 600 →
  wall_height = 22.5 →
  num_bricks = 6000 →
  ∃ (brick_width : ℝ), brick_width = 11.25 ∧ 
    wall_length * wall_width * wall_height = num_bricks * brick_length * brick_width * brick_height :=
by sorry

end brick_width_calculation_l2610_261030


namespace inequality_equivalence_l2610_261084

theorem inequality_equivalence (n : ℕ) (hn : n > 0) :
  (2 * n - 1 : ℝ) * Real.log (1 + Real.log 2023 / Real.log 2) > 
  Real.log 2023 / Real.log 2 * (Real.log 2 + Real.log n) ↔ 
  n ≥ 3 :=
sorry

end inequality_equivalence_l2610_261084


namespace raj_ate_ten_bananas_l2610_261005

/-- The number of bananas Raj ate -/
def bananas_eaten (initial_bananas : ℕ) (remaining_bananas : ℕ) : ℕ :=
  initial_bananas - remaining_bananas - 2 * remaining_bananas

/-- Theorem stating that Raj ate 10 bananas -/
theorem raj_ate_ten_bananas :
  bananas_eaten 310 100 = 10 := by
  sorry

end raj_ate_ten_bananas_l2610_261005


namespace cindy_walking_speed_l2610_261058

/-- Cindy's running speed in miles per hour -/
def running_speed : ℝ := 3

/-- Distance Cindy runs in miles -/
def run_distance : ℝ := 0.5

/-- Distance Cindy walks in miles -/
def walk_distance : ℝ := 0.5

/-- Total time for the journey in minutes -/
def total_time : ℝ := 40

/-- Cindy's walking speed in miles per hour -/
def walking_speed : ℝ := 1

theorem cindy_walking_speed :
  running_speed = 3 ∧
  run_distance = 0.5 ∧
  walk_distance = 0.5 ∧
  total_time = 40 →
  walking_speed = 1 := by sorry

end cindy_walking_speed_l2610_261058


namespace max_condition_implies_a_range_l2610_261056

/-- Given a function f with derivative f'(x) = a(x+1)(x-a), 
    if f has a maximum at x = a, then a is in the open interval (-1, 0) -/
theorem max_condition_implies_a_range (f : ℝ → ℝ) (a : ℝ) 
  (h_deriv : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h_max : IsLocalMax f a) : 
  a ∈ Set.Ioo (-1 : ℝ) 0 := by
sorry

end max_condition_implies_a_range_l2610_261056


namespace range_of_m_l2610_261019

/-- The piecewise function f(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ :=
  if x < m then x else x^2 + 4*x

/-- The property that for all p < m, there exists q ≥ m such that f(p) + f(q) = 0 -/
def property (m : ℝ) : Prop :=
  ∀ p < m, ∃ q ≥ m, f m p + f m q = 0

/-- The theorem stating the range of m -/
theorem range_of_m : ∀ m : ℝ, property m ↔ m ≤ 0 := by sorry

end range_of_m_l2610_261019


namespace one_pepperoni_fell_off_l2610_261045

/-- Represents a pizza with pepperoni slices -/
structure Pizza :=
  (total_pepperoni : ℕ)
  (num_slices : ℕ)
  (pepperoni_on_given_slice : ℕ)

/-- Calculates the number of pepperoni slices that fell off -/
def pepperoni_fell_off (p : Pizza) : ℕ :=
  (p.total_pepperoni / p.num_slices) - p.pepperoni_on_given_slice

/-- Theorem stating that one pepperoni slice fell off -/
theorem one_pepperoni_fell_off (p : Pizza) 
    (h1 : p.total_pepperoni = 40)
    (h2 : p.num_slices = 4)
    (h3 : p.pepperoni_on_given_slice = 9) : 
  pepperoni_fell_off p = 1 := by
  sorry

#eval pepperoni_fell_off { total_pepperoni := 40, num_slices := 4, pepperoni_on_given_slice := 9 }

end one_pepperoni_fell_off_l2610_261045


namespace puppies_adopted_l2610_261091

/-- The cost to get a cat ready for adoption -/
def cat_cost : ℕ := 50

/-- The cost to get an adult dog ready for adoption -/
def adult_dog_cost : ℕ := 100

/-- The cost to get a puppy ready for adoption -/
def puppy_cost : ℕ := 150

/-- The number of cats adopted -/
def cats_adopted : ℕ := 2

/-- The number of adult dogs adopted -/
def adult_dogs_adopted : ℕ := 3

/-- The total cost for all adopted animals -/
def total_cost : ℕ := 700

/-- Theorem stating that the number of puppies adopted is 2 -/
theorem puppies_adopted : 
  ∃ (p : ℕ), p = 2 ∧ 
  cat_cost * cats_adopted + adult_dog_cost * adult_dogs_adopted + puppy_cost * p = total_cost :=
sorry

end puppies_adopted_l2610_261091


namespace cylinder_volume_ratio_l2610_261004

/-- The volume ratio of two right circular cylinders -/
theorem cylinder_volume_ratio :
  let r1 := 4 / Real.pi
  let h1 := 10
  let r2 := 5 / Real.pi
  let h2 := 8
  let v1 := Real.pi * r1^2 * h1
  let v2 := Real.pi * r2^2 * h2
  v1 / v2 = 4 / 5 := by sorry

end cylinder_volume_ratio_l2610_261004


namespace a_salary_is_5250_l2610_261008

/-- Proof that A's salary is $5250 given the conditions of the problem -/
theorem a_salary_is_5250 (a b : ℝ) : 
  a + b = 7000 →                   -- Total salary is $7000
  0.05 * a = 0.15 * b →            -- Savings are equal
  a = 5250 := by
    sorry

end a_salary_is_5250_l2610_261008


namespace square_even_implies_even_l2610_261010

theorem square_even_implies_even (a : ℤ) (h : Even (a^2)) : Even a := by
  sorry

end square_even_implies_even_l2610_261010


namespace unique_solution_l2610_261073

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, x > 0 → f x > 0) ∧ 
  (∀ x y, x > 0 → y > 0 → f (x * f y) = y * f x) ∧
  (Filter.Tendsto f Filter.atTop (nhds 0))

/-- The theorem stating that the function f(x) = 1/x is the unique solution -/
theorem unique_solution (f : ℝ → ℝ) (h : SatisfiesConditions f) : 
  ∀ x, x > 0 → f x = 1 / x := by
  sorry

end unique_solution_l2610_261073


namespace sum_of_evens_between_1_and_31_l2610_261015

def sumOfEvens : ℕ → ℕ
  | 0 => 0
  | n + 1 => if (n + 1) % 2 = 0 ∧ n + 1 < 31 then n + 1 + sumOfEvens n else sumOfEvens n

theorem sum_of_evens_between_1_and_31 : sumOfEvens 30 = 240 := by
  sorry

end sum_of_evens_between_1_and_31_l2610_261015


namespace divisible_by_ten_l2610_261022

theorem divisible_by_ten (n : ℕ) : ∃ k : ℤ, 3^(n+2) - 2^(n+2) + 3^n - 2^n = 10 * k := by
  sorry

end divisible_by_ten_l2610_261022


namespace remainder_sum_l2610_261034

theorem remainder_sum (p q : ℤ) (hp : p % 60 = 47) (hq : q % 45 = 36) : (p + q) % 30 = 23 := by
  sorry

end remainder_sum_l2610_261034


namespace farm_corn_cobs_l2610_261033

/-- The number of corn cobs in a row -/
def cobs_per_row : ℕ := 4

/-- The number of rows in the first field -/
def rows_field1 : ℕ := 13

/-- The number of rows in the second field -/
def rows_field2 : ℕ := 16

/-- The total number of corn cobs grown on the farm -/
def total_cobs : ℕ := rows_field1 * cobs_per_row + rows_field2 * cobs_per_row

theorem farm_corn_cobs : total_cobs = 116 := by
  sorry

end farm_corn_cobs_l2610_261033


namespace matthews_crackers_l2610_261031

theorem matthews_crackers (initial_cakes : ℕ) (num_friends : ℕ) (cakes_eaten_per_person : ℕ)
  (h1 : initial_cakes = 30)
  (h2 : num_friends = 2)
  (h3 : cakes_eaten_per_person = 15)
  : initial_cakes = num_friends * cakes_eaten_per_person :=
by
  sorry

#check matthews_crackers

end matthews_crackers_l2610_261031


namespace peach_difference_l2610_261000

theorem peach_difference (jill_peaches steven_peaches jake_peaches : ℕ) : 
  jill_peaches = 12 →
  steven_peaches = jill_peaches + 15 →
  jake_peaches + 1 = jill_peaches →
  steven_peaches - jake_peaches = 16 :=
by
  sorry

end peach_difference_l2610_261000


namespace wind_on_rainy_day_probability_l2610_261053

/-- Given probabilities in a meteorological context -/
structure WeatherProbabilities where
  rain : ℚ
  wind : ℚ
  both : ℚ

/-- The probability of wind on a rainy day -/
def windOnRainyDay (wp : WeatherProbabilities) : ℚ :=
  wp.both / wp.rain

/-- Theorem stating the probability of wind on a rainy day -/
theorem wind_on_rainy_day_probability (wp : WeatherProbabilities) 
  (h1 : wp.rain = 4/15)
  (h2 : wp.wind = 2/15)
  (h3 : wp.both = 1/10) :
  windOnRainyDay wp = 3/8 := by
  sorry

end wind_on_rainy_day_probability_l2610_261053


namespace maurice_job_search_l2610_261099

/-- The probability of a single application being accepted -/
def p_accept : ℚ := 1 / 5

/-- The probability threshold for stopping -/
def p_threshold : ℚ := 3 / 4

/-- The number of letters Maurice needs to write -/
def num_letters : ℕ := 7

theorem maurice_job_search :
  (1 - (1 - p_accept) ^ num_letters) ≥ p_threshold ∧
  ∀ n : ℕ, n < num_letters → (1 - (1 - p_accept) ^ n) < p_threshold :=
by sorry

end maurice_job_search_l2610_261099


namespace remainder_sum_l2610_261096

theorem remainder_sum (x y : ℤ) (hx : x % 60 = 53) (hy : y % 45 = 17) :
  (x + y) % 15 = 10 := by sorry

end remainder_sum_l2610_261096


namespace certain_number_problem_l2610_261049

theorem certain_number_problem : ∃ x : ℚ, (x + 720) / 125 = 7392 / 462 ∧ x = 1280 := by
  sorry

end certain_number_problem_l2610_261049


namespace lizzy_money_theorem_l2610_261071

def lizzy_money_problem (mother_gift uncle_gift father_gift candy_cost : ℕ) : Prop :=
  let initial_amount := mother_gift + father_gift
  let amount_after_spending := initial_amount - candy_cost
  let final_amount := amount_after_spending + uncle_gift
  final_amount = 140

theorem lizzy_money_theorem :
  lizzy_money_problem 80 70 40 50 := by
  sorry

end lizzy_money_theorem_l2610_261071


namespace greatest_perimeter_of_special_triangle_l2610_261064

theorem greatest_perimeter_of_special_triangle :
  ∀ a b : ℕ,
    a > 0 →
    b > 0 →
    b = 2 * a →
    17 + a > b →
    b + 17 > a →
    a + b > 17 →
    a + b + 17 ≤ 65 :=
by
  sorry

end greatest_perimeter_of_special_triangle_l2610_261064


namespace birds_on_branch_l2610_261013

theorem birds_on_branch (initial_parrots : ℕ) (remaining_parrots : ℕ) (remaining_crows : ℕ) :
  initial_parrots = 7 →
  remaining_parrots = 2 →
  remaining_crows = 1 →
  ∃ (initial_crows : ℕ) (flew_away : ℕ),
    flew_away = initial_parrots - remaining_parrots ∧
    flew_away = initial_crows - remaining_crows ∧
    initial_parrots + initial_crows = 13 :=
by sorry

end birds_on_branch_l2610_261013


namespace power_inequality_l2610_261066

theorem power_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  a^6 + b^6 > a^4*b^2 + a^2*b^4 := by
  sorry

end power_inequality_l2610_261066


namespace quadratic_expression_value_l2610_261039

theorem quadratic_expression_value (x : ℝ) : 
  let a : ℝ := 2010 * x + 2010
  let b : ℝ := 2010 * x + 2011
  let c : ℝ := 2010 * x + 2012
  a^2 + b^2 + c^2 - a*b - b*c - c*a = 3 := by
sorry

end quadratic_expression_value_l2610_261039


namespace angle_c_is_30_degrees_l2610_261090

theorem angle_c_is_30_degrees (A B C : ℝ) : 
  3 * Real.sin A + 4 * Real.cos B = 6 →
  4 * Real.sin B + 3 * Real.cos A = 1 →
  A + B + C = π →
  C = π / 6 := by
  sorry

end angle_c_is_30_degrees_l2610_261090


namespace challenge_probability_challenge_probability_value_l2610_261054

/-- The probability of selecting all letters from "CHALLENGE" when choosing 3 letters from "FARM", 
    4 letters from "BENCHES", and 2 letters from "GLOVE" -/
theorem challenge_probability : ℚ := by
  -- Define the number of letters in each word
  let farm_letters : ℕ := 4
  let benches_letters : ℕ := 7
  let glove_letters : ℕ := 5

  -- Define the number of letters to be selected from each word
  let farm_select : ℕ := 3
  let benches_select : ℕ := 4
  let glove_select : ℕ := 2

  -- Define the number of required letters from each word
  let farm_required : ℕ := 2  -- A and L
  let benches_required : ℕ := 3  -- C, H, and E
  let glove_required : ℕ := 2  -- G and E

  -- Calculate the probability
  sorry

-- The theorem states that the probability is 2/350
theorem challenge_probability_value : challenge_probability = 2 / 350 := by sorry

end challenge_probability_challenge_probability_value_l2610_261054


namespace equation_is_quadratic_l2610_261041

/-- A quadratic equation in terms of x is of the form ax^2 + bx + c = 0, where a ≠ 0 --/
def IsQuadraticEquation (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The function representing the equation 3x^2 + 1 = 0 --/
def f (x : ℝ) : ℝ := 3 * x^2 + 1

theorem equation_is_quadratic : IsQuadraticEquation f := by
  sorry


end equation_is_quadratic_l2610_261041


namespace f1_not_unique_l2610_261035

-- Define the type of our functions
def F := ℝ → ℝ

-- Define the recursive relationship
def recursive_relation (f₁ : F) (n : ℕ) : F :=
  match n with
  | 0 => id
  | 1 => f₁
  | n + 2 => f₁ ∘ (recursive_relation f₁ (n + 1))

-- State the theorem
theorem f1_not_unique :
  ∃ (f₁ g₁ : F),
    f₁ ≠ g₁ ∧
    (∀ (n : ℕ), n ≥ 2 → (recursive_relation f₁ n) = f₁ ∘ (recursive_relation f₁ (n - 1))) ∧
    (∀ (n : ℕ), n ≥ 2 → (recursive_relation g₁ n) = g₁ ∘ (recursive_relation g₁ (n - 1))) ∧
    (recursive_relation f₁ 5) 2 = 33 ∧
    (recursive_relation g₁ 5) 2 = 33 :=
by
  sorry

end f1_not_unique_l2610_261035


namespace malou_average_score_l2610_261095

def malou_quiz_scores : List ℝ := [91, 90, 92]

theorem malou_average_score : 
  (malou_quiz_scores.sum / malou_quiz_scores.length : ℝ) = 91 := by
  sorry

end malou_average_score_l2610_261095


namespace perfume_cost_calculation_l2610_261014

/-- The cost of a bottle of perfume given initial savings, earnings from jobs, and additional amount needed --/
def perfume_cost (christian_initial : ℕ) (sue_initial : ℕ) 
                 (yards_mowed : ℕ) (yard_price : ℕ) 
                 (dogs_walked : ℕ) (dog_price : ℕ) 
                 (additional_needed : ℕ) : ℕ :=
  christian_initial + sue_initial + 
  yards_mowed * yard_price + 
  dogs_walked * dog_price + 
  additional_needed

/-- Theorem stating the cost of the perfume given the problem conditions --/
theorem perfume_cost_calculation : 
  perfume_cost 5 7 4 5 6 2 6 = 50 := by
  sorry

end perfume_cost_calculation_l2610_261014


namespace mikail_birthday_money_l2610_261040

/-- Mikail's age tomorrow -/
def mikail_age : ℕ := 9

/-- Amount of money Mikail receives per year of age -/
def money_per_year : ℕ := 5

/-- Cost of the video game -/
def game_cost : ℕ := 80

/-- Theorem stating Mikail's situation -/
theorem mikail_birthday_money :
  (mikail_age = 3 * 3) ∧
  (mikail_age * money_per_year = 45) ∧
  (mikail_age * money_per_year < game_cost) :=
by sorry

end mikail_birthday_money_l2610_261040


namespace sum_of_largest_and_smallest_prime_factors_of_1560_l2610_261047

theorem sum_of_largest_and_smallest_prime_factors_of_1560 :
  ∃ (smallest largest : ℕ),
    smallest.Prime ∧ largest.Prime ∧
    smallest ∣ 1560 ∧ largest ∣ 1560 ∧
    (∀ p : ℕ, p.Prime → p ∣ 1560 → p ≤ largest) ∧
    (∀ p : ℕ, p.Prime → p ∣ 1560 → p ≥ smallest) ∧
    smallest + largest = 15 :=
by sorry

end sum_of_largest_and_smallest_prime_factors_of_1560_l2610_261047


namespace smallest_distance_between_complex_points_l2610_261092

open Complex

theorem smallest_distance_between_complex_points (z w : ℂ) 
  (hz : Complex.abs (z + 3 + 4*I) = 2)
  (hw : Complex.abs (w - 6 - 10*I) = 4) :
  ∃ (min_dist : ℝ), 
    (∀ (z' w' : ℂ), 
      Complex.abs (z' + 3 + 4*I) = 2 → 
      Complex.abs (w' - 6 - 10*I) = 4 → 
      Complex.abs (z' - w') ≥ min_dist) ∧ 
    min_dist = Real.sqrt 277 - 6 :=
sorry

end smallest_distance_between_complex_points_l2610_261092


namespace pentadecagon_triangles_l2610_261075

/-- The number of vertices in a regular pentadecagon -/
def n : ℕ := 15

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- The number of triangles that can be formed using the vertices of a regular pentadecagon -/
def num_triangles : ℕ := Nat.choose n k

theorem pentadecagon_triangles : num_triangles = 455 := by
  sorry

end pentadecagon_triangles_l2610_261075


namespace overtake_time_problem_l2610_261085

/-- Proves that under given conditions, k started 10 hours after a. -/
theorem overtake_time_problem (speed_a speed_b speed_k : ℝ) 
  (start_delay_b : ℝ) (overtake_time : ℝ) :
  speed_a = 30 →
  speed_b = 40 →
  speed_k = 60 →
  start_delay_b = 5 →
  speed_a * overtake_time = speed_b * (overtake_time - start_delay_b) →
  speed_a * overtake_time = speed_k * (overtake_time - (overtake_time - 10)) →
  overtake_time - (overtake_time - 10) = 10 :=
by sorry

end overtake_time_problem_l2610_261085


namespace shaded_area_calculation_l2610_261055

/-- The area of the shaded region in a configuration where a 5x5 square adjoins a 15x15 square,
    with a line drawn from the top left corner of the larger square to the bottom right corner
    of the smaller square, is 175/8 square inches. -/
theorem shaded_area_calculation : 
  let large_square_side : ℝ := 15
  let small_square_side : ℝ := 5
  let total_width : ℝ := large_square_side + small_square_side
  let triangle_base : ℝ := large_square_side * small_square_side / total_width
  let triangle_area : ℝ := 1/2 * triangle_base * small_square_side
  let small_square_area : ℝ := small_square_side ^ 2
  let shaded_area : ℝ := small_square_area - triangle_area
  shaded_area = 175/8 := by sorry

end shaded_area_calculation_l2610_261055


namespace smallest_sausage_packages_l2610_261072

theorem smallest_sausage_packages (sausage_pack : ℕ) (bun_pack : ℕ) 
  (h1 : sausage_pack = 10) (h2 : bun_pack = 15) :
  ∃ n : ℕ, n > 0 ∧ sausage_pack * n % bun_pack = 0 ∧ 
  ∀ m : ℕ, m > 0 → sausage_pack * m % bun_pack = 0 → n ≤ m :=
by
  sorry

end smallest_sausage_packages_l2610_261072


namespace preimage_of_four_l2610_261021

def f (x : ℝ) : ℝ := x^2

theorem preimage_of_four (x : ℝ) : f x = 4 ↔ x = 2 ∨ x = -2 := by
  sorry

end preimage_of_four_l2610_261021


namespace adjacent_teacher_performances_probability_l2610_261079

-- Define the number of student performances
def num_student_performances : ℕ := 5

-- Define the number of teacher performances
def num_teacher_performances : ℕ := 2

-- Define the total number of performances
def total_performances : ℕ := num_student_performances + num_teacher_performances

-- Define the function to calculate the probability
def probability_adjacent_teacher_performances : ℚ :=
  (num_student_performances + 1 : ℚ) * 2 / ((total_performances : ℚ) * (total_performances - 1 : ℚ) / 2)

-- Theorem statement
theorem adjacent_teacher_performances_probability :
  probability_adjacent_teacher_performances = 2 / 7 := by
  sorry

end adjacent_teacher_performances_probability_l2610_261079


namespace square_perimeter_from_rectangle_division_l2610_261029

/-- Given a square divided into four congruent rectangles, each with its longer side
    parallel to the sides of the square and having a perimeter of 40 inches,
    the perimeter of the square is 64 inches. -/
theorem square_perimeter_from_rectangle_division (s : ℝ) :
  s > 0 →
  (2 * (s + s/4) = 40) →
  (4 * s = 64) :=
by sorry

end square_perimeter_from_rectangle_division_l2610_261029


namespace calculate_death_rate_l2610_261097

/-- Calculates the death rate given birth rate and population growth rate -/
theorem calculate_death_rate (birth_rate : ℝ) (growth_rate : ℝ) : 
  birth_rate = 32 → growth_rate = 0.021 → 
  ∃ (death_rate : ℝ), death_rate = 11 ∧ birth_rate - death_rate = 1000 * growth_rate :=
by
  sorry

#check calculate_death_rate

end calculate_death_rate_l2610_261097


namespace pizza_area_increase_l2610_261020

/-- The radius of the larger pizza in inches -/
def r1 : ℝ := 5

/-- The radius of the smaller pizza in inches -/
def r2 : ℝ := 2

/-- The percentage increase in area from the smaller pizza to the larger pizza -/
def M : ℝ := 525

theorem pizza_area_increase :
  (π * r1^2 - π * r2^2) / (π * r2^2) * 100 = M :=
sorry

end pizza_area_increase_l2610_261020


namespace f_inequality_l2610_261009

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_increasing : ∀ x y, 0 < x ∧ x < y ∧ y < 2 → f x < f y
axiom f_even_shift : ∀ x, f (x + 2) = f (2 - x)

-- State the theorem
theorem f_inequality : f 3.5 < f 1 ∧ f 1 < f 2.5 := by
  sorry

end f_inequality_l2610_261009


namespace product_sixty_sum_diff_equality_l2610_261086

theorem product_sixty_sum_diff_equality (A B C D : ℕ+) :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A * B = 60 →
  C * D = 60 →
  A - B = C + D →
  A = 20 := by
sorry

end product_sixty_sum_diff_equality_l2610_261086


namespace intersection_M_N_l2610_261002

def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}

def N : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (1 - x)}

theorem intersection_M_N : M ∩ N = {x : ℝ | -2 ≤ x ∧ x ≤ 1} := by sorry

end intersection_M_N_l2610_261002


namespace johns_final_push_l2610_261016

/-- John's final push in a speed walking race -/
theorem johns_final_push (john_pace : ℝ) : 
  john_pace * 34 = 3.7 * 34 + (15 + 2) → john_pace = 4.2 := by
  sorry

end johns_final_push_l2610_261016


namespace eight_lines_theorem_l2610_261076

/-- The number of regions created by n lines in a plane, where no two are parallel and no three are concurrent -/
def num_regions (n : ℕ) : ℕ := 1 + n + (n * (n - 1)) / 2

/-- Representation of a set of lines in a plane -/
structure LineSet where
  num_lines : ℕ
  no_parallel : Bool
  no_concurrent : Bool

/-- The given set of lines -/
def given_lines : LineSet :=
  { num_lines := 8
  , no_parallel := true
  , no_concurrent := true }

theorem eight_lines_theorem (lines : LineSet) (h1 : lines.num_lines = 8) 
    (h2 : lines.no_parallel) (h3 : lines.no_concurrent) : 
  num_regions lines.num_lines = 37 := by
  sorry

#eval num_regions 8

end eight_lines_theorem_l2610_261076


namespace stating_correct_equation_representation_l2610_261080

/-- Represents the distribution of people in a campus beautification activity -/
def campus_beautification (initial_weeding : ℕ) (initial_planting : ℕ) (total_support : ℕ) 
  (support_weeding : ℕ) : Prop :=
  let final_weeding := initial_weeding + support_weeding
  let final_planting := initial_planting + (total_support - support_weeding)
  final_weeding = 2 * final_planting

/-- 
Theorem stating that the equation correctly represents the final distribution
of people in the campus beautification activity.
-/
theorem correct_equation_representation 
  (initial_weeding : ℕ) (initial_planting : ℕ) (total_support : ℕ) (support_weeding : ℕ) :
  campus_beautification initial_weeding initial_planting total_support support_weeding →
  initial_weeding + support_weeding = 2 * (initial_planting + (total_support - support_weeding)) :=
by
  sorry

end stating_correct_equation_representation_l2610_261080


namespace sqrt_b_minus_3_domain_l2610_261043

theorem sqrt_b_minus_3_domain : {b : ℝ | ∃ x : ℝ, x^2 = b - 3} = {b : ℝ | b ≥ 3} := by sorry

end sqrt_b_minus_3_domain_l2610_261043


namespace trisha_money_theorem_l2610_261018

/-- The amount of money Trisha spent on meat -/
def meat_cost : ℕ := 17

/-- The amount of money Trisha spent on chicken -/
def chicken_cost : ℕ := 22

/-- The amount of money Trisha spent on veggies -/
def veggies_cost : ℕ := 43

/-- The amount of money Trisha spent on eggs -/
def eggs_cost : ℕ := 5

/-- The amount of money Trisha spent on dog's food -/
def dog_food_cost : ℕ := 45

/-- The amount of money Trisha had left after shopping -/
def money_left : ℕ := 35

/-- The total amount of money Trisha brought at the beginning -/
def total_money : ℕ := meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost + money_left

theorem trisha_money_theorem : total_money = 167 := by
  sorry

end trisha_money_theorem_l2610_261018


namespace math_club_team_selection_l2610_261052

theorem math_club_team_selection (boys girls : ℕ) (h1 : boys = 7) (h2 : girls = 9) :
  (Finset.sum (Finset.range 3) (λ i =>
    Nat.choose boys (i + 2) * Nat.choose girls (4 - i))) = 6846 := by
  sorry

end math_club_team_selection_l2610_261052


namespace square_of_threes_and_four_exist_three_digits_for_infinite_squares_l2610_261026

/-- Represents a number with n threes followed by a four -/
def number_with_threes_and_four (n : ℕ) : ℕ :=
  (3 * (10^n - 1) / 9) * 10 + 4

/-- Represents a number with n+1 ones, followed by n fives, and ending with a six -/
def number_with_ones_fives_and_six (n : ℕ) : ℕ :=
  (10^(2*n + 2) - 1) / 9 * 10^n * 5 + 6

/-- Theorem stating that the square of number_with_threes_and_four is equal to number_with_ones_fives_and_six -/
theorem square_of_threes_and_four (n : ℕ) :
  (number_with_threes_and_four n)^2 = number_with_ones_fives_and_six n := by
  sorry

/-- Corollary stating that there exist three non-zero digits that can be used to form
    an infinite number of decimal representations of squares of different integers -/
theorem exist_three_digits_for_infinite_squares :
  ∃ (d₁ d₂ d₃ : ℕ), d₁ ≠ 0 ∧ d₂ ≠ 0 ∧ d₃ ≠ 0 ∧
    ∀ (n : ℕ), ∃ (m : ℕ), m^2 = number_with_ones_fives_and_six n ∧
    (∀ (k : ℕ), k < n → number_with_ones_fives_and_six k ≠ number_with_ones_fives_and_six n) := by
  sorry

end square_of_threes_and_four_exist_three_digits_for_infinite_squares_l2610_261026


namespace intersection_point_product_l2610_261057

noncomputable section

-- Define the curves in polar coordinates
def C₁ (θ : Real) : Real := 2 * Real.cos θ
def C₂ (θ : Real) : Real := 3 / (Real.cos θ + Real.sin θ)

-- Define the condition for α
def valid_α (α : Real) : Prop := 0 < α ∧ α < Real.pi / 2

-- State the theorem
theorem intersection_point_product (α : Real) 
  (h₁ : valid_α α) 
  (h₂ : C₁ α * C₂ α = 3) : 
  α = Real.pi / 4 := by
  sorry

end

end intersection_point_product_l2610_261057


namespace function_below_x_axis_iff_k_in_range_l2610_261050

/-- The function f(x) parameterized by k -/
def f (k : ℝ) (x : ℝ) : ℝ := (k^2 - k - 2) * x^2 - (k - 2) * x - 1

/-- The theorem stating the equivalence between the function being always below the x-axis and the range of k -/
theorem function_below_x_axis_iff_k_in_range :
  ∀ k : ℝ, (∀ x : ℝ, f k x < 0) ↔ k ∈ Set.Ioo (-2/5 : ℝ) 2 ∪ {2} :=
sorry

end function_below_x_axis_iff_k_in_range_l2610_261050


namespace extended_box_volume_sum_l2610_261036

/-- Represents a rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of the set of points inside or within one unit of a box -/
def volume_extended_box (b : Box) : ℝ := sorry

/-- Checks if two natural numbers are relatively prime -/
def are_relatively_prime (a b : ℕ) : Prop := sorry

theorem extended_box_volume_sum (b : Box) (m n p : ℕ) :
  b.length = 3 ∧ b.width = 4 ∧ b.height = 5 →
  volume_extended_box b = (m : ℝ) + (n : ℝ) * Real.pi / (p : ℝ) →
  m > 0 ∧ n > 0 ∧ p > 0 →
  are_relatively_prime n p →
  m + n + p = 505 := by
  sorry

end extended_box_volume_sum_l2610_261036


namespace book_cost_price_l2610_261089

theorem book_cost_price (selling_price_1 : ℝ) (selling_price_2 : ℝ) : 
  (selling_price_1 = 1.10 * 1800) → 
  (selling_price_2 = 1.15 * 1800) → 
  (selling_price_2 - selling_price_1 = 90) → 
  1800 = 1800 := by
sorry

end book_cost_price_l2610_261089


namespace integer_solutions_inequalities_l2610_261048

theorem integer_solutions_inequalities (x : ℤ) : 
  ((x - 2) / 2 ≤ -x / 2 + 2 ∧ 4 - 7*x < -3) ↔ (x = 2 ∨ x = 3) :=
by sorry

end integer_solutions_inequalities_l2610_261048


namespace exp_ge_e_l2610_261003

theorem exp_ge_e (x : ℝ) (h : x > 0) : Real.exp x ≥ Real.exp 1 := by
  sorry

end exp_ge_e_l2610_261003


namespace power_of_power_l2610_261067

theorem power_of_power (k : ℕ+) : (k^5)^3 = k^15 := by sorry

end power_of_power_l2610_261067


namespace sum_product_inequality_l2610_261017

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + c * a ≤ 0 := by
  sorry

end sum_product_inequality_l2610_261017


namespace candy_bar_profit_l2610_261046

/-- Calculates the profit from selling candy bars -/
def candy_profit (
  num_bars : ℕ
  ) (purchase_price : ℚ)
    (selling_price : ℚ)
    (sales_fee : ℚ) : ℚ :=
  num_bars * selling_price - num_bars * purchase_price - num_bars * sales_fee

/-- Theorem stating the profit from the candy bar sale -/
theorem candy_bar_profit :
  candy_profit 800 (3/4) (2/3) (1/20) = -533/5 := by
  sorry

end candy_bar_profit_l2610_261046


namespace truck_load_problem_l2610_261023

/-- Proves that the number of crates loaded yesterday is 10 --/
theorem truck_load_problem :
  let truck_capacity : ℕ := 13500
  let box_weight : ℕ := 100
  let box_count : ℕ := 100
  let crate_weight : ℕ := 60
  let sack_weight : ℕ := 50
  let sack_count : ℕ := 50
  let bag_weight : ℕ := 40
  let bag_count : ℕ := 10

  let total_box_weight := box_weight * box_count
  let total_sack_weight := sack_weight * sack_count
  let total_bag_weight := bag_weight * bag_count

  let remaining_weight := truck_capacity - (total_box_weight + total_sack_weight + total_bag_weight)

  ∃ crate_count : ℕ, crate_count * crate_weight = remaining_weight ∧ crate_count = 10 :=
by
  sorry

end truck_load_problem_l2610_261023


namespace perimeter_of_C_l2610_261059

-- Define squares A, B, and C
def square_A : Real → Real := λ s ↦ 4 * s
def square_B : Real → Real := λ s ↦ 4 * s
def square_C : Real → Real := λ s ↦ 4 * s

-- Define the conditions
def perimeter_A : Real := 20
def perimeter_B : Real := 36

-- Define the side length of C as the difference between side lengths of A and B
def side_C (a b : Real) : Real := b - a

-- Theorem statement
theorem perimeter_of_C : 
  ∀ (a b : Real),
  square_A a = perimeter_A →
  square_B b = perimeter_B →
  square_C (side_C a b) = 16 := by
  sorry

end perimeter_of_C_l2610_261059


namespace expression_simplification_l2610_261094

theorem expression_simplification (x y : ℝ) (hx : x = 1) (hy : y = 2) :
  (x + 2) * (y - 2) - 2 * (x * y - 2) = 0 := by
  sorry

end expression_simplification_l2610_261094
