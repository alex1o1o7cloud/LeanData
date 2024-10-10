import Mathlib

namespace division_theorem_l2460_246099

theorem division_theorem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 140 → 
  divisor = 15 → 
  remainder = 5 → 
  dividend = divisor * quotient + remainder → 
  quotient = 9 := by
sorry

end division_theorem_l2460_246099


namespace smallest_x_value_l2460_246095

theorem smallest_x_value (x : ℚ) : 
  (6 * (9 * x^2 + 9 * x + 10) = x * (9 * x - 45)) → x ≥ -4/3 :=
by sorry

end smallest_x_value_l2460_246095


namespace total_boys_l2460_246036

theorem total_boys (total_children happy_children sad_children neutral_children : ℕ)
  (girls happy_boys sad_girls neutral_boys : ℕ)
  (h1 : total_children = 60)
  (h2 : happy_children = 30)
  (h3 : sad_children = 10)
  (h4 : neutral_children = 20)
  (h5 : girls = 41)
  (h6 : happy_boys = 6)
  (h7 : sad_girls = 4)
  (h8 : neutral_boys = 7)
  (h9 : total_children = happy_children + sad_children + neutral_children)
  (h10 : total_children = girls + (happy_boys + (sad_children - sad_girls) + neutral_boys)) :
  happy_boys + (sad_children - sad_girls) + neutral_boys = 19 := by
  sorry

#check total_boys

end total_boys_l2460_246036


namespace black_cars_count_l2460_246024

theorem black_cars_count (total : ℕ) (blue_fraction red_fraction green_fraction : ℚ) :
  total = 1824 →
  blue_fraction = 2 / 5 →
  red_fraction = 1 / 3 →
  green_fraction = 1 / 8 →
  ∃ (blue red green black : ℕ),
    blue + red + green + black = total ∧
    blue = ⌊blue_fraction * total⌋ ∧
    red = red_fraction * total ∧
    green = green_fraction * total ∧
    black = 259 :=
by sorry

end black_cars_count_l2460_246024


namespace existence_of_special_integers_l2460_246025

theorem existence_of_special_integers : ∃ (a b : ℕ+), 
  (¬ (7 ∣ (a.val * b.val * (a.val + b.val)))) ∧ 
  (7 ∣ ((a.val + b.val)^7 - a.val^7 - b.val^7)) ∧
  (a.val = 18 ∧ b.val = 1) := by
sorry

end existence_of_special_integers_l2460_246025


namespace integer_representation_l2460_246054

theorem integer_representation (k : ℤ) (h : -1985 ≤ k ∧ k ≤ 1985) :
  ∃ (a : Fin 8 → ℤ), (∀ i, a i ∈ ({-1, 0, 1} : Set ℤ)) ∧
    k = (a 0) * 1 + (a 1) * 3 + (a 2) * 9 + (a 3) * 27 +
        (a 4) * 81 + (a 5) * 243 + (a 6) * 729 + (a 7) * 2187 :=
by sorry

end integer_representation_l2460_246054


namespace sum_of_six_odd_squares_not_1986_l2460_246091

theorem sum_of_six_odd_squares_not_1986 : ¬ ∃ (a b c d e f : ℤ), 
  (Odd a ∧ Odd b ∧ Odd c ∧ Odd d ∧ Odd e ∧ Odd f) ∧
  (a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 1986) := by
  sorry

end sum_of_six_odd_squares_not_1986_l2460_246091


namespace group_age_problem_l2460_246082

theorem group_age_problem (n : ℕ) (h1 : n > 0) : 
  (15 * n + 35) / (n + 1) = 17 → n = 9 := by sorry

end group_age_problem_l2460_246082


namespace fraction_reducibility_l2460_246050

theorem fraction_reducibility (l : ℤ) :
  ∃ (d : ℤ), d > 1 ∧ d ∣ (5 * l + 6) ∧ d ∣ (8 * l + 7) ↔ ∃ (k : ℤ), l = 13 * k + 4 :=
sorry

end fraction_reducibility_l2460_246050


namespace simplify_expression_l2460_246027

theorem simplify_expression (x : ℝ) (h : x^2 ≥ 49) :
  (7 - Real.sqrt (x^2 - 49))^2 = x^2 - 14 * Real.sqrt (x^2 - 49) := by
  sorry

end simplify_expression_l2460_246027


namespace projection_parallel_condition_l2460_246008

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a 3D line
  -- (simplified for this example)

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane
  -- (simplified for this example)

/-- Projection of a line onto a plane -/
def project (l : Line3D) (p : Plane3D) : Line3D :=
  sorry -- Definition of projection

/-- Parallel lines -/
def parallel (l1 l2 : Line3D) : Prop :=
  sorry -- Definition of parallel lines

theorem projection_parallel_condition 
  (a b m n : Line3D) (α : Plane3D) 
  (h1 : a ≠ b)
  (h2 : m = project a α)
  (h3 : n = project b α)
  (h4 : m ≠ n) :
  (∀ (a b : Line3D), parallel a b → parallel (project a α) (project b α)) ∧
  (∃ (a b : Line3D), parallel (project a α) (project b α) ∧ ¬parallel a b) :=
sorry

end projection_parallel_condition_l2460_246008


namespace no_nontrivial_solution_x2_plus_y2_eq_3z2_l2460_246015

theorem no_nontrivial_solution_x2_plus_y2_eq_3z2 :
  ∀ (x y z : ℤ), x^2 + y^2 = 3 * z^2 → x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end no_nontrivial_solution_x2_plus_y2_eq_3z2_l2460_246015


namespace solutions_to_equation_all_solutions_l2460_246058

def solutions : Set ℂ := {1 + Complex.I * (2 : ℂ)^(1/3 : ℂ), 
                          1 - Complex.I * (2 : ℂ)^(1/3 : ℂ), 
                          -1 + Complex.I * (2 : ℂ)^(1/3 : ℂ), 
                          -1 - Complex.I * (2 : ℂ)^(1/3 : ℂ), 
                          Complex.I * (2 : ℂ)^(1/3 : ℂ), 
                          -Complex.I * (2 : ℂ)^(1/3 : ℂ)}

theorem solutions_to_equation : ∀ z ∈ solutions, z^6 = -8 :=
by sorry

theorem all_solutions : ∀ z : ℂ, z^6 = -8 → z ∈ solutions :=
by sorry

end solutions_to_equation_all_solutions_l2460_246058


namespace difference_value_l2460_246005

theorem difference_value (n : ℚ) : n = 45 → (n / 3 - 5 : ℚ) = 10 := by
  sorry

end difference_value_l2460_246005


namespace fraction_equivalence_l2460_246014

theorem fraction_equivalence (x : ℝ) (h : x ≠ 5) :
  ¬(∀ x : ℝ, x ≠ 5 → (x + 3) / (x - 5) = 3 / (-5)) :=
by sorry

end fraction_equivalence_l2460_246014


namespace max_kitchen_towel_sets_is_13_l2460_246055

-- Define the given parameters
def budget : ℚ := 600
def guest_bathroom_sets : ℕ := 2
def master_bathroom_sets : ℕ := 4
def hand_towel_sets : ℕ := 3
def guest_bathroom_price : ℚ := 40
def master_bathroom_price : ℚ := 50
def hand_towel_price : ℚ := 30
def kitchen_towel_price : ℚ := 20
def guest_bathroom_discount : ℚ := 0.15
def master_bathroom_discount : ℚ := 0.20
def hand_towel_discount : ℚ := 0.15
def kitchen_towel_discount : ℚ := 0.10
def sales_tax : ℚ := 0.08

-- Define the function to calculate the maximum number of kitchen towel sets
def max_kitchen_towel_sets : ℕ :=
  let guest_bathroom_cost := guest_bathroom_sets * guest_bathroom_price * (1 - guest_bathroom_discount)
  let master_bathroom_cost := master_bathroom_sets * master_bathroom_price * (1 - master_bathroom_discount)
  let hand_towel_cost := hand_towel_sets * hand_towel_price * (1 - hand_towel_discount)
  let total_cost_before_tax := guest_bathroom_cost + master_bathroom_cost + hand_towel_cost
  let total_cost_after_tax := total_cost_before_tax * (1 + sales_tax)
  let remaining_budget := budget - total_cost_after_tax
  let kitchen_towel_cost_after_tax := kitchen_towel_price * (1 - kitchen_towel_discount) * (1 + sales_tax)
  (remaining_budget / kitchen_towel_cost_after_tax).floor.toNat

-- Theorem statement
theorem max_kitchen_towel_sets_is_13 : max_kitchen_towel_sets = 13 := by
  sorry


end max_kitchen_towel_sets_is_13_l2460_246055


namespace lcm_of_8_12_15_l2460_246038

theorem lcm_of_8_12_15 : Nat.lcm (Nat.lcm 8 12) 15 = 120 := by sorry

end lcm_of_8_12_15_l2460_246038


namespace complex_power_four_l2460_246071

theorem complex_power_four : 
  (3 * (Complex.cos (π / 6) + Complex.I * Complex.sin (π / 6)))^4 = 
  Complex.mk (-40.5) (40.5 * Real.sqrt 3) := by
  sorry

end complex_power_four_l2460_246071


namespace battleship_theorem_l2460_246044

/-- Represents a rectangular grid -/
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a ship on the grid -/
structure Ship :=
  (length : ℕ)
  (width : ℕ)

/-- Represents a set of connected cells -/
structure ConnectedCells :=
  (num_cells : ℕ)

/-- The minimum number of shots needed to guarantee hitting a ship on a grid -/
def min_shots_to_hit_ship (g : Grid) (s : Ship) : ℕ := sorry

/-- The minimum number of shots needed to guarantee hitting connected cells on a grid -/
def min_shots_to_hit_connected_cells (g : Grid) (c : ConnectedCells) : ℕ := sorry

/-- The main theorem for the Battleship problem -/
theorem battleship_theorem (g : Grid) (s : Ship) (c : ConnectedCells) :
  g.rows = 7 ∧ g.cols = 7 ∧ 
  ((s.length = 1 ∧ s.width = 4) ∨ (s.length = 4 ∧ s.width = 1)) ∧
  c.num_cells = 4 →
  (min_shots_to_hit_ship g s = 12) ∧
  (min_shots_to_hit_connected_cells g c = 20) := by sorry

end battleship_theorem_l2460_246044


namespace age_ratio_problem_l2460_246011

theorem age_ratio_problem (p q : ℕ) 
  (h1 : p - 12 = (q - 12) / 2)
  (h2 : p + q = 42) :
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ a * q = b * p ∧ a = 3 ∧ b = 4 :=
sorry

end age_ratio_problem_l2460_246011


namespace edith_books_count_l2460_246051

theorem edith_books_count : ∀ (novels : ℕ) (writing_books : ℕ),
  novels = 80 →
  writing_books = 2 * novels →
  novels + writing_books = 240 :=
by
  sorry

end edith_books_count_l2460_246051


namespace karina_birth_year_l2460_246087

def current_year : ℕ := 2022
def brother_birth_year : ℕ := 1990

theorem karina_birth_year (karina_age brother_age : ℕ) 
  (h1 : karina_age = 2 * brother_age)
  (h2 : brother_age = current_year - brother_birth_year) :
  current_year - karina_age = 1958 := by
sorry

end karina_birth_year_l2460_246087


namespace matthew_cakes_l2460_246083

theorem matthew_cakes (initial_crackers : ℕ) (friends : ℕ) (crackers_eaten : ℕ) :
  initial_crackers = 22 →
  friends = 11 →
  crackers_eaten = 2 →
  ∃ (initial_cakes : ℕ),
    initial_cakes = 22 ∧
    initial_crackers / friends = crackers_eaten ∧
    initial_cakes / friends = crackers_eaten :=
by sorry

end matthew_cakes_l2460_246083


namespace total_flooring_cost_l2460_246041

/-- Represents the dimensions and costs associated with a room's flooring replacement. -/
structure Room where
  length : ℝ
  width : ℝ
  removal_cost : ℝ
  new_flooring_cost_per_sqft : ℝ

/-- Calculates the total cost of replacing flooring in a room. -/
def room_cost (r : Room) : ℝ :=
  r.removal_cost + r.length * r.width * r.new_flooring_cost_per_sqft

/-- Theorem stating that the total cost of replacing flooring in all rooms is $264. -/
theorem total_flooring_cost (living_room bedroom kitchen : Room)
    (h1 : living_room = { length := 8, width := 7, removal_cost := 50, new_flooring_cost_per_sqft := 1.25 })
    (h2 : bedroom = { length := 6, width := 6, removal_cost := 35, new_flooring_cost_per_sqft := 1.50 })
    (h3 : kitchen = { length := 5, width := 4, removal_cost := 20, new_flooring_cost_per_sqft := 1.75 }) :
    room_cost living_room + room_cost bedroom + room_cost kitchen = 264 := by
  sorry

end total_flooring_cost_l2460_246041


namespace stream_speed_l2460_246019

theorem stream_speed (upstream_distance : ℝ) (downstream_distance : ℝ) (time : ℝ) 
  (h1 : upstream_distance = 16)
  (h2 : downstream_distance = 24)
  (h3 : time = 4)
  (h4 : upstream_distance / time + downstream_distance / time = 10) :
  let stream_speed := (downstream_distance - upstream_distance) / (2 * time)
  stream_speed = 1 := by
  sorry

end stream_speed_l2460_246019


namespace interview_problem_l2460_246018

/-- The number of people to be hired -/
def people_hired : ℕ := 3

/-- The probability of two specific individuals being hired together -/
def prob_two_hired : ℚ := 1 / 70

/-- The total number of people interviewed -/
def total_interviewed : ℕ := 21

theorem interview_problem :
  (people_hired = 3) →
  (prob_two_hired = 1 / 70) →
  (total_interviewed = 21) := by
  sorry

end interview_problem_l2460_246018


namespace prime_residue_theorem_l2460_246021

/-- Definition of suitable triple -/
def suitable (p : ℕ) (a b c : ℕ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  a % p ≠ b % p ∧ b % p ≠ c % p ∧ a % p ≠ c % p

/-- Definition of f_k function -/
def f_k (p k a b c : ℕ) : ℤ :=
  a * (b - c)^(p - k) + b * (c - a)^(p - k) + c * (a - b)^(p - k)

theorem prime_residue_theorem (p : ℕ) (hp : p.Prime) (hp11 : p ≥ 11) :
  (∃ a b c : ℕ, suitable p a b c ∧ (p : ℤ) ∣ f_k p 2 a b c) ∧
  (∀ a b c : ℕ, suitable p a b c → (p : ℤ) ∣ f_k p 2 a b c →
    (∃ k : ℕ, k ≥ 3 ∧ ¬((p : ℤ) ∣ f_k p k a b c))) ∧
  (∀ a b c : ℕ, suitable p a b c → (p : ℤ) ∣ f_k p 2 a b c →
    (∀ k : ℕ, k ≥ 3 → k < 4 → (p : ℤ) ∣ f_k p k a b c)) :=
sorry

end prime_residue_theorem_l2460_246021


namespace max_min_m_l2460_246007

theorem max_min_m (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0)
  (h4 : 3*a + 2*b + c = 5) (h5 : 2*a + b - 3*c = 1) :
  let m := 3*a + b - 7*c
  ∃ (m_max m_min : ℝ), 
    (∀ x, x = m → x ≤ m_max) ∧ 
    (∀ x, x = m → x ≥ m_min) ∧ 
    m_max = -1/11 ∧ 
    m_min = -5/7 :=
sorry

end max_min_m_l2460_246007


namespace chris_savings_l2460_246056

theorem chris_savings (x : ℝ) 
  (grandmother : ℝ) (aunt_uncle : ℝ) (parents : ℝ) (total : ℝ)
  (h1 : grandmother = 25)
  (h2 : aunt_uncle = 20)
  (h3 : parents = 75)
  (h4 : total = 279)
  (h5 : x + grandmother + aunt_uncle + parents = total) :
  x = 159 := by
sorry

end chris_savings_l2460_246056


namespace sara_onions_l2460_246066

theorem sara_onions (sally_onions fred_onions total_onions : ℕ) 
  (h1 : sally_onions = 5)
  (h2 : fred_onions = 9)
  (h3 : total_onions = 18)
  : total_onions - (sally_onions + fred_onions) = 4 := by
  sorry

end sara_onions_l2460_246066


namespace smallest_prime_divisor_of_sum_l2460_246074

theorem smallest_prime_divisor_of_sum (p : ℕ → ℕ → Prop) :
  (∀ x y, p x y ↔ x ∣ y) →
  p 2 (3^19 + 11^13) ∧ 
  (∀ q, q < 2 → q.Prime → ¬p q (3^19 + 11^13)) :=
by sorry

end smallest_prime_divisor_of_sum_l2460_246074


namespace cherry_pies_count_l2460_246059

theorem cherry_pies_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) 
  (h_total : total_pies = 36)
  (h_ratio : apple_ratio = 2 ∧ blueberry_ratio = 5 ∧ cherry_ratio = 4) :
  (cherry_ratio * total_pies) / (apple_ratio + blueberry_ratio + cherry_ratio) = 13 := by
  sorry

end cherry_pies_count_l2460_246059


namespace oak_trees_after_planting_l2460_246047

/-- The number of oak trees in the park after planting -/
def total_oak_trees (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

theorem oak_trees_after_planting :
  total_oak_trees 5 4 = 9 := by
  sorry

end oak_trees_after_planting_l2460_246047


namespace sandy_molly_age_ratio_l2460_246062

/-- Proves that the ratio of Sandy's current age to Molly's current age is 4:3 -/
theorem sandy_molly_age_ratio :
  let sandy_future_age : ℕ := 34
  let years_to_future : ℕ := 6
  let molly_current_age : ℕ := 21
  let sandy_current_age : ℕ := sandy_future_age - years_to_future
  (sandy_current_age : ℚ) / (molly_current_age : ℚ) = 4 / 3 := by
  sorry

#check sandy_molly_age_ratio

end sandy_molly_age_ratio_l2460_246062


namespace square_area_increase_l2460_246029

theorem square_area_increase (x y : ℝ) : 
  (∀ s : ℝ, s = 3 → (s + x)^2 - s^2 = y) → 
  y = x^2 + 6*x := by sorry

end square_area_increase_l2460_246029


namespace line_passes_through_fixed_point_l2460_246084

/-- A line in the form (m-2)x-y+3m+2=0 passes through the point (-3, 8) for all real m -/
theorem line_passes_through_fixed_point :
  ∀ (m : ℝ), (m - 2) * (-3) - 8 + 3 * m + 2 = 0 := by
  sorry

end line_passes_through_fixed_point_l2460_246084


namespace no_x_squared_term_l2460_246002

theorem no_x_squared_term (p : ℚ) : 
  (∀ x, (x^2 + p*x) * (x^2 - 3*x + 1) = x^4 + (p-3)*x^3 + 0*x^2 + p*x) → p = 1/3 := by
sorry

end no_x_squared_term_l2460_246002


namespace mother_carrots_count_l2460_246039

/-- The number of carrots Vanessa picked -/
def vanessa_carrots : ℕ := 17

/-- The number of good carrots -/
def good_carrots : ℕ := 24

/-- The number of bad carrots -/
def bad_carrots : ℕ := 7

/-- The number of carrots Vanessa's mother picked -/
def mother_carrots : ℕ := (good_carrots + bad_carrots) - vanessa_carrots

theorem mother_carrots_count : mother_carrots = 14 := by
  sorry

end mother_carrots_count_l2460_246039


namespace smallest_two_digit_prime_with_reversed_composite_div_five_l2460_246003

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(m ∣ n)

def reverse_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let ones := n % 10
  ones * 10 + tens

def is_composite (n : ℕ) : Prop := ∃ m : ℕ, m > 1 ∧ m < n ∧ m ∣ n

theorem smallest_two_digit_prime_with_reversed_composite_div_five :
  ∃ (n : ℕ),
    n ≥ 20 ∧ n < 30 ∧
    is_prime n ∧
    is_composite (reverse_digits n) ∧
    (reverse_digits n) % 5 = 0 ∧
    ∀ (m : ℕ), m ≥ 20 ∧ m < n →
      ¬(is_prime m ∧ is_composite (reverse_digits m) ∧ (reverse_digits m) % 5 = 0) :=
  by sorry

end smallest_two_digit_prime_with_reversed_composite_div_five_l2460_246003


namespace chris_age_l2460_246026

theorem chris_age (a b c : ℕ) : 
  (a + b + c) / 3 = 12 →
  b - 5 = 2 * (c + 2) →
  b + 3 = a + 3 →
  c = 4 :=
by sorry

end chris_age_l2460_246026


namespace product_sum_7293_l2460_246070

theorem product_sum_7293 : ∃ (a b : ℕ), 
  10 ≤ a ∧ a ≤ 99 ∧ 
  10 ≤ b ∧ b ≤ 99 ∧ 
  a * b = 7293 ∧ 
  a + b = 114 := by
sorry

end product_sum_7293_l2460_246070


namespace min_dot_product_op_ab_l2460_246068

open Real

/-- The minimum dot product of OP and AB -/
theorem min_dot_product_op_ab :
  ∀ x : ℝ,
  let A : ℝ × ℝ := (1, 0)
  let B : ℝ × ℝ := (0, 1)
  let P : ℝ × ℝ := (x, exp x)
  let OP : ℝ × ℝ := P
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  (OP.1 * AB.1 + OP.2 * AB.2) ≥ 1 :=
by sorry

#check min_dot_product_op_ab

end min_dot_product_op_ab_l2460_246068


namespace earnings_calculation_l2460_246042

/-- Calculates the discounted price for a given quantity and unit price with a discount rate and minimum quantity for discount --/
def discountedPrice (quantity : ℕ) (unitPrice : ℚ) (discountRate : ℚ) (minQuantity : ℕ) : ℚ :=
  if quantity ≥ minQuantity then
    (1 - discountRate) * (quantity : ℚ) * unitPrice
  else
    (quantity : ℚ) * unitPrice

/-- Calculates the total earnings after all discounts --/
def totalEarnings (smallQuantity mediumQuantity largeQuantity extraLargeQuantity : ℕ) : ℚ :=
  let smallPrice := discountedPrice smallQuantity (30 : ℚ) (1/10 : ℚ) 4
  let mediumPrice := discountedPrice mediumQuantity (45 : ℚ) (3/20 : ℚ) 3
  let largePrice := discountedPrice largeQuantity (60 : ℚ) (1/20 : ℚ) 6
  let extraLargePrice := discountedPrice extraLargeQuantity (85 : ℚ) (2/25 : ℚ) 2
  let subtotal := smallPrice + mediumPrice + largePrice + extraLargePrice
  if smallQuantity + mediumQuantity ≥ 10 then
    (97/100 : ℚ) * subtotal
  else
    subtotal

theorem earnings_calculation (smallQuantity mediumQuantity largeQuantity extraLargeQuantity : ℕ) :
  smallQuantity = 8 ∧ mediumQuantity = 11 ∧ largeQuantity = 4 ∧ extraLargeQuantity = 3 →
  totalEarnings smallQuantity mediumQuantity largeQuantity extraLargeQuantity = (1078.01 : ℚ) := by
  sorry

end earnings_calculation_l2460_246042


namespace parabola_and_line_properties_l2460_246053

-- Define the parabola
structure Parabola where
  vertex : ℝ × ℝ
  focus : ℝ × ℝ
  a : ℝ

-- Define a point on the parabola
def PointOnParabola (p : Parabola) (x y : ℝ) : Prop :=
  y = p.a * x^2

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  slope : ℝ
  y_intercept : ℝ

-- Main theorem
theorem parabola_and_line_properties
  (p : Parabola)
  (A : ℝ × ℝ)
  (c : Circle)
  (m : Line) :
  p.vertex = (0, 0) →
  p.focus.1 = 0 →
  p.focus.2 > 0 →
  PointOnParabola p A.1 A.2 →
  c.center = A →
  c.radius = 2 →
  c.center.2 - c.radius = p.focus.2 →
  m.y_intercept = 6 →
  ∃ (P Q : ℝ × ℝ),
    PointOnParabola p P.1 P.2 ∧
    PointOnParabola p Q.1 Q.2 ∧
    P.2 = m.slope * P.1 + m.y_intercept ∧
    Q.2 = m.slope * Q.1 + m.y_intercept →
  (∀ (x y : ℝ), y = p.a * x^2 ↔ y = (1/4) * x^2) ∧
  (m.slope = 1/2 ∨ m.slope = -1/2) :=
sorry

end parabola_and_line_properties_l2460_246053


namespace average_and_subtraction_l2460_246092

theorem average_and_subtraction (y : ℝ) : 
  (15 + 25 + y) / 3 = 22 → y - 7 = 19 := by
  sorry

end average_and_subtraction_l2460_246092


namespace blue_ball_count_l2460_246001

/-- The number of balls of each color in a box --/
structure BallCounts where
  blue : ℕ
  red : ℕ
  green : ℕ
  yellow : ℕ

/-- The conditions of the ball counting problem --/
def ballProblem (counts : BallCounts) : Prop :=
  counts.red = 4 ∧
  counts.green = 3 * counts.blue ∧
  counts.yellow = 2 * counts.red ∧
  counts.blue + counts.red + counts.green + counts.yellow = 36

theorem blue_ball_count :
  ∃ (counts : BallCounts), ballProblem counts ∧ counts.blue = 6 := by
  sorry

end blue_ball_count_l2460_246001


namespace volume_maximized_at_one_meter_l2460_246064

/-- Represents the dimensions of a rectangular box --/
structure BoxDimensions where
  x : Real  -- Length of the shorter side of the base
  h : Real  -- Height of the box

/-- Calculates the volume of the box given its dimensions --/
def boxVolume (d : BoxDimensions) : Real :=
  2 * d.x^2 * d.h

/-- Calculates the total wire length used for the box frame --/
def wireLengthUsed (d : BoxDimensions) : Real :=
  12 * d.x + 4 * d.h

/-- Theorem stating that the volume is maximized when the shorter side is 1m --/
theorem volume_maximized_at_one_meter :
  ∃ (d : BoxDimensions),
    wireLengthUsed d = 18 ∧
    (∀ (d' : BoxDimensions), wireLengthUsed d' = 18 → boxVolume d' ≤ boxVolume d) ∧
    d.x = 1 :=
  sorry

end volume_maximized_at_one_meter_l2460_246064


namespace initial_knives_count_l2460_246088

/-- Represents the initial number of knives --/
def initial_knives : ℕ := 24

/-- Represents the initial number of teaspoons --/
def initial_teaspoons : ℕ := 2 * initial_knives

/-- Represents the additional knives --/
def additional_knives : ℕ := initial_knives / 3

/-- Represents the additional teaspoons --/
def additional_teaspoons : ℕ := (2 * initial_teaspoons) / 3

/-- The total number of cutlery pieces after additions --/
def total_cutlery : ℕ := 112

theorem initial_knives_count : 
  initial_knives + initial_teaspoons + additional_knives + additional_teaspoons = total_cutlery :=
by sorry

end initial_knives_count_l2460_246088


namespace blue_face_prob_half_l2460_246080

/-- A rectangular prism with colored faces -/
structure ColoredPrism where
  green_faces : ℕ
  yellow_faces : ℕ
  blue_faces : ℕ

/-- The probability of rolling a blue face on a colored prism -/
def blue_face_probability (prism : ColoredPrism) : ℚ :=
  prism.blue_faces / (prism.green_faces + prism.yellow_faces + prism.blue_faces)

/-- Theorem: The probability of rolling a blue face on the given prism is 1/2 -/
theorem blue_face_prob_half :
  let prism : ColoredPrism := ⟨4, 2, 6⟩
  blue_face_probability prism = 1/2 := by
  sorry

end blue_face_prob_half_l2460_246080


namespace urn_problem_l2460_246081

theorem urn_problem (N : ℝ) : 
  (5 / 10 * 20 / (20 + N) + 5 / 10 * N / (20 + N) = 0.6) → N = 20 := by
  sorry

end urn_problem_l2460_246081


namespace total_players_count_l2460_246049

theorem total_players_count (kabadi : ℕ) (kho_kho_only : ℕ) (both : ℕ) : 
  kabadi = 10 → kho_kho_only = 30 → both = 5 → 
  kabadi + kho_kho_only - both = 35 := by
  sorry

end total_players_count_l2460_246049


namespace parabola_directrix_l2460_246073

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := y = 8 * x^2

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop := y = -2

/-- Theorem stating that the given directrix equation is correct for the given parabola -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), parabola_equation x y → (∃ (d : ℝ), directrix_equation d ∧ 
    -- Additional conditions to relate the parabola and directrix
    (x^2 + (y - 2)^2 = (y + 2)^2)) := by
  sorry

end parabola_directrix_l2460_246073


namespace quadratic_radical_equivalence_l2460_246035

theorem quadratic_radical_equivalence (x : ℝ) :
  (∃ (y : ℝ), y > 0 ∧ y * y = x - 1 ∧ (∀ (z : ℝ), z > 0 → z * z = 8 → ∃ (k : ℚ), y = k * z)) →
  x = 9 := by
  sorry

end quadratic_radical_equivalence_l2460_246035


namespace smallest_coin_arrangement_l2460_246085

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The number of proper factors of a positive integer -/
def num_proper_factors (n : ℕ+) : ℕ := sorry

/-- Theorem stating that 180 is the smallest positive integer satisfying the given conditions -/
theorem smallest_coin_arrangement :
  ∀ n : ℕ+, (num_factors n = 9 ∧ num_proper_factors n = 7) → n ≥ 180 :=
sorry

end smallest_coin_arrangement_l2460_246085


namespace complement_intersection_theorem_l2460_246045

-- Define the universe set U
def U : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 10}

-- Define subset A
def A : Set ℝ := {x : ℝ | 2 < x ∧ x ≤ 4}

-- Define subset B
def B : Set ℝ := {x : ℝ | 3 < x ∧ x ≤ 5}

-- State the theorem
theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {x : ℝ | (0 ≤ x ∧ x ≤ 2) ∨ (5 < x ∧ x < 10)} := by
  sorry

end complement_intersection_theorem_l2460_246045


namespace expression_equality_l2460_246000

theorem expression_equality (y a : ℝ) (h1 : y > 0) 
  (h2 : (a * y) / 20 + (3 * y) / 10 = 0.5 * y) : a = 4 := by
  sorry

end expression_equality_l2460_246000


namespace probability_three_white_two_black_l2460_246040

/-- The number of white balls in the box -/
def white_balls : ℕ := 8

/-- The number of black balls in the box -/
def black_balls : ℕ := 9

/-- The total number of balls drawn -/
def drawn_balls : ℕ := 5

/-- The number of white balls drawn -/
def white_drawn : ℕ := 3

/-- The number of black balls drawn -/
def black_drawn : ℕ := 2

/-- The probability of drawing 3 white balls and 2 black balls -/
theorem probability_three_white_two_black :
  (Nat.choose white_balls white_drawn * Nat.choose black_balls black_drawn : ℚ) /
  (Nat.choose (white_balls + black_balls) drawn_balls : ℚ) = 672 / 2063 :=
sorry

end probability_three_white_two_black_l2460_246040


namespace fraction_inequality_l2460_246097

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : c < d) (h4 : d < 0) : 
  a / d < b / c := by
  sorry

end fraction_inequality_l2460_246097


namespace max_area_rectangular_prism_volume_l2460_246060

/-- The volume of a rectangular prism with maximum base area -/
theorem max_area_rectangular_prism_volume
  (base_perimeter : ℝ)
  (height : ℝ)
  (h_base_perimeter : base_perimeter = 32)
  (h_height : height = 9)
  (h_max_area : ∀ (l w : ℝ), l + w = base_perimeter / 2 → l * w ≤ (base_perimeter / 4) ^ 2) :
  (base_perimeter / 4) ^ 2 * height = 576 :=
sorry

end max_area_rectangular_prism_volume_l2460_246060


namespace tangent_perpendicular_l2460_246028

-- Define the curve f(x) = x²
def f (x : ℝ) : ℝ := x^2

-- Define the line perpendicular to the tangent
def perp_line (x y : ℝ) : Prop := x + 4*y - 8 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 4*x - y - 4 = 0

-- Theorem statement
theorem tangent_perpendicular :
  ∃ (x₀ y₀ : ℝ),
    -- (x₀, y₀) is on the curve
    f x₀ = y₀ ∧
    -- The tangent at (x₀, y₀) is perpendicular to the given line
    (∀ (x y : ℝ), perp_line x y → (y - y₀) = -(1/4) * (x - x₀)) ∧
    -- The tangent line equation
    tangent_line x₀ y₀ :=
sorry

end tangent_perpendicular_l2460_246028


namespace combination_20_choose_6_l2460_246016

theorem combination_20_choose_6 : Nat.choose 20 6 = 19380 := by
  sorry

end combination_20_choose_6_l2460_246016


namespace cos_225_degrees_l2460_246075

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_degrees_l2460_246075


namespace gold_bars_problem_l2460_246076

theorem gold_bars_problem (initial : ℕ) : 
  (initial : ℚ) * (1 - 0.1) * 0.5 = 27 → initial = 60 := by
  sorry

end gold_bars_problem_l2460_246076


namespace convex_curve_sum_containment_l2460_246078

/-- A convex curve in a 2D plane -/
structure ConvexCurve where
  points : Set (ℝ × ℝ)
  convex : sorry -- Add appropriate convexity condition

/-- The Minkowski sum of two convex curves -/
def minkowski_sum (K L : ConvexCurve) : ConvexCurve :=
  sorry

/-- One curve does not go beyond another -/
def not_beyond (K L : ConvexCurve) : Prop :=
  K.points ⊆ L.points

theorem convex_curve_sum_containment
  (K₁ K₂ L₁ L₂ : ConvexCurve)
  (h₁ : not_beyond K₁ L₁)
  (h₂ : not_beyond K₂ L₂) :
  not_beyond (minkowski_sum K₁ K₂) (minkowski_sum L₁ L₂) :=
sorry

end convex_curve_sum_containment_l2460_246078


namespace dannys_english_marks_l2460_246013

theorem dannys_english_marks 
  (math_marks : ℕ) 
  (physics_marks : ℕ) 
  (chemistry_marks : ℕ) 
  (biology_marks : ℕ) 
  (average_marks : ℕ) 
  (total_subjects : ℕ) 
  (h1 : math_marks = 65) 
  (h2 : physics_marks = 82) 
  (h3 : chemistry_marks = 67) 
  (h4 : biology_marks = 75) 
  (h5 : average_marks = 73) 
  (h6 : total_subjects = 5) : 
  ∃ (english_marks : ℕ), english_marks = 76 ∧ 
    (english_marks + math_marks + physics_marks + chemistry_marks + biology_marks) / total_subjects = average_marks :=
by
  sorry

end dannys_english_marks_l2460_246013


namespace nine_rooks_on_checkerboard_l2460_246034

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (is_checkerboard : Bool)

/-- Represents a rook placement on a chessboard -/
structure RookPlacement :=
  (board : Chessboard)
  (num_rooks : Nat)
  (same_color : Bool)
  (non_attacking : Bool)

/-- Counts the number of valid rook placements -/
def count_rook_placements (placement : RookPlacement) : Nat :=
  sorry

/-- Theorem: The number of ways to place 9 non-attacking rooks on cells of the same color on a 9x9 checkerboard is 2880 -/
theorem nine_rooks_on_checkerboard :
  ∀ (board : Chessboard) (placement : RookPlacement),
    board.size = 9 ∧
    board.is_checkerboard = true ∧
    placement.board = board ∧
    placement.num_rooks = 9 ∧
    placement.same_color = true ∧
    placement.non_attacking = true →
    count_rook_placements placement = 2880 :=
  sorry

end nine_rooks_on_checkerboard_l2460_246034


namespace lucy_fish_purchase_l2460_246020

theorem lucy_fish_purchase (current : ℕ) (desired : ℕ) (h1 : current = 212) (h2 : desired = 280) :
  desired - current = 68 := by
  sorry

end lucy_fish_purchase_l2460_246020


namespace even_multiples_of_45_l2460_246032

theorem even_multiples_of_45 :
  let lower_bound := 449
  let upper_bound := 990
  let count_even_multiples := (upper_bound - lower_bound) / (45 * 2)
  count_even_multiples = 6.022222222222222 := by
  sorry

end even_multiples_of_45_l2460_246032


namespace bulb_over_4000_hours_probability_l2460_246094

-- Define the probabilities
def prob_x : ℝ := 0.60  -- Probability of a bulb coming from factory X
def prob_y : ℝ := 1 - prob_x  -- Probability of a bulb coming from factory Y
def prob_x_over_4000 : ℝ := 0.59  -- Probability of factory X's bulb lasting over 4000 hours
def prob_y_over_4000 : ℝ := 0.65  -- Probability of factory Y's bulb lasting over 4000 hours

-- Define the theorem
theorem bulb_over_4000_hours_probability :
  prob_x * prob_x_over_4000 + prob_y * prob_y_over_4000 = 0.614 :=
by sorry

end bulb_over_4000_hours_probability_l2460_246094


namespace prop_q_indeterminate_l2460_246012

theorem prop_q_indeterminate (h1 : p ∨ q) (h2 : ¬(¬p)) : 
  (q ∨ ¬q) ∧ (∃ (v : Prop), v = q) :=
by sorry

end prop_q_indeterminate_l2460_246012


namespace binomial_18_10_l2460_246046

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 47190 := by
  sorry

end binomial_18_10_l2460_246046


namespace problem_solution_l2460_246009

theorem problem_solution (x y : ℝ) 
  (h1 : Real.sqrt (3 + Real.sqrt x) = 4) 
  (h2 : x + y = 58) : 
  y = -111 := by sorry

end problem_solution_l2460_246009


namespace max_sum_reciprocals_l2460_246048

theorem max_sum_reciprocals (p q r x y z : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hpqr : p + q + r = 2) (hxyz : x + y + z = 1) :
  1/(p+q) + 1/(p+r) + 1/(q+r) + 1/(x+y) + 1/(x+z) + 1/(y+z) ≤ 27/4 := by
sorry

end max_sum_reciprocals_l2460_246048


namespace smallest_number_l2460_246077

def A : ℕ := 36

def B : ℕ := 27 + 5

def C : ℕ := 3 * 10

def D : ℕ := 40 - 3

theorem smallest_number (h : A = 36 ∧ B = 27 + 5 ∧ C = 3 * 10 ∧ D = 40 - 3) :
  C ≤ A ∧ C ≤ B ∧ C ≤ D :=
by sorry

end smallest_number_l2460_246077


namespace alex_peeled_22_potatoes_l2460_246004

/-- The number of potatoes Alex peeled -/
def alexPotatoes (totalPotatoes : ℕ) (homerRate alextRate : ℕ) (alexJoinTime : ℕ) : ℕ :=
  let homerPotatoes := homerRate * alexJoinTime
  let remainingPotatoes := totalPotatoes - homerPotatoes
  let combinedRate := homerRate + alextRate
  let remainingTime := remainingPotatoes / combinedRate
  alextRate * remainingTime

theorem alex_peeled_22_potatoes :
  alexPotatoes 60 4 6 6 = 22 := by
  sorry

end alex_peeled_22_potatoes_l2460_246004


namespace remaining_days_temperature_l2460_246067

/-- Calculates the total temperature of the remaining days in a week given specific temperature conditions. -/
theorem remaining_days_temperature
  (avg_temp : ℝ)
  (days_in_week : ℕ)
  (first_three_temp : ℝ)
  (thursday_friday_temp : ℝ)
  (h1 : avg_temp = 60)
  (h2 : days_in_week = 7)
  (h3 : first_three_temp = 40)
  (h4 : thursday_friday_temp = 80) :
  (days_in_week : ℝ) * avg_temp - (3 * first_three_temp + 2 * thursday_friday_temp) = 140 := by
  sorry

#check remaining_days_temperature

end remaining_days_temperature_l2460_246067


namespace square_diagonal_from_rectangle_area_l2460_246096

theorem square_diagonal_from_rectangle_area (length width : ℝ) (h1 : length = 90) (h2 : width = 80) :
  let rectangle_area := length * width
  let square_side := (rectangle_area : ℝ).sqrt
  let square_diagonal := (2 * square_side ^ 2).sqrt
  square_diagonal = 120 := by sorry

end square_diagonal_from_rectangle_area_l2460_246096


namespace intersection_implies_a_less_than_two_l2460_246010

def A : Set ℝ := {1}
def B (a : ℝ) : Set ℝ := {x | a - 2*x < 0}

theorem intersection_implies_a_less_than_two (a : ℝ) : 
  (A ∩ B a).Nonempty → a < 2 := by
  sorry

end intersection_implies_a_less_than_two_l2460_246010


namespace largest_integer_in_special_set_l2460_246065

theorem largest_integer_in_special_set (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d →                   -- Four different integers
  (a + b + c + d) / 4 = 70 →                -- Average is 70
  a ≥ 13 →                                  -- Smallest integer is at least 13
  d ≤ 238 :=                                -- Largest integer is at most 238
by sorry

end largest_integer_in_special_set_l2460_246065


namespace carp_classification_l2460_246072

-- Define the characteristics of an individual
structure IndividualCharacteristics where
  birth : Bool
  death : Bool
  gender : Bool
  age : ℕ

-- Define the characteristics of a population
structure PopulationCharacteristics where
  birthRate : ℝ
  deathRate : ℝ
  genderRatio : ℝ
  ageComposition : List ℝ

-- Define the types
inductive EntityType
  | Carp
  | CarpPopulation

-- Define the main theorem
theorem carp_classification 
  (a : IndividualCharacteristics) 
  (b : PopulationCharacteristics) : 
  (EntityType.Carp, EntityType.CarpPopulation) = 
  (
    match a with
    | { birth := _, death := _, gender := _, age := _ } => EntityType.Carp,
    match b with
    | { birthRate := _, deathRate := _, genderRatio := _, ageComposition := _ } => EntityType.CarpPopulation
  ) := by
  sorry

end carp_classification_l2460_246072


namespace percentage_calculation_correct_l2460_246061

/-- The total number of students in the class -/
def total_students : ℕ := 30

/-- The number of students scoring in the 70%-79% range -/
def students_in_range : ℕ := 8

/-- The percentage of students scoring in the 70%-79% range -/
def percentage_in_range : ℚ := 26.67

/-- Theorem stating that the percentage of students scoring in the 70%-79% range is correct -/
theorem percentage_calculation_correct : 
  (students_in_range : ℚ) / (total_students : ℚ) * 100 = percentage_in_range := by
  sorry

end percentage_calculation_correct_l2460_246061


namespace inequality_implies_range_l2460_246079

theorem inequality_implies_range (a : ℝ) : 
  (∀ x ∈ Set.Icc (0 : ℝ) (1/2), 4^x + x - a ≤ 3/2) → a ∈ Set.Ici 1 := by
  sorry

end inequality_implies_range_l2460_246079


namespace quadratic_inequality_solution_set_l2460_246052

theorem quadratic_inequality_solution_set :
  let f : ℝ → ℝ := λ x => x^2 - 50*x + 601
  let solution_set := {x : ℝ | f x ≤ 9}
  let lower_bound := (50 - Real.sqrt 132) / 2
  let upper_bound := (50 + Real.sqrt 132) / 2
  solution_set = Set.Icc lower_bound upper_bound :=
by
  sorry

end quadratic_inequality_solution_set_l2460_246052


namespace pool_earnings_theorem_l2460_246086

def calculate_weekly_earnings (kid_fee : ℚ) (adult_fee : ℚ) (weekend_surcharge : ℚ) 
  (weekday_kids : ℕ) (weekday_adults : ℕ) (weekend_kids : ℕ) (weekend_adults : ℕ) 
  (weekdays : ℕ) (weekend_days : ℕ) : ℚ :=
  let weekday_earnings := (kid_fee * weekday_kids + adult_fee * weekday_adults) * weekdays
  let weekend_kid_fee := kid_fee * (1 + weekend_surcharge)
  let weekend_adult_fee := adult_fee * (1 + weekend_surcharge)
  let weekend_earnings := (weekend_kid_fee * weekend_kids + weekend_adult_fee * weekend_adults) * weekend_days
  weekday_earnings + weekend_earnings

theorem pool_earnings_theorem : 
  calculate_weekly_earnings 3 6 (1/2) 8 10 12 15 5 2 = 798 := by
  sorry

end pool_earnings_theorem_l2460_246086


namespace dans_candy_purchase_l2460_246006

def candy_problem (initial_money remaining_money candy_cost : ℕ) : Prop :=
  let spent_money := initial_money - remaining_money
  let num_candy_bars := spent_money / candy_cost
  num_candy_bars = 1

theorem dans_candy_purchase :
  candy_problem 4 1 3 := by sorry

end dans_candy_purchase_l2460_246006


namespace number_of_trucks_filled_l2460_246069

/-- Prove that the number of trucks filled up is 2, given the specified conditions. -/
theorem number_of_trucks_filled (service_cost : ℚ) (fuel_cost_per_liter : ℚ) (total_cost : ℚ)
  (num_minivans : ℕ) (minivan_capacity : ℚ) (truck_capacity_factor : ℚ) :
  service_cost = 23/10 →
  fuel_cost_per_liter = 7/10 →
  total_cost = 396 →
  num_minivans = 4 →
  minivan_capacity = 65 →
  truck_capacity_factor = 22/10 →
  ∃ (num_trucks : ℕ), num_trucks = 2 ∧
    total_cost = (num_minivans * (service_cost + fuel_cost_per_liter * minivan_capacity)) +
                 (num_trucks * (service_cost + fuel_cost_per_liter * (minivan_capacity * truck_capacity_factor))) :=
by sorry


end number_of_trucks_filled_l2460_246069


namespace not_perfect_square_l2460_246017

theorem not_perfect_square (n d : ℕ+) (h : d ∣ (2 * n^2)) :
  ¬ ∃ (x : ℕ), (n : ℕ)^2 + (d : ℕ) = x^2 := by
  sorry

end not_perfect_square_l2460_246017


namespace min_abs_z_l2460_246093

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z - Complex.I * 7) = 17) : 
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 56 / 17 := by
  sorry

end min_abs_z_l2460_246093


namespace sum_of_sixth_powers_of_roots_l2460_246037

theorem sum_of_sixth_powers_of_roots (p q : ℝ) : 
  p^2 - 3*p*Real.sqrt 3 + 3 = 0 → 
  q^2 - 3*q*Real.sqrt 3 + 3 = 0 → 
  p^6 + q^6 = 99171 := by
sorry

end sum_of_sixth_powers_of_roots_l2460_246037


namespace expression_evaluation_l2460_246063

theorem expression_evaluation (a b : ℚ) (h1 : a = 5) (h2 : b = 6) : 
  (3 * b) / (a + b) = 18 / 11 := by
  sorry

end expression_evaluation_l2460_246063


namespace latestPossibleTime_is_latest_l2460_246090

/-- Represents a time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60
  seconds_valid : seconds < 60

/-- Checks if a given time matches the visible digits pattern -/
def matchesVisibleDigits (t : Time) : Bool :=
  let h1 := t.hours / 10
  let h2 := t.hours % 10
  let m1 := t.minutes / 10
  let m2 := t.minutes % 10
  let s1 := t.seconds / 10
  let s2 := t.seconds % 10
  (h1 = 2 ∧ m1 = 0 ∧ s1 = 2 ∧ s2 = 2) ∨
  (h2 = 2 ∧ m1 = 0 ∧ s1 = 2 ∧ s2 = 2) ∨
  (h1 = 2 ∧ m2 = 0 ∧ s1 = 2 ∧ s2 = 2) ∨
  (h1 = 2 ∧ m1 = 0 ∧ s2 = 2 ∧ h2 = 2) ∨
  (h1 = 2 ∧ m1 = 0 ∧ s1 = 2 ∧ m2 = 2)

/-- The latest possible time satisfying the conditions -/
def latestPossibleTime : Time := {
  hours := 23
  minutes := 50
  seconds := 22
  hours_valid := by simp
  minutes_valid := by simp
  seconds_valid := by simp
}

/-- Theorem stating that the latestPossibleTime is indeed the latest time satisfying the conditions -/
theorem latestPossibleTime_is_latest :
  matchesVisibleDigits latestPossibleTime ∧
  ∀ t : Time, matchesVisibleDigits t → t.hours * 3600 + t.minutes * 60 + t.seconds ≤
    latestPossibleTime.hours * 3600 + latestPossibleTime.minutes * 60 + latestPossibleTime.seconds :=
by
  sorry


end latestPossibleTime_is_latest_l2460_246090


namespace absent_laborers_l2460_246089

/-- Proves that 6 laborers were absent given the problem conditions -/
theorem absent_laborers (total_laborers : ℕ) (planned_days : ℕ) (actual_days : ℕ)
  (h1 : total_laborers = 15)
  (h2 : planned_days = 9)
  (h3 : actual_days = 15)
  (h4 : total_laborers * planned_days = (total_laborers - absent) * actual_days) :
  absent = 6 := by
  sorry

end absent_laborers_l2460_246089


namespace tetrahedron_volume_is_4_sqrt_6_div_3_l2460_246031

/-- Tetrahedron with specific face angles and areas -/
structure Tetrahedron where
  /-- Face angle APB -/
  angle_APB : ℝ
  /-- Face angle BPC -/
  angle_BPC : ℝ
  /-- Face angle CPA -/
  angle_CPA : ℝ
  /-- Area of face PAB -/
  area_PAB : ℝ
  /-- Area of face PBC -/
  area_PBC : ℝ
  /-- Area of face PCA -/
  area_PCA : ℝ
  /-- All face angles are 60° -/
  angle_constraint : angle_APB = 60 ∧ angle_BPC = 60 ∧ angle_CPA = 60
  /-- Areas of faces are √3/2, 2, and 1 -/
  area_constraint : area_PAB = Real.sqrt 3 / 2 ∧ area_PBC = 2 ∧ area_PCA = 1

/-- Volume of a tetrahedron -/
noncomputable def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Theorem: The volume of the specified tetrahedron is 4√6/3 -/
theorem tetrahedron_volume_is_4_sqrt_6_div_3 (t : Tetrahedron) :
  tetrahedronVolume t = 4 * Real.sqrt 6 / 3 := by
  sorry


end tetrahedron_volume_is_4_sqrt_6_div_3_l2460_246031


namespace complex_equation_solution_l2460_246057

theorem complex_equation_solution (m : ℝ) : 
  (m - 1 : ℂ) + 2*m*Complex.I = 1 + 4*Complex.I → m = 2 := by
  sorry

end complex_equation_solution_l2460_246057


namespace arccos_cos_eight_l2460_246043

theorem arccos_cos_eight : Real.arccos (Real.cos 8) = 8 - 2 * Real.pi := by sorry

end arccos_cos_eight_l2460_246043


namespace unique_valid_number_l2460_246023

def is_valid_number (n : ℕ) : Prop :=
  350000 ≤ n ∧ n ≤ 359992 ∧ n % 100 = 2 ∧ n % 6 = 0

theorem unique_valid_number :
  ∃! n : ℕ, is_valid_number n ∧ n = 351152 := by sorry

end unique_valid_number_l2460_246023


namespace f_seven_equals_f_nine_l2460_246030

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f being decreasing on (8, +∞)
def DecreasingAfterEight (f : ℝ → ℝ) : Prop :=
  ∀ x y, x > 8 ∧ y > x → f y < f x

-- Define the property of f(x+8) being an even function
def EvenShiftedByEight (f : ℝ → ℝ) : Prop :=
  ∀ x, f (x + 8) = f (-x + 8)

-- State the theorem
theorem f_seven_equals_f_nine
  (h1 : DecreasingAfterEight f)
  (h2 : EvenShiftedByEight f) :
  f 7 = f 9 := by
  sorry

end f_seven_equals_f_nine_l2460_246030


namespace ten_steps_climb_ways_l2460_246098

/-- Number of ways to climb n steps when one can move to the next step or skip one step -/
def climbWays : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => climbWays (n + 1) + climbWays n

/-- The number of ways to climb 10 steps is 89 -/
theorem ten_steps_climb_ways : climbWays 10 = 89 := by sorry

end ten_steps_climb_ways_l2460_246098


namespace optimal_chair_removal_l2460_246022

theorem optimal_chair_removal (chairs_per_row : ℕ) (total_chairs : ℕ) (expected_participants : ℕ) 
  (h1 : chairs_per_row = 15)
  (h2 : total_chairs = 225)
  (h3 : expected_participants = 140) :
  let chairs_to_remove := 75
  let remaining_chairs := total_chairs - chairs_to_remove
  (remaining_chairs % chairs_per_row = 0) ∧ 
  (remaining_chairs ≥ expected_participants) ∧
  (∀ n : ℕ, n < chairs_to_remove → 
    (total_chairs - n) % chairs_per_row ≠ 0 ∨ 
    (total_chairs - n < expected_participants)) :=
by sorry

end optimal_chair_removal_l2460_246022


namespace total_cost_calculation_l2460_246033

def muffin_cost : ℚ := 0.75
def juice_cost : ℚ := 1.45
def muffin_count : ℕ := 3

theorem total_cost_calculation : 
  (muffin_count : ℚ) * muffin_cost + juice_cost = 3.70 := by
  sorry

end total_cost_calculation_l2460_246033
