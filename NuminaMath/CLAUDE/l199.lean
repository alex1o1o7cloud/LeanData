import Mathlib

namespace NUMINAMATH_CALUDE_wait_ratio_l199_19919

def total_time : ℕ := 180
def uber_to_house : ℕ := 10
def check_bag : ℕ := 15
def wait_for_boarding : ℕ := 20

def uber_to_airport : ℕ := 5 * uber_to_house
def security : ℕ := 3 * check_bag

def time_before_takeoff : ℕ := 
  uber_to_house + uber_to_airport + check_bag + security + wait_for_boarding

def wait_before_takeoff : ℕ := total_time - time_before_takeoff

theorem wait_ratio : 
  wait_before_takeoff = 2 * wait_for_boarding :=
sorry

end NUMINAMATH_CALUDE_wait_ratio_l199_19919


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l199_19935

theorem floor_equation_solutions (x y : ℝ) :
  (∀ n : ℕ+, x * ⌊n * y⌋ = y * ⌊n * x⌋) ↔
  (x = y ∨ x = 0 ∨ y = 0 ∨ (∃ a b : ℤ, x = a ∧ y = b)) :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l199_19935


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l199_19966

theorem largest_divisor_of_n_squared_divisible_by_72 (n : ℕ) (h1 : n > 0) (h2 : 72 ∣ n^2) :
  ∃ q : ℕ, q > 0 ∧ q ∣ n ∧ ∀ m : ℕ, m > 0 ∧ m ∣ n → m ≤ q ∧ q = 12 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_divisible_by_72_l199_19966


namespace NUMINAMATH_CALUDE_lines_one_unit_from_origin_l199_19997

theorem lines_one_unit_from_origin (x y : ℝ) (y' : ℝ → ℝ) :
  (∀ α : ℝ, x * Real.cos α + y * Real.sin α = 1) ↔
  y = x * y' x + Real.sqrt (1 + (y' x)^2) :=
sorry

end NUMINAMATH_CALUDE_lines_one_unit_from_origin_l199_19997


namespace NUMINAMATH_CALUDE_g_composed_has_two_distinct_roots_l199_19982

/-- The function g(x) = x^2 + 4x + d -/
def g (d : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + d

/-- The composition of g with itself -/
def g_composed (d : ℝ) (x : ℝ) : ℝ := g d (g d x)

/-- Theorem stating that g(g(x)) has exactly 2 distinct real roots when d = 8 -/
theorem g_composed_has_two_distinct_roots :
  ∃! (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ 
  (∀ x : ℝ, g_composed 8 x = 0 ↔ x = r₁ ∨ x = r₂) :=
sorry

end NUMINAMATH_CALUDE_g_composed_has_two_distinct_roots_l199_19982


namespace NUMINAMATH_CALUDE_july_rainfall_l199_19934

theorem july_rainfall (march april may june : ℝ) (h1 : march = 3.79) (h2 : april = 4.5) 
  (h3 : may = 3.95) (h4 : june = 3.09) (h5 : (march + april + may + june + july) / 5 = 4) : 
  july = 4.67 := by
  sorry

end NUMINAMATH_CALUDE_july_rainfall_l199_19934


namespace NUMINAMATH_CALUDE_max_m_inequality_l199_19977

theorem max_m_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 2 / a + 1 / b = 1 / 4) : 
  (∃ (m : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → 2 / x + 1 / y = 1 / 4 → 2*x + y ≥ 4*m) ∧ 
               (∀ (ε : ℝ), ε > 0 → ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 2 / x + 1 / y = 1 / 4 ∧ 2*x + y < 4*(m + ε))) ∧
  (∀ (n : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → 2 / x + 1 / y = 1 / 4 → 2*x + y ≥ 4*n) → n ≤ m) ∧
  m = 9 :=
sorry

end NUMINAMATH_CALUDE_max_m_inequality_l199_19977


namespace NUMINAMATH_CALUDE_quadratic_solution_l199_19926

theorem quadratic_solution (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h1 : p^2 + 2*p*p + q = 0)
  (h2 : q^2 + 2*p*q + q = 0) :
  p = 1 ∧ q = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_l199_19926


namespace NUMINAMATH_CALUDE_vector_subtraction_l199_19976

/-- Given two vectors OM and ON in R², prove that MN = ON - OM -/
theorem vector_subtraction (OM ON : ℝ × ℝ) (h1 : OM = (3, -2)) (h2 : ON = (-5, -1)) :
  ON - OM = (-8, 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l199_19976


namespace NUMINAMATH_CALUDE_crates_lost_l199_19915

/-- Proves the number of crates lost given initial conditions --/
theorem crates_lost (initial_crates : ℕ) (total_cost : ℚ) (selling_price : ℚ) (profit_percentage : ℚ) : 
  initial_crates = 10 →
  total_cost = 160 →
  selling_price = 25 →
  profit_percentage = 25 / 100 →
  ∃ (lost_crates : ℕ), lost_crates = 2 ∧ 
    selling_price * (initial_crates - lost_crates) = total_cost * (1 + profit_percentage) :=
by sorry

end NUMINAMATH_CALUDE_crates_lost_l199_19915


namespace NUMINAMATH_CALUDE_smallest_possible_a_l199_19909

theorem smallest_possible_a (a b : ℤ) (x : ℝ) (h1 : a > x) (h2 : a < 41)
  (h3 : b > 39) (h4 : b < 51)
  (h5 : (↑40 / ↑40 : ℚ) - (↑a / ↑50 : ℚ) = 2/5) : a ≥ 30 := by
  sorry

end NUMINAMATH_CALUDE_smallest_possible_a_l199_19909


namespace NUMINAMATH_CALUDE_construction_costs_l199_19984

/-- Calculate the total construction costs for a house project. -/
theorem construction_costs
  (land_cost_per_sqm : ℝ)
  (brick_cost_per_1000 : ℝ)
  (tile_cost_per_tile : ℝ)
  (land_area : ℝ)
  (brick_count : ℝ)
  (tile_count : ℝ)
  (h1 : land_cost_per_sqm = 50)
  (h2 : brick_cost_per_1000 = 100)
  (h3 : tile_cost_per_tile = 10)
  (h4 : land_area = 2000)
  (h5 : brick_count = 10000)
  (h6 : tile_count = 500) :
  land_cost_per_sqm * land_area +
  brick_cost_per_1000 * (brick_count / 1000) +
  tile_cost_per_tile * tile_count = 106000 := by
  sorry


end NUMINAMATH_CALUDE_construction_costs_l199_19984


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l199_19923

theorem quadratic_roots_sum_product (m : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 + m*x - 3 = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ + x₂ - x₁*x₂ = 5 →
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l199_19923


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l199_19950

theorem problem_1 : (1) - 1/2 / 3 * (3 - (-3)^2) = 1 := by sorry

theorem problem_2 (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) : 
  2*x / (x^2 - 4) - 1 / (x - 2) = 1 / (x + 2) := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l199_19950


namespace NUMINAMATH_CALUDE_arthurs_spending_l199_19999

/-- The cost of Arthur's purchase on the first day -/
def arthurs_first_day_cost (hamburger_price hot_dog_price : ℝ) : ℝ :=
  3 * hamburger_price + 4 * hot_dog_price

/-- The cost of Arthur's purchase on the second day -/
def arthurs_second_day_cost (hamburger_price hot_dog_price : ℝ) : ℝ :=
  2 * hamburger_price + 3 * hot_dog_price

theorem arthurs_spending : 
  ∀ (hamburger_price : ℝ),
    arthurs_second_day_cost hamburger_price 1 = 7 →
    arthurs_first_day_cost hamburger_price 1 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arthurs_spending_l199_19999


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l199_19920

/-- 
Given a right triangle PQR with legs PQ and PR, where U is on PQ and V is on PR,
prove that if PU:UQ = PV:VR = 1:3, QU = 18 units, and RV = 45 units, 
then the length of the hypotenuse QR is 12√29 units.
-/
theorem right_triangle_hypotenuse (P Q R U V : ℝ × ℝ) : 
  let pq := ‖Q - P‖
  let pr := ‖R - P‖
  let qu := ‖U - Q‖
  let rv := ‖V - R‖
  let qr := ‖R - Q‖
  (P.1 - Q.1) * (R.2 - P.2) = (P.2 - Q.2) * (R.1 - P.1) → -- right angle at P
  (∃ t : ℝ, t > 0 ∧ t < 1 ∧ U = t • P + (1 - t) • Q) → -- U is on PQ
  (∃ s : ℝ, s > 0 ∧ s < 1 ∧ V = s • P + (1 - s) • R) → -- V is on PR
  ‖P - U‖ / ‖U - Q‖ = 1 / 3 → -- PU:UQ = 1:3
  ‖P - V‖ / ‖V - R‖ = 1 / 3 → -- PV:VR = 1:3
  qu = 18 →
  rv = 45 →
  qr = 12 * Real.sqrt 29 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l199_19920


namespace NUMINAMATH_CALUDE_lisa_decorative_spoons_l199_19903

/-- The number of children Lisa has -/
def num_children : ℕ := 4

/-- The number of baby spoons each child had -/
def baby_spoons_per_child : ℕ := 3

/-- The number of large spoons in the new cutlery set -/
def new_large_spoons : ℕ := 10

/-- The number of teaspoons in the new cutlery set -/
def new_teaspoons : ℕ := 15

/-- The total number of spoons Lisa has now -/
def total_spoons : ℕ := 39

/-- The number of decorative spoons Lisa created -/
def decorative_spoons : ℕ := total_spoons - (new_large_spoons + new_teaspoons) - (num_children * baby_spoons_per_child)

theorem lisa_decorative_spoons : decorative_spoons = 2 := by
  sorry

end NUMINAMATH_CALUDE_lisa_decorative_spoons_l199_19903


namespace NUMINAMATH_CALUDE_bellas_age_l199_19904

theorem bellas_age : 
  ∀ (bella_age : ℕ), 
  (bella_age + (bella_age + 9) + (bella_age / 2) = 27) → 
  bella_age = 6 := by
sorry

end NUMINAMATH_CALUDE_bellas_age_l199_19904


namespace NUMINAMATH_CALUDE_max_k_value_l199_19956

theorem max_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 17 = 0 ∧ y^2 + k*y + 17 = 0 ∧ |x - y| = Real.sqrt 85) →
  k ≤ Real.sqrt 153 :=
sorry

end NUMINAMATH_CALUDE_max_k_value_l199_19956


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l199_19994

-- Define the sample space
def Ω : Finset Nat := {1, 2, 3, 4, 5, 6}

-- Define events A and B
def A : Finset Nat := {2, 3, 5}
def B : Finset Nat := {1, 2, 4, 5, 6}

-- Define the probability measure
noncomputable def P (S : Finset Nat) : ℝ := (S.card : ℝ) / (Ω.card : ℝ)

-- State the theorem
theorem conditional_probability_B_given_A : P (A ∩ B) / P A = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l199_19994


namespace NUMINAMATH_CALUDE_triangle_max_perimeter_l199_19992

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin x ^ 2 + Real.sin x * Real.cos x - Real.sqrt 3 / 2

theorem triangle_max_perimeter 
  (A : ℝ) 
  (h_acute : 0 < A ∧ A < π / 2)
  (h_f_A : f A = Real.sqrt 3 / 2)
  (h_a : ∀ (a b c : ℝ), a = 2 → a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A) :
  ∃ (b c : ℝ), 2 + b + c ≤ 6 ∧ 
    ∀ (b' c' : ℝ), 2 + b' + c' ≤ 2 + b + c := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_perimeter_l199_19992


namespace NUMINAMATH_CALUDE_smallest_angle_in_triangle_l199_19942

theorem smallest_angle_in_triangle (a b c : ℝ) (C : ℝ) : 
  a = 2 →
  b = 2 →
  c ≥ 4 →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  C ≥ 120 * Real.pi / 180 :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_in_triangle_l199_19942


namespace NUMINAMATH_CALUDE_square_perimeter_l199_19944

/-- The perimeter of a square with side length 19 cm is 76 cm. -/
theorem square_perimeter : 
  ∀ (s : ℝ), s = 19 → 4 * s = 76 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l199_19944


namespace NUMINAMATH_CALUDE_perfect_square_from_condition_l199_19929

theorem perfect_square_from_condition (a b c : ℤ) (h : a * b + b * c + c * a = 1) :
  ∃ n : ℤ, (1 + a^2) * (1 + b^2) * (1 + c^2) = n^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_from_condition_l199_19929


namespace NUMINAMATH_CALUDE_smallest_rectangular_block_l199_19906

theorem smallest_rectangular_block (l m n : ℕ) : 
  (l - 1) * (m - 1) * (n - 1) = 462 → 
  l * m * n ≥ 672 ∧ 
  ∃ (l' m' n' : ℕ), (l' - 1) * (m' - 1) * (n' - 1) = 462 ∧ l' * m' * n' = 672 :=
by sorry

end NUMINAMATH_CALUDE_smallest_rectangular_block_l199_19906


namespace NUMINAMATH_CALUDE_waiter_customers_l199_19928

/-- Given a number of customers who left and the number of remaining customers,
    calculate the initial number of customers. -/
def initial_customers (left : ℕ) (remaining : ℕ) : ℕ := left + remaining

/-- Theorem: Given that 9 customers left and 12 remained, 
    prove that there were initially 21 customers. -/
theorem waiter_customers : initial_customers 9 12 = 21 := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l199_19928


namespace NUMINAMATH_CALUDE_cyclic_quadrilateral_inequality_l199_19946

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
def CyclicQuadrilateral (A B C D : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ) (r : ℝ), 
    dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O D = r

/-- The theorem states that for a cyclic quadrilateral ABCD, 
    the sum of the absolute differences between opposite sides 
    is greater than or equal to twice the absolute difference between the diagonals. -/
theorem cyclic_quadrilateral_inequality 
  (A B C D : ℝ × ℝ) 
  (h : CyclicQuadrilateral A B C D) : 
  |dist A B - dist C D| + |dist A D - dist B C| ≥ 2 * |dist A C - dist B D| :=
sorry

end NUMINAMATH_CALUDE_cyclic_quadrilateral_inequality_l199_19946


namespace NUMINAMATH_CALUDE_m_intersect_n_l199_19986

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = a^2}

theorem m_intersect_n : M ∩ N = {0, 1} := by sorry

end NUMINAMATH_CALUDE_m_intersect_n_l199_19986


namespace NUMINAMATH_CALUDE_least_difference_l199_19974

theorem least_difference (x y z : ℤ) : 
  x < y ∧ y < z ∧ 
  y - x > 3 ∧
  Even x ∧ Odd y ∧ Odd z →
  ∀ w, w = z - x → w ≥ 7 := by
sorry

end NUMINAMATH_CALUDE_least_difference_l199_19974


namespace NUMINAMATH_CALUDE_min_value_expression_l199_19960

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_prod : x * y * z = 108) : 
  x^2 + 9*x*y + 9*y^2 + 3*z^2 ≥ 324 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l199_19960


namespace NUMINAMATH_CALUDE_track_completion_time_l199_19954

/-- Represents a circular running track --/
structure Track where
  circumference : ℝ
  circumference_positive : circumference > 0

/-- Represents a runner on the track --/
structure Runner where
  speed : ℝ
  speed_positive : speed > 0

/-- Represents an event where two runners meet --/
structure MeetingEvent where
  time : ℝ
  time_nonnegative : time ≥ 0

/-- The main theorem to prove --/
theorem track_completion_time
  (track : Track)
  (runner1 runner2 runner3 : Runner)
  (meeting12 : MeetingEvent)
  (meeting23 : MeetingEvent)
  (meeting31 : MeetingEvent)
  (h1 : meeting23.time - meeting12.time = 15)
  (h2 : meeting31.time - meeting23.time = 25) :
  track.circumference / runner1.speed = 80 :=
sorry

end NUMINAMATH_CALUDE_track_completion_time_l199_19954


namespace NUMINAMATH_CALUDE_prob_different_suits_78_card_deck_l199_19980

/-- A custom deck of cards -/
structure CustomDeck where
  total_cards : ℕ
  num_suits : ℕ
  cards_per_suit : ℕ
  total_cards_eq : total_cards = num_suits * cards_per_suit

/-- The probability of drawing two cards of different suits from a custom deck -/
def prob_different_suits (deck : CustomDeck) : ℚ :=
  let remaining_cards := deck.total_cards - 1
  let cards_different_suit := (deck.num_suits - 1) * deck.cards_per_suit
  cards_different_suit / remaining_cards

/-- The main theorem stating the probability for the specific deck -/
theorem prob_different_suits_78_card_deck :
  ∃ (deck : CustomDeck),
    deck.total_cards = 78 ∧
    deck.num_suits = 6 ∧
    deck.cards_per_suit = 13 ∧
    prob_different_suits deck = 65 / 77 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_suits_78_card_deck_l199_19980


namespace NUMINAMATH_CALUDE_tree_planting_equation_holds_l199_19933

/-- Represents the tree planting project with increased efficiency -/
structure TreePlantingProject where
  total_trees : ℕ
  efficiency_increase : ℝ
  days_ahead : ℕ
  trees_per_day : ℝ

/-- The equation holds for the given tree planting project -/
theorem tree_planting_equation_holds (project : TreePlantingProject) 
  (h1 : project.total_trees = 20000)
  (h2 : project.efficiency_increase = 0.25)
  (h3 : project.days_ahead = 5) :
  project.total_trees / project.trees_per_day - 
  project.total_trees / (project.trees_per_day * (1 + project.efficiency_increase)) = 
  project.days_ahead := by
  sorry

end NUMINAMATH_CALUDE_tree_planting_equation_holds_l199_19933


namespace NUMINAMATH_CALUDE_tangent_line_and_max_value_l199_19962

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 2*x) * Real.log x + a*x^2 + 2

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f a x - x - 2

theorem tangent_line_and_max_value :
  (∀ x : ℝ, x > 0 → (3*x + f (-1) x - 4 = 0 ↔ x = 1)) ∧
  (∀ a : ℝ, a > 0 →
    (∃! x : ℝ, g a x = 0) →
    (∀ x : ℝ, Real.exp (-2) < x → x < Real.exp 1 → g a x ≤ 2 * Real.exp 2 - 3 * Real.exp 1) ∧
    (∃ x : ℝ, Real.exp (-2) < x ∧ x < Real.exp 1 ∧ g a x = 2 * Real.exp 2 - 3 * Real.exp 1)) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_and_max_value_l199_19962


namespace NUMINAMATH_CALUDE_remainder_777_power_777_mod_13_l199_19916

theorem remainder_777_power_777_mod_13 : 777^777 ≡ 12 [ZMOD 13] := by
  sorry

end NUMINAMATH_CALUDE_remainder_777_power_777_mod_13_l199_19916


namespace NUMINAMATH_CALUDE_probability_all_odd_is_one_forty_second_l199_19936

def total_slips : ℕ := 10
def odd_slips : ℕ := 5
def draws : ℕ := 4

def probability_all_odd : ℚ := (odd_slips.choose draws) / (total_slips.choose draws)

theorem probability_all_odd_is_one_forty_second :
  probability_all_odd = 1 / 42 := by sorry

end NUMINAMATH_CALUDE_probability_all_odd_is_one_forty_second_l199_19936


namespace NUMINAMATH_CALUDE_vector_triangle_rule_l199_19969

-- Define a triangle ABC in a vector space
variable {V : Type*} [AddCommGroup V]
variable (A B C : V)

-- State the theorem
theorem vector_triangle_rule :
  (C - A) - (B - A) + (B - C) = (0 : V) := by
  sorry

end NUMINAMATH_CALUDE_vector_triangle_rule_l199_19969


namespace NUMINAMATH_CALUDE_rectangle_length_l199_19925

theorem rectangle_length (w l : ℝ) (h1 : w > 0) (h2 : l > 0) : 
  (2*l + 2*w) / w = 5 → l * w = 150 → l = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_l199_19925


namespace NUMINAMATH_CALUDE_watch_cost_price_l199_19913

/-- The cost price of a watch satisfying certain conditions -/
theorem watch_cost_price : ∃ (cp : ℝ), 
  (cp > 0) ∧ 
  (0.80 * cp + 520 = 1.06 * cp) ∧ 
  (cp = 2000) := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l199_19913


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l199_19958

theorem sqrt_equation_solution (y : ℝ) : Real.sqrt (y + 5) = 7 → y = 44 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l199_19958


namespace NUMINAMATH_CALUDE_april_largest_difference_l199_19911

/-- Represents the months of cookie sales --/
inductive Month
| january
| february
| march
| april
| may

/-- Calculates the percentage difference between two sales values --/
def percentageDifference (x y : ℕ) : ℚ :=
  (max x y - min x y : ℚ) / (min x y : ℚ) * 100

/-- Returns the sales data for Rangers and Scouts for a given month --/
def salesData (m : Month) : ℕ × ℕ :=
  match m with
  | .january => (5, 4)
  | .february => (6, 4)
  | .march => (5, 5)
  | .april => (7, 4)
  | .may => (3, 5)

/-- Theorem: April has the largest percentage difference in cookie sales --/
theorem april_largest_difference :
  ∀ m : Month, m ≠ Month.april →
    percentageDifference (salesData Month.april).1 (salesData Month.april).2 ≥
    percentageDifference (salesData m).1 (salesData m).2 :=
by sorry

end NUMINAMATH_CALUDE_april_largest_difference_l199_19911


namespace NUMINAMATH_CALUDE_coin_toss_probability_l199_19996

theorem coin_toss_probability (p_heads : ℚ) (h1 : p_heads = 1/4) :
  1 - p_heads = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_probability_l199_19996


namespace NUMINAMATH_CALUDE_solution_set_inequality_l199_19983

theorem solution_set_inequality (x : ℝ) : 
  (2*x - 1) / (3*x + 1) > 1 ↔ -2 < x ∧ x < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l199_19983


namespace NUMINAMATH_CALUDE_max_triangles_is_eleven_l199_19989

/-- Represents an equilateral triangle with a line segment connecting the midpoints of two sides -/
structure EquilateralTriangleWithMidline where
  side_length : ℝ
  midline_position : ℝ

/-- Represents the configuration of two overlapping equilateral triangles -/
structure OverlappingTriangles where
  triangle_a : EquilateralTriangleWithMidline
  triangle_b : EquilateralTriangleWithMidline
  overlap_distance : ℝ

/-- Counts the number of triangles formed in a given configuration -/
def count_triangles (config : OverlappingTriangles) : ℕ :=
  sorry

/-- Finds the maximum number of triangles formed during the overlap process -/
def max_triangles (triangle : EquilateralTriangleWithMidline) : ℕ :=
  sorry

/-- Main theorem: The maximum number of triangles formed is 11 -/
theorem max_triangles_is_eleven (triangle : EquilateralTriangleWithMidline) :
  max_triangles triangle = 11 :=
sorry

end NUMINAMATH_CALUDE_max_triangles_is_eleven_l199_19989


namespace NUMINAMATH_CALUDE_difference_c_minus_a_l199_19990

theorem difference_c_minus_a (a b c : ℝ) 
  (h1 : (a + b) / 2 = 45)
  (h2 : (b + c) / 2 = 90) : 
  c - a = 90 := by
sorry

end NUMINAMATH_CALUDE_difference_c_minus_a_l199_19990


namespace NUMINAMATH_CALUDE_roots_sum_and_product_l199_19963

theorem roots_sum_and_product (c d : ℝ) : 
  c^2 - 6*c + 8 = 0 → d^2 - 6*d + 8 = 0 → c^3 + c^4*d^2 + c^2*d^4 + d^3 = 1352 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_and_product_l199_19963


namespace NUMINAMATH_CALUDE_notebook_puzzle_l199_19922

/-- Represents a set of statements where the i-th statement claims 
    "There are exactly i false statements in this set" --/
def StatementSet (n : ℕ) := Fin n → Prop

/-- The property that exactly one statement in the set is true --/
def ExactlyOneTrue (s : StatementSet n) : Prop :=
  ∃! i, s i

/-- The i-th statement claims there are exactly i false statements --/
def StatementClaim (s : StatementSet n) (i : Fin n) : Prop :=
  s i ↔ (∃ k : Fin n, k.val = n - i.val ∧ (∀ j : Fin n, s j ↔ j = k))

/-- The main theorem --/
theorem notebook_puzzle :
  ∀ (s : StatementSet 100),
    (∀ i, StatementClaim s i) →
    ExactlyOneTrue s →
    s ⟨99, by norm_num⟩ :=
by sorry

end NUMINAMATH_CALUDE_notebook_puzzle_l199_19922


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l199_19931

theorem binomial_expansion_coefficient (x : ℝ) :
  let expansion := (1 + 2*x)^5
  ∃ a₀ a₁ a₂ a₃ a₄ a₅ : ℝ,
    expansion = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 ∧
    a₃ = 80 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l199_19931


namespace NUMINAMATH_CALUDE_men_left_hostel_l199_19945

/-- Proves that 50 men left the hostel given the initial and final conditions -/
theorem men_left_hostel (initial_men : ℕ) (initial_days : ℕ) (final_days : ℕ) 
  (h1 : initial_men = 250)
  (h2 : initial_days = 40)
  (h3 : final_days = 50)
  (h4 : initial_men * initial_days = (initial_men - men_left) * final_days) :
  men_left = 50 := by
  sorry

#check men_left_hostel

end NUMINAMATH_CALUDE_men_left_hostel_l199_19945


namespace NUMINAMATH_CALUDE_cubic_sum_equals_265_l199_19939

theorem cubic_sum_equals_265 (a b : ℝ) (h1 : a + b = 5) (h2 : a * b = -14) :
  a^3 + a^2*b + a*b^2 + b^3 = 265 := by sorry

end NUMINAMATH_CALUDE_cubic_sum_equals_265_l199_19939


namespace NUMINAMATH_CALUDE_smallest_binary_divisible_by_product_l199_19968

def is_binary_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

def product_of_first_six : ℕ := (List.range 6).map (· + 1) |>.prod

theorem smallest_binary_divisible_by_product :
  let n : ℕ := 1111111110000
  (is_binary_number n) ∧
  (n % product_of_first_six = 0) ∧
  (∀ m : ℕ, m < n → is_binary_number m → m % product_of_first_six ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_smallest_binary_divisible_by_product_l199_19968


namespace NUMINAMATH_CALUDE_probability_three_in_same_group_l199_19973

/-- The number of people to be partitioned -/
def total_people : ℕ := 15

/-- The number of groups -/
def num_groups : ℕ := 6

/-- The sizes of the groups -/
def group_sizes : List ℕ := [3, 3, 3, 2, 2, 2]

/-- The number of people we're interested in (Petruk, Gareng, and Bagong) -/
def num_interested : ℕ := 3

/-- The probability that Petruk, Gareng, and Bagong are in the same group -/
def probability_same_group : ℚ := 3 / 455

theorem probability_three_in_same_group :
  let total_ways := (total_people.factorial) / (group_sizes.map Nat.factorial).prod
  let favorable_ways := 3 * ((total_people - num_interested).factorial) / 
    ((group_sizes.tail.map Nat.factorial).prod)
  (favorable_ways : ℚ) / total_ways = probability_same_group := by
  sorry

end NUMINAMATH_CALUDE_probability_three_in_same_group_l199_19973


namespace NUMINAMATH_CALUDE_six_digit_divisibility_theorem_l199_19901

/-- Represents a 6-digit number in the form 739ABC -/
def SixDigitNumber (a b c : Nat) : Nat :=
  739000 + 100 * a + 10 * b + c

/-- Checks if a number is divisible by 7, 8, and 9 -/
def isDivisibleBy789 (n : Nat) : Prop :=
  n % 7 = 0 ∧ n % 8 = 0 ∧ n % 9 = 0

/-- The main theorem stating the possible values for A, B, and C -/
theorem six_digit_divisibility_theorem :
  ∀ a b c : Nat,
  a < 10 ∧ b < 10 ∧ c < 10 →
  isDivisibleBy789 (SixDigitNumber a b c) →
  (a = 3 ∧ b = 6 ∧ c = 8) ∨ (a = 8 ∧ b = 7 ∧ c = 2) :=
by sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_theorem_l199_19901


namespace NUMINAMATH_CALUDE_tax_center_revenue_l199_19985

/-- Calculates the total revenue for a tax center based on the number and types of returns sold --/
theorem tax_center_revenue (federal_price state_price quarterly_price : ℕ)
                           (federal_sold state_sold quarterly_sold : ℕ) :
  federal_price = 50 →
  state_price = 30 →
  quarterly_price = 80 →
  federal_sold = 60 →
  state_sold = 20 →
  quarterly_sold = 10 →
  federal_price * federal_sold + state_price * state_sold + quarterly_price * quarterly_sold = 4400 :=
by sorry

end NUMINAMATH_CALUDE_tax_center_revenue_l199_19985


namespace NUMINAMATH_CALUDE_sanya_towels_count_l199_19975

/-- The number of bath towels Sanya can wash in one wash -/
def towels_per_wash : ℕ := 7

/-- The number of hours Sanya has per day for washing -/
def hours_per_day : ℕ := 2

/-- The number of days it takes to wash all towels -/
def days_to_wash_all : ℕ := 7

/-- The total number of bath towels Sanya has -/
def total_towels : ℕ := towels_per_wash * hours_per_day * days_to_wash_all

theorem sanya_towels_count : total_towels = 98 := by
  sorry

end NUMINAMATH_CALUDE_sanya_towels_count_l199_19975


namespace NUMINAMATH_CALUDE_onion_harvest_bags_per_trip_l199_19959

/-- Calculates the number of bags carried per trip given the total harvest weight,
    weight per bag, and number of trips. -/
def bagsPerTrip (totalHarvest : ℕ) (weightPerBag : ℕ) (numTrips : ℕ) : ℕ :=
  (totalHarvest / weightPerBag) / numTrips

/-- Theorem stating that given the specific conditions of Titan's father's onion harvest,
    the number of bags carried per trip is 10. -/
theorem onion_harvest_bags_per_trip :
  bagsPerTrip 10000 50 20 = 10 := by
  sorry

#eval bagsPerTrip 10000 50 20

end NUMINAMATH_CALUDE_onion_harvest_bags_per_trip_l199_19959


namespace NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l199_19971

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_12th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 7 + a 9 = 16)
  (h_4th : a 4 = 1) :
  a 12 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_12th_term_l199_19971


namespace NUMINAMATH_CALUDE_sequence_formula_correct_l199_19952

def a (n : ℕ) : ℤ := (-1)^n * (4*n - 1)

theorem sequence_formula_correct :
  (a 1 = -3) ∧ (a 2 = 7) ∧ (a 3 = -11) ∧ (a 4 = 15) :=
by sorry

end NUMINAMATH_CALUDE_sequence_formula_correct_l199_19952


namespace NUMINAMATH_CALUDE_mias_test_score_l199_19995

theorem mias_test_score (total_students : ℕ) (initial_average : ℚ) (average_after_ethan : ℚ) (final_average : ℚ) :
  total_students = 20 →
  initial_average = 84 →
  average_after_ethan = 85 →
  final_average = 86 →
  (total_students * final_average - (total_students - 1) * average_after_ethan : ℚ) = 105 := by
  sorry

end NUMINAMATH_CALUDE_mias_test_score_l199_19995


namespace NUMINAMATH_CALUDE_floor_double_floor_eq_42_l199_19900

theorem floor_double_floor_eq_42 (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 42 ↔ 7 ≤ x ∧ x < 43/6 := by
  sorry

end NUMINAMATH_CALUDE_floor_double_floor_eq_42_l199_19900


namespace NUMINAMATH_CALUDE_neg_i_cubed_l199_19947

theorem neg_i_cubed (i : ℂ) (h : i^2 = -1) : (-i)^3 = -i := by
  sorry

end NUMINAMATH_CALUDE_neg_i_cubed_l199_19947


namespace NUMINAMATH_CALUDE_new_supervisor_salary_is_960_l199_19961

/-- Represents the monthly salary structure of a factory -/
structure FactorySalary where
  initial_avg : ℝ
  old_supervisor_salary : ℝ
  old_supervisor_bonus_rate : ℝ
  worker_increment_rate : ℝ
  old_supervisor_increment_rate : ℝ
  new_avg : ℝ
  new_supervisor_bonus_rate : ℝ
  new_supervisor_increment_rate : ℝ

/-- Calculates the new supervisor's monthly salary -/
def calculate_new_supervisor_salary (fs : FactorySalary) : ℝ :=
  sorry

/-- Theorem stating that given the factory salary conditions, 
    the new supervisor's monthly salary is $960 -/
theorem new_supervisor_salary_is_960 (fs : FactorySalary) 
  (h1 : fs.initial_avg = 430)
  (h2 : fs.old_supervisor_salary = 870)
  (h3 : fs.old_supervisor_bonus_rate = 0.05)
  (h4 : fs.worker_increment_rate = 0.03)
  (h5 : fs.old_supervisor_increment_rate = 0.04)
  (h6 : fs.new_avg = 450)
  (h7 : fs.new_supervisor_bonus_rate = 0.03)
  (h8 : fs.new_supervisor_increment_rate = 0.035) :
  calculate_new_supervisor_salary fs = 960 :=
sorry

end NUMINAMATH_CALUDE_new_supervisor_salary_is_960_l199_19961


namespace NUMINAMATH_CALUDE_youngest_child_age_l199_19988

/-- Represents a family with its members and ages -/
structure Family where
  members : Nat
  total_age : Nat

/-- Calculates the average age of a family -/
def average_age (f : Family) : Nat :=
  f.total_age / f.members

theorem youngest_child_age :
  let initial_family : Family := { members := 4, total_age := 96 }
  let current_family : Family := { members := 6, total_age := 144 }
  let age_difference : Nat := 2
  average_age initial_family = 24 →
  average_age current_family = 24 →
  ∃ (youngest_age : Nat),
    youngest_age = 3 ∧
    youngest_age + (youngest_age + age_difference) = current_family.total_age - (initial_family.total_age + 40) :=
by sorry

end NUMINAMATH_CALUDE_youngest_child_age_l199_19988


namespace NUMINAMATH_CALUDE_total_people_l199_19957

/-- Calculates the total number of people in two tribes of soldiers -/
theorem total_people (cannoneers : ℕ) : 
  cannoneers = 63 → 
  (let women := 2 * cannoneers
   let men := cannoneers + 2 * women
   women + men) = 441 := by
sorry

end NUMINAMATH_CALUDE_total_people_l199_19957


namespace NUMINAMATH_CALUDE_opposite_sides_line_range_l199_19979

/-- Given two points on opposite sides of a line, prove the range of the line's constant term -/
theorem opposite_sides_line_range (m : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁ = 3 ∧ y₁ = 1 ∧ x₂ = -4 ∧ y₂ = 6) ∧ 
    ((3 * x₁ - 2 * y₁ + m) * (3 * x₂ - 2 * y₂ + m) < 0)) →
  (-7 < m ∧ m < 24) :=
by sorry

end NUMINAMATH_CALUDE_opposite_sides_line_range_l199_19979


namespace NUMINAMATH_CALUDE_function_inequality_relation_l199_19927

theorem function_inequality_relation (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = 3 * x + 1) →
  a > 0 →
  b > 0 →
  (∀ x, |x - 1| < b → |f x - 4| < a) →
  a ≥ 3 * b :=
sorry

end NUMINAMATH_CALUDE_function_inequality_relation_l199_19927


namespace NUMINAMATH_CALUDE_cone_sphere_ratio_l199_19981

/-- Given a right circular cone and a sphere with the same radius,
    if the volume of the cone is two-fifths that of the sphere,
    then the ratio of the cone's altitude to twice its base radius is 4/5. -/
theorem cone_sphere_ratio (r h : ℝ) (hr : r > 0) : 
  (1 / 3 * π * r^2 * h) = (2 / 5 * (4 / 3 * π * r^3)) → 
  h / (2 * r) = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_cone_sphere_ratio_l199_19981


namespace NUMINAMATH_CALUDE_students_taking_german_prove_students_taking_german_l199_19917

theorem students_taking_german (total : ℕ) (french : ℕ) (both : ℕ) (neither : ℕ) : ℕ :=
  let students_taking_at_least_one := total - neither
  let students_taking_only_french := french - both
  let students_taking_german := students_taking_at_least_one - students_taking_only_french
  students_taking_german

/-- Given a class of 69 students, where 41 are taking French, 9 are taking both French and German,
    and 15 are not taking either course, prove that 22 students are taking German. -/
theorem prove_students_taking_german :
  students_taking_german 69 41 9 15 = 22 := by
  sorry

end NUMINAMATH_CALUDE_students_taking_german_prove_students_taking_german_l199_19917


namespace NUMINAMATH_CALUDE_min_equal_triangles_is_18_l199_19914

/-- A non-convex hexagon representing a chessboard with one corner square cut out. -/
structure CutoutChessboard :=
  (area : ℝ)
  (is_non_convex : Bool)

/-- The minimum number of equal triangles into which the cutout chessboard can be divided. -/
def min_equal_triangles (board : CutoutChessboard) : ℕ :=
  sorry

/-- Theorem stating that the minimum number of equal triangles is 18 for a cutout chessboard with area 63. -/
theorem min_equal_triangles_is_18 (board : CutoutChessboard) 
  (h1 : board.area = 63)
  (h2 : board.is_non_convex = true) : 
  min_equal_triangles board = 18 :=
sorry

end NUMINAMATH_CALUDE_min_equal_triangles_is_18_l199_19914


namespace NUMINAMATH_CALUDE_public_transport_support_percentage_l199_19910

theorem public_transport_support_percentage
  (gov_employees : ℕ) (gov_support_rate : ℚ)
  (citizens : ℕ) (citizen_support_rate : ℚ) :
  gov_employees = 150 →
  gov_support_rate = 70 / 100 →
  citizens = 800 →
  citizen_support_rate = 60 / 100 →
  let total_surveyed := gov_employees + citizens
  let total_supporters := gov_employees * gov_support_rate + citizens * citizen_support_rate
  (total_supporters / total_surveyed : ℚ) = 6158 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_public_transport_support_percentage_l199_19910


namespace NUMINAMATH_CALUDE_max_value_expression_l199_19991

theorem max_value_expression : 
  ∃ (M : ℝ), M = 27 ∧ 
  ∀ (x y : ℝ), 
    (Real.sqrt (36 - 4 * Real.sqrt 5) * Real.sin x - Real.sqrt (2 * (1 + Real.cos (2 * x))) - 2) * 
    (3 + 2 * Real.sqrt (10 - Real.sqrt 5) * Real.cos y - Real.cos (2 * y)) ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l199_19991


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l199_19965

/-- The surface area of a rectangular solid. -/
def surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + length * depth + width * depth)

/-- Theorem: The surface area of a rectangular solid with length 6 meters, width 5 meters, 
    and depth 2 meters is 104 square meters. -/
theorem rectangular_solid_surface_area :
  surface_area 6 5 2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l199_19965


namespace NUMINAMATH_CALUDE_number_proportion_l199_19951

theorem number_proportion (x : ℚ) : 
  (x / 5 = 30 / (10 * 60)) → x = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_number_proportion_l199_19951


namespace NUMINAMATH_CALUDE_centroid_unique_point_l199_19905

/-- Definition of a triangle -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Definition of the centroid of a triangle -/
def centroid (t : Triangle) : Point :=
  sorry

/-- Definition of a point being inside or on the boundary of a triangle -/
def insideOrOnBoundary (p : Point) (t : Triangle) : Prop :=
  sorry

/-- Definition of the area of a triangle -/
def area (t : Triangle) : ℝ :=
  sorry

/-- Theorem stating that the centroid is the unique point satisfying the condition -/
theorem centroid_unique_point (t : Triangle) :
  ∃! M, insideOrOnBoundary M t ∧
    ∀ N, insideOrOnBoundary N t →
      ∃ P, insideOrOnBoundary P t ∧
        area (Triangle.mk M N P) ≥ (1/6 : ℝ) * area t :=
  sorry

end NUMINAMATH_CALUDE_centroid_unique_point_l199_19905


namespace NUMINAMATH_CALUDE_arrangement_count_l199_19970

theorem arrangement_count :
  let total_men : ℕ := 4
  let total_women : ℕ := 5
  let group_of_four_men : ℕ := 2
  let group_of_four_women : ℕ := 2
  let remaining_men : ℕ := total_men - group_of_four_men
  let remaining_women : ℕ := total_women - group_of_four_women
  (Nat.choose total_men group_of_four_men) *
  (Nat.choose total_women group_of_four_women) *
  (Nat.choose remaining_women remaining_men) = 180 :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l199_19970


namespace NUMINAMATH_CALUDE_only_shape3_symmetric_shape3_is_symmetric_other_shapes_not_symmetric_l199_19940

-- Define the type for L-like shapes
inductive LLikeShape
| Shape1
| Shape2
| Shape3
| Shape4
| Shape5

-- Define a function to check if a shape is symmetric to the original
def isSymmetric (shape : LLikeShape) : Prop :=
  match shape with
  | LLikeShape.Shape3 => True
  | _ => False

-- Theorem stating that only Shape3 is symmetric
theorem only_shape3_symmetric :
  ∀ (shape : LLikeShape), isSymmetric shape ↔ shape = LLikeShape.Shape3 :=
by sorry

-- Theorem stating that Shape3 is indeed symmetric
theorem shape3_is_symmetric : isSymmetric LLikeShape.Shape3 :=
by sorry

-- Theorem stating that other shapes are not symmetric
theorem other_shapes_not_symmetric :
  ∀ (shape : LLikeShape), shape ≠ LLikeShape.Shape3 → ¬(isSymmetric shape) :=
by sorry

end NUMINAMATH_CALUDE_only_shape3_symmetric_shape3_is_symmetric_other_shapes_not_symmetric_l199_19940


namespace NUMINAMATH_CALUDE_magicians_marbles_l199_19967

/-- The number of marbles left after the magician's trick --/
def marbles_left (red_initial blue_initial green_initial yellow_initial : ℕ) : ℕ :=
  let red_removed := red_initial / 4
  let blue_removed := 3 * (green_initial / 5)
  let green_removed := (green_initial * 3) / 10  -- 30% rounded down
  let yellow_removed := 25

  let red_left := red_initial - red_removed
  let blue_left := blue_initial - blue_removed
  let green_left := green_initial - green_removed
  let yellow_left := yellow_initial - yellow_removed

  red_left + blue_left + green_left + yellow_left

/-- Theorem stating that given the initial conditions, the number of marbles left is 213 --/
theorem magicians_marbles :
  marbles_left 80 120 75 50 = 213 :=
by sorry

end NUMINAMATH_CALUDE_magicians_marbles_l199_19967


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l199_19953

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between planes
variable (perpendicular_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicular_lines : Line → Line → Prop)

-- Define the intersection of two planes
variable (intersection : Plane → Plane → Line)

-- Define the subset relation for a line in a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (α β : Plane) 
  (h1 : perpendicular_planes α β)
  (h2 : intersection α β = m)
  (h3 : subset n α) :
  perpendicular_line_plane n β ↔ perpendicular_lines n m :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l199_19953


namespace NUMINAMATH_CALUDE_benny_eggs_count_l199_19948

def dozen : ℕ := 12

def eggs_bought (num_dozens : ℕ) : ℕ := num_dozens * dozen

theorem benny_eggs_count : eggs_bought 7 = 84 := by sorry

end NUMINAMATH_CALUDE_benny_eggs_count_l199_19948


namespace NUMINAMATH_CALUDE_sequence_non_positive_l199_19943

theorem sequence_non_positive (n : ℕ) (a : ℕ → ℝ) 
  (h0 : a 0 = 0) 
  (hn : a n = 0) 
  (h : ∀ k : ℕ, k ∈ Finset.range (n - 1) → a k - 2 * a (k + 1) + a (k + 2) ≥ 0) : 
  ∀ k : ℕ, k ≤ n → a k ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_sequence_non_positive_l199_19943


namespace NUMINAMATH_CALUDE_arun_weight_upper_limit_l199_19987

theorem arun_weight_upper_limit (arun_lower : ℝ) (arun_upper : ℝ) 
  (brother_lower : ℝ) (brother_upper : ℝ) (mother_upper : ℝ) (average : ℝ) :
  arun_lower = 61 →
  arun_upper = 72 →
  brother_lower = 60 →
  brother_upper = 70 →
  average = 63 →
  arun_lower < mother_upper →
  mother_upper ≤ brother_upper →
  (arun_lower + mother_upper) / 2 = average →
  mother_upper = 65 := by
sorry

end NUMINAMATH_CALUDE_arun_weight_upper_limit_l199_19987


namespace NUMINAMATH_CALUDE_number_division_l199_19955

theorem number_division (x : ℝ) : x + 8 = 88 → x / 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_division_l199_19955


namespace NUMINAMATH_CALUDE_min_max_sum_l199_19930

theorem min_max_sum (a b c d e : ℕ+) (h_sum : a + b + c + d + e = 2018) :
  673 ≤ max (a + b) (max (b + c) (max (c + d) (d + e))) ∧
  ∃ (a' b' c' d' e' : ℕ+), a' + b' + c' + d' + e' = 2018 ∧
    max (a' + b') (max (b' + c') (max (c' + d') (d' + e'))) = 673 :=
by sorry

end NUMINAMATH_CALUDE_min_max_sum_l199_19930


namespace NUMINAMATH_CALUDE_constant_term_of_product_l199_19937

def is_monic (p : Polynomial ℝ) : Prop :=
  p.leadingCoeff = 1

theorem constant_term_of_product (p q : Polynomial ℝ) :
  is_monic p →
  is_monic q →
  p.degree = 3 →
  q.degree = 3 →
  (∃ c : ℝ, c > 0 ∧ p.coeff 0 = c ∧ q.coeff 0 = c) →
  (∃ a : ℝ, p.coeff 1 = a ∧ q.coeff 1 = a) →
  p * q = Polynomial.monomial 6 1 + Polynomial.monomial 5 2 + Polynomial.monomial 4 1 +
          Polynomial.monomial 3 2 + Polynomial.monomial 2 9 + Polynomial.monomial 1 12 +
          Polynomial.monomial 0 36 →
  p.coeff 0 = 6 ∧ q.coeff 0 = 6 :=
by sorry

end NUMINAMATH_CALUDE_constant_term_of_product_l199_19937


namespace NUMINAMATH_CALUDE_valid_numbers_count_l199_19907

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧
  100 ≤ n^2 ∧ n^2 < 1000 ∧
  100 ≤ (10 * (n % 10) + n / 10)^2 ∧ (10 * (n % 10) + n / 10)^2 < 1000 ∧
  n^2 = (10 * (n % 10) + n / 10)^2 % 10 * 100 + ((10 * (n % 10) + n / 10)^2 / 10 % 10) * 10 + (10 * (n % 10) + n / 10)^2 / 100

theorem valid_numbers_count :
  ∃ (S : Finset ℕ), S.card = 4 ∧ (∀ n, n ∈ S ↔ is_valid_number n) :=
sorry

end NUMINAMATH_CALUDE_valid_numbers_count_l199_19907


namespace NUMINAMATH_CALUDE_quadratic_inequality_nonnegative_l199_19921

theorem quadratic_inequality_nonnegative (x : ℝ) : x^2 - x + 1 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_nonnegative_l199_19921


namespace NUMINAMATH_CALUDE_truck_loading_time_l199_19932

theorem truck_loading_time 
  (worker1_rate : ℝ) 
  (worker2_rate : ℝ) 
  (h1 : worker1_rate = 1 / 6) 
  (h2 : worker2_rate = 1 / 4) : 
  1 / (worker1_rate + worker2_rate) = 12 / 5 := by
sorry

end NUMINAMATH_CALUDE_truck_loading_time_l199_19932


namespace NUMINAMATH_CALUDE_sum_of_fractions_l199_19978

theorem sum_of_fractions : (2 : ℚ) / 5 + 3 / 8 + 1 / 4 = 41 / 40 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_l199_19978


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l199_19998

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (X^3 + 3*X^2 - 4 : Polynomial ℝ) = (X^2 + X - 2 : Polynomial ℝ) * q + 0 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l199_19998


namespace NUMINAMATH_CALUDE_sarah_desserts_l199_19941

theorem sarah_desserts (michael_cookies : ℕ) (sarah_cupcakes : ℕ) :
  michael_cookies = 5 →
  sarah_cupcakes = 9 →
  sarah_cupcakes / 3 = sarah_cupcakes - (sarah_cupcakes / 3) →
  michael_cookies + (sarah_cupcakes - (sarah_cupcakes / 3)) = 11 :=
by sorry

end NUMINAMATH_CALUDE_sarah_desserts_l199_19941


namespace NUMINAMATH_CALUDE_hyperbola_parabola_focus_l199_19949

/-- The value of 'a' for a hyperbola with equation x^2 - y^2 = a^2 (a > 0) 
    whose right focus coincides with the focus of the parabola y^2 = 4x -/
theorem hyperbola_parabola_focus (a : ℝ) : a > 0 → 
  (∃ (x y : ℝ), x^2 - y^2 = a^2 ∧ y^2 = 4*x ∧ (x, y) = (1, 0)) → 
  a = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_focus_l199_19949


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l199_19912

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (m n : Line) (α β : Plane) 
  (h1 : parallel m n) 
  (h2 : plane_perpendicular α β) 
  (h3 : perpendicular m α) : 
  perpendicular n α :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l199_19912


namespace NUMINAMATH_CALUDE_tile_cutting_theorem_l199_19972

/-- Represents a rectangular tile -/
structure Tile where
  width : ℝ
  height : ℝ

/-- Represents the arrangement of tiles -/
structure TileArrangement where
  tiles : List Tile
  width : ℝ
  height : ℝ
  tileCount : ℕ

/-- Represents a part of a cut tile -/
structure TilePart where
  width : ℝ
  height : ℝ

theorem tile_cutting_theorem (arrangement : TileArrangement) 
  (h1 : arrangement.width < arrangement.height)
  (h2 : arrangement.tileCount > 0) :
  ∃ (squareParts rectangleParts : List TilePart),
    (∀ t ∈ arrangement.tiles, ∃ p1 p2, p1 ∈ squareParts ∧ p2 ∈ rectangleParts) ∧
    (∃ s, s > 0 ∧ (∀ p ∈ squareParts, p.width * p.height = s^2 / arrangement.tileCount)) ∧
    (∃ w h, w > 0 ∧ h > 0 ∧ w ≠ h ∧ 
      (∀ p ∈ rectangleParts, p.width * p.height = w * h / arrangement.tileCount)) :=
by sorry

end NUMINAMATH_CALUDE_tile_cutting_theorem_l199_19972


namespace NUMINAMATH_CALUDE_election_outcomes_l199_19924

/-- The number of students participating in the election -/
def total_students : ℕ := 4

/-- The number of students eligible for the entertainment committee member role -/
def eligible_for_entertainment : ℕ := total_students - 1

/-- The number of positions available for each role -/
def positions_per_role : ℕ := 1

/-- Theorem: The number of ways to select a class monitor and an entertainment committee member
    from 4 students, where one specific student cannot be the entertainment committee member,
    is equal to 9. -/
theorem election_outcomes :
  (eligible_for_entertainment.choose positions_per_role) *
  (eligible_for_entertainment.choose positions_per_role) = 9 := by
  sorry

end NUMINAMATH_CALUDE_election_outcomes_l199_19924


namespace NUMINAMATH_CALUDE_chlorine_discount_is_20_percent_l199_19964

def original_chlorine_price : ℝ := 10
def original_soap_price : ℝ := 16
def soap_discount : ℝ := 0.25
def chlorine_quantity : ℕ := 3
def soap_quantity : ℕ := 5
def total_savings : ℝ := 26

theorem chlorine_discount_is_20_percent :
  ∃ (chlorine_discount : ℝ),
    chlorine_discount = 0.20 ∧
    (chlorine_quantity : ℝ) * original_chlorine_price * (1 - chlorine_discount) +
    soap_quantity * original_soap_price * (1 - soap_discount) =
    chlorine_quantity * original_chlorine_price +
    soap_quantity * original_soap_price - total_savings :=
by sorry

end NUMINAMATH_CALUDE_chlorine_discount_is_20_percent_l199_19964


namespace NUMINAMATH_CALUDE_steve_coins_problem_l199_19938

/-- Represents the number of coins of each type -/
structure CoinCount where
  dimes : ℕ
  nickels : ℕ

/-- Represents the value of coins in cents -/
def coinValue (c : CoinCount) : ℕ := c.dimes * 10 + c.nickels * 5

theorem steve_coins_problem :
  ∃ (c : CoinCount),
    c.dimes + c.nickels = 36 ∧
    coinValue c = 310 ∧
    c.dimes = 26 := by
  sorry

end NUMINAMATH_CALUDE_steve_coins_problem_l199_19938


namespace NUMINAMATH_CALUDE_sum_of_divisors_119_l199_19918

/-- The sum of all positive integer divisors of 119 is 144. -/
theorem sum_of_divisors_119 : (Finset.filter (· ∣ 119) (Finset.range 120)).sum id = 144 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_119_l199_19918


namespace NUMINAMATH_CALUDE_intersection_value_l199_19902

/-- Given a proportional function y = kx (k ≠ 0) and an inverse proportional function y = -5/x
    intersecting at points A(x₁, y₁) and B(x₂, y₂), the value of x₁y₂ - 3x₂y₁ is equal to 10. -/
theorem intersection_value (k : ℝ) (x₁ y₁ x₂ y₂ : ℝ) : 
  k ≠ 0 →
  y₁ = k * x₁ →
  y₁ = -5 / x₁ →
  y₂ = k * x₂ →
  y₂ = -5 / x₂ →
  x₁ * y₂ - 3 * x₂ * y₁ = 10 := by
  sorry

end NUMINAMATH_CALUDE_intersection_value_l199_19902


namespace NUMINAMATH_CALUDE_area_between_curves_l199_19908

-- Define the functions for the curves
def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := x^3

-- Define the bounds of integration
def lower_bound : ℝ := 0
def upper_bound : ℝ := 1

-- State the theorem
theorem area_between_curves : 
  (∫ x in lower_bound..upper_bound, f x - g x) = 1/12 := by sorry

end NUMINAMATH_CALUDE_area_between_curves_l199_19908


namespace NUMINAMATH_CALUDE_angle_equality_l199_19993

-- Define what it means for two angles to be vertical
def are_vertical_angles (A B : ℝ) : Prop := sorry

-- State the theorem that vertical angles are equal
axiom vertical_angles_are_equal : ∀ A B : ℝ, are_vertical_angles A B → A = B

-- The statement to be proved
theorem angle_equality (A B : ℝ) (h : are_vertical_angles A B) : A = B := by
  sorry

end NUMINAMATH_CALUDE_angle_equality_l199_19993
