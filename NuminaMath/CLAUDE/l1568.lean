import Mathlib

namespace NUMINAMATH_CALUDE_random_function_iff_stochastic_process_l1568_156858

open MeasureTheory ProbabilityTheory

/-- A random function X = (X_t)_{t ∈ T} taking values in (ℝ^T, ℬ(ℝ^T)) -/
def RandomFunction (T : Type) (Ω : Type) [MeasurableSpace Ω] : Type :=
  Ω → (T → ℝ)

/-- A stochastic process (collection of random variables X_t) -/
def StochasticProcess (T : Type) (Ω : Type) [MeasurableSpace Ω] : Type :=
  T → (Ω → ℝ)

/-- Theorem stating the equivalence between random functions and stochastic processes -/
theorem random_function_iff_stochastic_process (T : Type) (Ω : Type) [MeasurableSpace Ω] :
  (∃ X : RandomFunction T Ω, Measurable X) ↔ (∃ Y : StochasticProcess T Ω, ∀ t, Measurable (Y t)) :=
sorry


end NUMINAMATH_CALUDE_random_function_iff_stochastic_process_l1568_156858


namespace NUMINAMATH_CALUDE_fraction_value_l1568_156860

theorem fraction_value (a b c : ℚ) (h1 : a = 5) (h2 : b = -3) (h3 : c = 2) :
  3 * c / (a + b) = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1568_156860


namespace NUMINAMATH_CALUDE_ball_probability_theorem_l1568_156811

/-- Given a bag with m red balls and n white balls, where m ≥ n ≥ 2, prove that if the probability
    of drawing two red balls is an integer multiple of the probability of drawing one red and one
    white ball, then m must be odd. Also, find all pairs (m, n) such that m + n ≤ 40 and the
    probability of drawing two balls of the same color equals the probability of drawing two balls
    of different colors. -/
theorem ball_probability_theorem (m n : ℕ) (h1 : m ≥ n) (h2 : n ≥ 2) :
  (∃ k : ℕ, Nat.choose m 2 * (Nat.choose (m + n) 2) = k * m * n * (Nat.choose (m + n) 2)) →
  Odd m ∧
  (m + n ≤ 40 →
    Nat.choose m 2 + Nat.choose n 2 = m * n →
    ∃ (p q : ℕ), p = m ∧ q = n) :=
by sorry

end NUMINAMATH_CALUDE_ball_probability_theorem_l1568_156811


namespace NUMINAMATH_CALUDE_sin_pi_half_plus_alpha_l1568_156823

/-- Given a point P(-4, 3) on the terminal side of angle α, prove that sin(π/2 + α) = -4/5 -/
theorem sin_pi_half_plus_alpha (α : ℝ) : 
  let P : ℝ × ℝ := (-4, 3)
  Real.sin (π / 2 + α) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_pi_half_plus_alpha_l1568_156823


namespace NUMINAMATH_CALUDE_janet_dress_pockets_janet_dress_problem_l1568_156800

theorem janet_dress_pockets (total_dresses : ℕ) (dresses_with_pockets : ℕ) 
  (dresses_unknown_pockets : ℕ) (known_pockets : ℕ) (total_pockets : ℕ) : ℕ :=
  let dresses_known_pockets := dresses_with_pockets - dresses_unknown_pockets
  let unknown_pockets := (total_pockets - dresses_known_pockets * known_pockets) / dresses_unknown_pockets
  unknown_pockets

theorem janet_dress_problem : janet_dress_pockets 24 12 4 3 32 = 2 := by
  sorry

end NUMINAMATH_CALUDE_janet_dress_pockets_janet_dress_problem_l1568_156800


namespace NUMINAMATH_CALUDE_buy_one_get_one_free_cost_l1568_156812

/-- Calculates the total cost of cans under a "buy 1 get one free" offer -/
def totalCost (totalCans : ℕ) (pricePerCan : ℚ) : ℚ :=
  (totalCans / 2 : ℚ) * pricePerCan

/-- Proves that the total cost for 30 cans at $0.60 each under a "buy 1 get one free" offer is $9 -/
theorem buy_one_get_one_free_cost :
  totalCost 30 (60 / 100) = 9 := by
  sorry

end NUMINAMATH_CALUDE_buy_one_get_one_free_cost_l1568_156812


namespace NUMINAMATH_CALUDE_pies_difference_l1568_156896

/-- The number of pies sold by Smith's Bakery -/
def smiths_pies : ℕ := 70

/-- The number of pies sold by Mcgee's Bakery -/
def mcgees_pies : ℕ := 16

/-- Theorem stating the difference between Smith's pies and four times Mcgee's pies -/
theorem pies_difference : smiths_pies - 4 * mcgees_pies = 6 := by
  sorry

end NUMINAMATH_CALUDE_pies_difference_l1568_156896


namespace NUMINAMATH_CALUDE_skateboard_speed_l1568_156848

/-- 
Given Pedro's skateboarding speed and time, prove Liam's required speed 
to cover the same distance in a different time.
-/
theorem skateboard_speed 
  (pedro_speed : ℝ) 
  (pedro_time : ℝ) 
  (liam_time : ℝ) 
  (h1 : pedro_speed = 10) 
  (h2 : pedro_time = 4) 
  (h3 : liam_time = 5) : 
  (pedro_speed * pedro_time) / liam_time = 8 := by
  sorry

#check skateboard_speed

end NUMINAMATH_CALUDE_skateboard_speed_l1568_156848


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l1568_156875

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water (current_speed : ℝ) (downstream_distance : ℝ) (travel_time_minutes : ℝ) :
  current_speed = 7 →
  downstream_distance = 35.93 →
  travel_time_minutes = 44 →
  ∃ (v : ℝ), abs (v - 42) < 0.01 ∧ downstream_distance = (v + current_speed) * (travel_time_minutes / 60) :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l1568_156875


namespace NUMINAMATH_CALUDE_shelbys_drive_l1568_156888

/-- Shelby's driving problem -/
theorem shelbys_drive (sunny_speed rainy_speed foggy_speed : ℚ)
  (total_distance total_time : ℚ) (sunny_time rainy_time foggy_time : ℚ) :
  sunny_speed = 35 →
  rainy_speed = 25 →
  foggy_speed = 15 →
  total_distance = 20 →
  total_time = 60 →
  sunny_time + rainy_time + foggy_time = total_time →
  sunny_speed * sunny_time / 60 + rainy_speed * rainy_time / 60 + foggy_speed * foggy_time / 60 = total_distance →
  foggy_time = 45 := by
  sorry

#check shelbys_drive

end NUMINAMATH_CALUDE_shelbys_drive_l1568_156888


namespace NUMINAMATH_CALUDE_point_in_third_quadrant_m_range_l1568_156883

theorem point_in_third_quadrant_m_range (m : ℝ) : 
  (m - 4 < 0 ∧ 1 - 2*m < 0) → (1/2 < m ∧ m < 4) :=
by sorry

end NUMINAMATH_CALUDE_point_in_third_quadrant_m_range_l1568_156883


namespace NUMINAMATH_CALUDE_optimal_strategy_l1568_156846

-- Define the set of available numbers
def availableNumbers : Finset Nat := Finset.range 17

-- Define the rules of the game
def isValidChoice (chosen : Finset Nat) (n : Nat) : Bool :=
  n ∈ availableNumbers ∧
  n ∉ chosen ∧
  ¬(∃m ∈ chosen, n = 2 * m ∨ 2 * n = m)

-- Define the state after Player A's move
def initialState : Finset Nat := {8}

-- Define Player B's optimal move
def optimalMove : Nat := 6

-- Theorem to prove
theorem optimal_strategy :
  isValidChoice initialState optimalMove ∧
  ∀ n : Nat, n ≠ optimalMove → 
    (isValidChoice initialState n → 
      ∃ m : Nat, isValidChoice (insert n initialState) m) →
    ¬(∃ m : Nat, isValidChoice (insert optimalMove initialState) m) :=
sorry

end NUMINAMATH_CALUDE_optimal_strategy_l1568_156846


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1568_156828

-- Define the universal set U
def U : Set ℤ := {x | 0 < x ∧ x < 5}

-- Define sets A and B
def A : Set ℤ := {1, 2}
def B : Set ℤ := {2, 3}

-- State the theorem
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {3} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1568_156828


namespace NUMINAMATH_CALUDE_cat_stickers_count_l1568_156874

theorem cat_stickers_count (space_stickers : Nat) (friends : Nat) (leftover : Nat) (cat_stickers : Nat) : 
  space_stickers = 100 →
  friends = 3 →
  leftover = 3 →
  (space_stickers + cat_stickers - leftover) % friends = 0 →
  cat_stickers = 2 := by
  sorry

end NUMINAMATH_CALUDE_cat_stickers_count_l1568_156874


namespace NUMINAMATH_CALUDE_fibonacci_fourth_term_divisible_by_three_l1568_156816

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_fourth_term_divisible_by_three (k : ℕ) :
  3 ∣ fibonacci (4 * k) := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_fourth_term_divisible_by_three_l1568_156816


namespace NUMINAMATH_CALUDE_alternating_dodecagon_area_l1568_156862

/-- An equilateral 12-gon with alternating interior angles -/
structure AlternatingDodecagon where
  side_length : ℝ
  interior_angles : Fin 12 → ℝ
  is_equilateral : ∀ i : Fin 12, side_length > 0
  angle_pattern : ∀ i : Fin 12, interior_angles i = 
    if i % 3 = 0 ∨ i % 3 = 1 then 90 else 270

/-- The area of the alternating dodecagon -/
noncomputable def area (d : AlternatingDodecagon) : ℝ := sorry

/-- Theorem stating that the area of the specific alternating dodecagon is 500 -/
theorem alternating_dodecagon_area :
  ∀ d : AlternatingDodecagon, d.side_length = 10 → area d = 500 := by sorry

end NUMINAMATH_CALUDE_alternating_dodecagon_area_l1568_156862


namespace NUMINAMATH_CALUDE_min_value_sum_squares_l1568_156817

theorem min_value_sum_squares (x y z : ℝ) (h : 2*x + 2*y + z + 8 = 0) :
  ∃ (m : ℝ), m = 9 ∧ ∀ (x' y' z' : ℝ), 2*x' + 2*y' + z' + 8 = 0 →
    (x' - 1)^2 + (y' + 2)^2 + (z' - 3)^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_squares_l1568_156817


namespace NUMINAMATH_CALUDE_convergence_and_bound_l1568_156852

def u : ℕ → ℚ
  | 0 => 1/6
  | n + 1 => 2 * u n - 2 * (u n)^2 + 1/3

def L : ℚ := 5/6

theorem convergence_and_bound :
  (∃ (k : ℕ), ∀ (n : ℕ), n ≥ k → |u n - L| ≤ 1 / 2^500) ∧
  (∀ (k : ℕ), k < 9 → ∃ (n : ℕ), n ≥ k ∧ |u n - L| > 1 / 2^500) ∧
  (∀ (n : ℕ), n ≥ 9 → |u n - L| ≤ 1 / 2^500) :=
sorry

end NUMINAMATH_CALUDE_convergence_and_bound_l1568_156852


namespace NUMINAMATH_CALUDE_delta_value_l1568_156832

theorem delta_value (Δ : ℤ) : 4 * (-3) = Δ + 5 → Δ = -17 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l1568_156832


namespace NUMINAMATH_CALUDE_f_properties_l1568_156859

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + Real.log x

theorem f_properties (a : ℝ) :
  (∃ x ∈ Set.Icc 1 2, f a x ≥ 0 → a ≤ 2 + 1/2 * Real.log 2) ∧
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Ioi 1 ∧ 
    (∀ x : ℝ, deriv (f a) x = 0 ↔ x = x₁ ∨ x = x₂) →
    f a x₁ - f a x₂ < -3/4 + Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l1568_156859


namespace NUMINAMATH_CALUDE_books_sum_is_67_l1568_156836

/-- The total number of books Sandy, Benny, and Tim have together -/
def total_books (sandy_books benny_books tim_books : ℕ) : ℕ :=
  sandy_books + benny_books + tim_books

/-- Theorem stating that the total number of books is 67 -/
theorem books_sum_is_67 :
  total_books 10 24 33 = 67 := by
  sorry

end NUMINAMATH_CALUDE_books_sum_is_67_l1568_156836


namespace NUMINAMATH_CALUDE_alpha_range_l1568_156801

theorem alpha_range (α : Real) 
  (h1 : 0 ≤ α ∧ α < 2 * Real.pi) 
  (h2 : Real.sin α > Real.sqrt 3 * Real.cos α) : 
  Real.pi / 3 < α ∧ α < 4 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_alpha_range_l1568_156801


namespace NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l1568_156897

theorem cos_four_arccos_two_fifths :
  Real.cos (4 * Real.arccos (2/5)) = -47/625 := by
  sorry

end NUMINAMATH_CALUDE_cos_four_arccos_two_fifths_l1568_156897


namespace NUMINAMATH_CALUDE_abs_sqrt_mul_eq_three_l1568_156882

theorem abs_sqrt_mul_eq_three : |(-3 : ℤ)| + Real.sqrt 4 + (-2 : ℤ) * (1 : ℤ) = 3 := by
  sorry

end NUMINAMATH_CALUDE_abs_sqrt_mul_eq_three_l1568_156882


namespace NUMINAMATH_CALUDE_share_difference_l1568_156818

/-- Given a distribution ratio and Vasim's share, calculate the difference between Ranjith's and Faruk's shares -/
theorem share_difference (faruk_ratio vasim_ratio ranjith_ratio : ℕ) (vasim_share : ℕ) : 
  faruk_ratio = 3 →
  vasim_ratio = 5 →
  ranjith_ratio = 6 →
  vasim_share = 1500 →
  (ranjith_ratio * vasim_share / vasim_ratio) - (faruk_ratio * vasim_share / vasim_ratio) = 900 := by
sorry

end NUMINAMATH_CALUDE_share_difference_l1568_156818


namespace NUMINAMATH_CALUDE_manicure_cost_proof_l1568_156857

/-- The cost of a hair updo in dollars -/
def hair_updo_cost : ℝ := 50

/-- The total cost including tips for both services in dollars -/
def total_cost_with_tips : ℝ := 96

/-- The tip percentage as a decimal -/
def tip_percentage : ℝ := 0.20

/-- The cost of a manicure in dollars -/
def manicure_cost : ℝ := 30

theorem manicure_cost_proof :
  (hair_updo_cost + manicure_cost) * (1 + tip_percentage) = total_cost_with_tips := by
  sorry

end NUMINAMATH_CALUDE_manicure_cost_proof_l1568_156857


namespace NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1568_156827

/-- Given a sphere with surface area 400π cm², prove its volume is (4000/3)π cm³ -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
  (4 * π * r^2 = 400 * π) →  -- Surface area formula
  ((4 / 3) * π * r^3 = (4000 / 3) * π) -- Volume formula
  := by sorry

end NUMINAMATH_CALUDE_sphere_volume_from_surface_area_l1568_156827


namespace NUMINAMATH_CALUDE_sandy_paint_area_l1568_156868

/-- The area Sandy needs to paint on a wall with a decorative region -/
theorem sandy_paint_area (wall_height wall_length decor_height decor_length : ℝ)
  (h_wall_height : wall_height = 10)
  (h_wall_length : wall_length = 15)
  (h_decor_height : decor_height = 3)
  (h_decor_length : decor_length = 5) :
  wall_height * wall_length - decor_height * decor_length = 135 := by
  sorry

end NUMINAMATH_CALUDE_sandy_paint_area_l1568_156868


namespace NUMINAMATH_CALUDE_white_bread_cost_l1568_156851

/-- Represents the cost of bread items in dollars -/
structure BreadCosts where
  white : ℝ
  baguette : ℝ := 1.50
  sourdough : ℝ := 4.50
  croissant : ℝ := 2.00

/-- Represents the weekly purchase of bread items -/
structure WeeklyPurchase where
  white : ℕ := 2
  baguette : ℕ := 1
  sourdough : ℕ := 2
  croissant : ℕ := 1

def total_spent_over_4_weeks : ℝ := 78

/-- Calculates the weekly cost of non-white bread items -/
def weekly_non_white_cost (costs : BreadCosts) (purchase : WeeklyPurchase) : ℝ :=
  costs.baguette * purchase.baguette + 
  costs.sourdough * purchase.sourdough + 
  costs.croissant * purchase.croissant

/-- Theorem stating that the cost of each loaf of white bread is $3.50 -/
theorem white_bread_cost (costs : BreadCosts) (purchase : WeeklyPurchase) :
  costs.white = 3.50 ↔ 
  total_spent_over_4_weeks = 
    4 * (weekly_non_white_cost costs purchase + costs.white * purchase.white) :=
sorry

end NUMINAMATH_CALUDE_white_bread_cost_l1568_156851


namespace NUMINAMATH_CALUDE_secretary_discussions_l1568_156810

/-- Represents the number of emails sent in a small discussion -/
def small_discussion_emails : ℕ := 7 * 6

/-- Represents the number of emails sent in a large discussion -/
def large_discussion_emails : ℕ := 15 * 14

/-- Represents the total number of emails sent excluding the secretary's -/
def total_emails : ℕ := 1994

/-- Represents the maximum number of discussions a jury member can participate in -/
def max_discussions : ℕ := 10

theorem secretary_discussions (m b : ℕ) :
  m + b ≤ max_discussions →
  small_discussion_emails * m + large_discussion_emails * b + 6 * m + 14 * b = total_emails →
  m = 6 ∧ b = 2 := by
  sorry

#check secretary_discussions

end NUMINAMATH_CALUDE_secretary_discussions_l1568_156810


namespace NUMINAMATH_CALUDE_square_roots_ratio_l1568_156869

-- Define the complex polynomial z^2 + az + b
def complex_polynomial (a b z : ℂ) : ℂ := z^2 + a*z + b

-- Define the theorem
theorem square_roots_ratio (a b z₁ : ℂ) :
  (complex_polynomial a b z₁ = 0) →
  (complex_polynomial a b (Complex.I * z₁) = 0) →
  a^2 / b = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_ratio_l1568_156869


namespace NUMINAMATH_CALUDE_oil_storage_solution_l1568_156895

/-- Represents the oil storage problem with given constraints --/
def oil_storage_problem (total_oil : ℕ) (large_barrel_capacity : ℕ) (small_barrels_used : ℕ) : Prop :=
  ∃ (small_barrel_capacity : ℕ) (large_barrels_used : ℕ),
    total_oil = large_barrels_used * large_barrel_capacity + small_barrels_used * small_barrel_capacity ∧
    small_barrels_used > 0 ∧
    small_barrel_capacity > 0 ∧
    small_barrel_capacity < large_barrel_capacity ∧
    ∀ (other_large : ℕ) (other_small : ℕ),
      total_oil = other_large * large_barrel_capacity + other_small * small_barrel_capacity →
      other_small ≥ small_barrels_used →
      other_large + other_small ≥ large_barrels_used + small_barrels_used

/-- The solution to the oil storage problem --/
theorem oil_storage_solution :
  oil_storage_problem 95 6 1 →
  ∃ (small_barrel_capacity : ℕ), small_barrel_capacity = 5 := by
  sorry

end NUMINAMATH_CALUDE_oil_storage_solution_l1568_156895


namespace NUMINAMATH_CALUDE_airplane_seats_l1568_156861

/-- Given an airplane with a total of 387 seats, where the number of coach class seats
    is 2 more than 4 times the number of first-class seats, prove that there are
    77 first-class seats. -/
theorem airplane_seats (total_seats : ℕ) (first_class : ℕ) (coach : ℕ)
    (h1 : total_seats = 387)
    (h2 : coach = 4 * first_class + 2)
    (h3 : total_seats = first_class + coach) :
    first_class = 77 := by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_l1568_156861


namespace NUMINAMATH_CALUDE_vivian_yogurt_count_l1568_156841

/-- The number of banana slices per yogurt -/
def slices_per_yogurt : ℕ := 8

/-- The number of slices one banana yields -/
def slices_per_banana : ℕ := 10

/-- The number of bananas Vivian needs to buy -/
def bananas_to_buy : ℕ := 4

/-- The number of yogurts Vivian needs to make -/
def yogurts_to_make : ℕ := (bananas_to_buy * slices_per_banana) / slices_per_yogurt

theorem vivian_yogurt_count : yogurts_to_make = 5 := by
  sorry

end NUMINAMATH_CALUDE_vivian_yogurt_count_l1568_156841


namespace NUMINAMATH_CALUDE_cell_population_after_10_days_l1568_156887

/-- The number of cells in a colony after a given number of days, 
    where the initial population is 5 cells and the population triples every 3 days. -/
def cell_population (days : ℕ) : ℕ :=
  5 * 3^(days / 3)

/-- Theorem stating that the cell population after 10 days is 135 cells. -/
theorem cell_population_after_10_days : cell_population 10 = 135 := by
  sorry

end NUMINAMATH_CALUDE_cell_population_after_10_days_l1568_156887


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1568_156892

-- Define the sets A and B
def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (1 - x^2)}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = Set.Icc 0 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1568_156892


namespace NUMINAMATH_CALUDE_triangle_side_length_l1568_156871

/-- Given a triangle ABC with circumradius R, prove that if cos B and cos A are known,
    then the length of side c can be determined. -/
theorem triangle_side_length (A B C : Real) (R : Real) (h1 : R = 5/6)
  (h2 : Real.cos B = 3/5) (h3 : Real.cos A = 12/13) :
  2 * R * Real.sin (A + B) = 21/13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l1568_156871


namespace NUMINAMATH_CALUDE_food_shelf_life_l1568_156876

/-- The shelf life function for a food product -/
noncomputable def shelf_life (k b : ℝ) (x : ℝ) : ℝ := Real.exp (k * x + b)

/-- Theorem stating the shelf life at 30°C and the maximum temperature for 80 hours shelf life -/
theorem food_shelf_life (k b : ℝ) :
  (shelf_life k b 0 = 160) →
  (shelf_life k b 20 = 40) →
  (shelf_life k b 30 = 20) ∧
  (∀ x : ℝ, shelf_life k b x ≥ 80 ↔ x ≤ 10) := by
  sorry


end NUMINAMATH_CALUDE_food_shelf_life_l1568_156876


namespace NUMINAMATH_CALUDE_triangle_inequality_from_squared_sum_l1568_156806

theorem triangle_inequality_from_squared_sum (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_ineq : (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4)) :
  a + b > c ∧ b + c > a ∧ c + a > b := by sorry

end NUMINAMATH_CALUDE_triangle_inequality_from_squared_sum_l1568_156806


namespace NUMINAMATH_CALUDE_chord_length_is_2_sqrt_2_l1568_156854

-- Define the line
def line (x y : ℝ) : Prop := x + y = 3

-- Define the curve
def curve (x y : ℝ) : Prop := x^2 + y^2 - 2*y - 3 = 0

-- Define the chord length
def chord_length (l : (ℝ → ℝ → Prop)) (c : (ℝ → ℝ → Prop)) : ℝ := 
  sorry  -- The actual computation of chord length would go here

-- Theorem statement
theorem chord_length_is_2_sqrt_2 :
  chord_length line curve = 2 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_chord_length_is_2_sqrt_2_l1568_156854


namespace NUMINAMATH_CALUDE_binary_addition_proof_l1568_156855

def binary_to_nat : List Bool → Nat
  | [] => 0
  | b::bs => (if b then 1 else 0) + 2 * binary_to_nat bs

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then
    []
  else
    (n % 2 = 1) :: nat_to_binary (n / 2)

theorem binary_addition_proof :
  let a := [false, true, false, true]  -- 1010₂
  let b := [false, true]               -- 10₂
  let sum := [false, false, true, true] -- 1100₂
  binary_to_nat a + binary_to_nat b = binary_to_nat sum := by
  sorry

end NUMINAMATH_CALUDE_binary_addition_proof_l1568_156855


namespace NUMINAMATH_CALUDE_initial_profit_percentage_l1568_156813

/-- Proves that the initial profit percentage is 5% given the conditions of the problem -/
theorem initial_profit_percentage (cost_price selling_price : ℝ) : 
  cost_price = 1000 →
  (0.95 * cost_price) * 1.1 = selling_price - 5 →
  (selling_price - cost_price) / cost_price = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_initial_profit_percentage_l1568_156813


namespace NUMINAMATH_CALUDE_convex_pentagon_angles_obtuse_l1568_156834

/-- A convex pentagon with equal sides and each angle less than 120° -/
structure ConvexPentagon where
  -- The pentagon is convex
  is_convex : Bool
  -- All sides are equal
  equal_sides : Bool
  -- Each angle is less than 120°
  angles_less_than_120 : Bool

/-- Theorem: In a convex pentagon with equal sides and each angle less than 120°, 
    each angle is greater than 90° -/
theorem convex_pentagon_angles_obtuse (p : ConvexPentagon) : 
  p.is_convex ∧ p.equal_sides ∧ p.angles_less_than_120 → 
  ∀ angle, angle > 90 := by sorry

end NUMINAMATH_CALUDE_convex_pentagon_angles_obtuse_l1568_156834


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1568_156856

/-- Proves that the solution set of the given inequality system is (-2, 1]. -/
theorem inequality_system_solution :
  ∀ x : ℝ, (x > -6 - 2*x ∧ x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1568_156856


namespace NUMINAMATH_CALUDE_unique_intersection_point_l1568_156889

-- Define the three lines
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 4 * y = 1
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 8

-- Define a point that lies on at least two of the lines
def intersection_point (p : ℝ × ℝ) : Prop :=
  (line1 p.1 p.2 ∧ line2 p.1 p.2) ∨
  (line1 p.1 p.2 ∧ line3 p.1 p.2) ∨
  (line2 p.1 p.2 ∧ line3 p.1 p.2)

-- Theorem stating that there is exactly one intersection point
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, intersection_point p :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_point_l1568_156889


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l1568_156845

theorem triangle_angle_proof (a b c A B C : Real) : 
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 
  0 < B ∧ B < π ∧ 
  0 < C ∧ C < π ∧ 
  A + B + C = π ∧
  b = a * (Real.sin C + Real.cos C) →
  A = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l1568_156845


namespace NUMINAMATH_CALUDE_craft_sales_sum_l1568_156830

/-- The sum of an arithmetic sequence with first term 3 and common difference 4 for 10 terms -/
theorem craft_sales_sum : 
  let a : ℕ → ℕ := fun n => 3 + 4 * (n - 1)
  let S : ℕ → ℕ := fun n => n * (a 1 + a n) / 2
  S 10 = 210 := by
sorry

end NUMINAMATH_CALUDE_craft_sales_sum_l1568_156830


namespace NUMINAMATH_CALUDE_distance_from_origin_and_point_specific_distances_l1568_156867

theorem distance_from_origin_and_point (d : ℝ) (p : ℝ) :
  -- A point at distance d from the origin represents either d or -d
  (∃ x : ℝ, x = d ∨ x = -d ∧ |x| = d) ∧
  -- A point at distance d from p represents either p + d or p - d
  (∃ y : ℝ, y = p + d ∨ y = p - d ∧ |y - p| = d) :=
by
  sorry

-- Specific instances for the given problem
theorem specific_distances :
  -- A point at distance √5 from the origin represents either √5 or -√5
  (∃ x : ℝ, x = Real.sqrt 5 ∨ x = -Real.sqrt 5 ∧ |x| = Real.sqrt 5) ∧
  -- A point at distance 2√5 from √5 represents either 3√5 or -√5
  (∃ y : ℝ, y = 3 * Real.sqrt 5 ∨ y = -Real.sqrt 5 ∧ |y - Real.sqrt 5| = 2 * Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_and_point_specific_distances_l1568_156867


namespace NUMINAMATH_CALUDE_peters_horses_l1568_156878

/-- The number of horses Peter has -/
def num_horses : ℕ := 4

/-- The amount of oats each horse eats per feeding -/
def oats_per_feeding : ℕ := 4

/-- The number of oat feedings per day -/
def oat_feedings_per_day : ℕ := 2

/-- The amount of grain each horse eats per day -/
def grain_per_day : ℕ := 3

/-- The number of days Peter feeds his horses -/
def feeding_days : ℕ := 3

/-- The total amount of food Peter needs for all his horses for the given days -/
def total_food : ℕ := 132

theorem peters_horses :
  num_horses * (oats_per_feeding * oat_feedings_per_day + grain_per_day) * feeding_days = total_food :=
by sorry

end NUMINAMATH_CALUDE_peters_horses_l1568_156878


namespace NUMINAMATH_CALUDE_beavers_swimming_l1568_156898

theorem beavers_swimming (initial_beavers final_beavers : ℕ) : 
  initial_beavers ≥ final_beavers → 
  initial_beavers - final_beavers = initial_beavers - final_beavers :=
by
  sorry

#check beavers_swimming 2 1

end NUMINAMATH_CALUDE_beavers_swimming_l1568_156898


namespace NUMINAMATH_CALUDE_apollo_chariot_payment_l1568_156843

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of months before the price increase -/
def months_before_increase : ℕ := 6

/-- The initial price in golden apples per month -/
def initial_price : ℕ := 3

/-- The price increase factor -/
def price_increase_factor : ℕ := 2

/-- The total number of golden apples paid for the year -/
def total_apples : ℕ := 
  initial_price * months_before_increase + 
  initial_price * price_increase_factor * (months_in_year - months_before_increase)

theorem apollo_chariot_payment :
  total_apples = 54 := by sorry

end NUMINAMATH_CALUDE_apollo_chariot_payment_l1568_156843


namespace NUMINAMATH_CALUDE_total_pet_owners_l1568_156807

/-- The number of people who own only dogs -/
def only_dogs : ℕ := 15

/-- The number of people who own only cats -/
def only_cats : ℕ := 10

/-- The number of people who own only cats and dogs -/
def cats_and_dogs : ℕ := 5

/-- The number of people who own cats, dogs, and snakes -/
def cats_dogs_snakes : ℕ := 3

/-- The total number of snakes -/
def total_snakes : ℕ := 29

/-- Theorem stating the total number of pet owners -/
theorem total_pet_owners : 
  only_dogs + only_cats + cats_and_dogs + cats_dogs_snakes = 33 := by
  sorry


end NUMINAMATH_CALUDE_total_pet_owners_l1568_156807


namespace NUMINAMATH_CALUDE_third_stick_shorter_by_one_cm_l1568_156833

/-- The length difference between the second and third stick -/
def stick_length_difference (first_stick second_stick third_stick : ℝ) : ℝ :=
  second_stick - third_stick

/-- Proof that the third stick is 1 cm shorter than the second stick -/
theorem third_stick_shorter_by_one_cm 
  (first_stick : ℝ)
  (second_stick : ℝ)
  (third_stick : ℝ)
  (h1 : first_stick = 3)
  (h2 : second_stick = 2 * first_stick)
  (h3 : first_stick + second_stick + third_stick = 14) :
  stick_length_difference first_stick second_stick third_stick = 1 := by
sorry

end NUMINAMATH_CALUDE_third_stick_shorter_by_one_cm_l1568_156833


namespace NUMINAMATH_CALUDE_special_polynomial_exists_l1568_156808

/-- A fifth-degree polynomial with specific root properties -/
def exists_special_polynomial : Prop :=
  ∃ (P : ℝ → ℝ),
    (∀ x : ℝ, ∃ (a b c d e f : ℝ), P x = a*x^5 + b*x^4 + c*x^3 + d*x^2 + e*x + f) ∧
    (∀ r : ℝ, P r = 0 → r < 0) ∧
    (∀ s : ℝ, (deriv P) s = 0 → s > 0) ∧
    (∃ t : ℝ, P t = 0) ∧
    (∃ u : ℝ, (deriv P) u = 0)

/-- Theorem stating the existence of a special polynomial -/
theorem special_polynomial_exists : exists_special_polynomial :=
sorry

end NUMINAMATH_CALUDE_special_polynomial_exists_l1568_156808


namespace NUMINAMATH_CALUDE_intersection_M_P_union_M_P_condition_l1568_156831

-- Define the sets M and P
def M (m : ℝ) : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 4*m - 2}
def P : Set ℝ := {x : ℝ | x > 2 ∨ x ≤ 1}

-- Theorem 1: Intersection of M and P when m = 2
theorem intersection_M_P : 
  M 2 ∩ P = {x : ℝ | (-1 ≤ x ∧ x ≤ 1) ∨ (2 < x ∧ x ≤ 6)} := by sorry

-- Theorem 2: Union of M and P is ℝ iff m ≥ 1
theorem union_M_P_condition (m : ℝ) : 
  M m ∪ P = Set.univ ↔ m ≥ 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_P_union_M_P_condition_l1568_156831


namespace NUMINAMATH_CALUDE_charcoal_drawings_l1568_156879

theorem charcoal_drawings (total : Nat) (colored_pencil : Nat) (blending_marker : Nat) :
  total = 60 → colored_pencil = 24 → blending_marker = 19 →
  total - colored_pencil - blending_marker = 17 := by
  sorry

end NUMINAMATH_CALUDE_charcoal_drawings_l1568_156879


namespace NUMINAMATH_CALUDE_plane_perpendicular_parallel_implies_perpendicular_l1568_156805

-- Define the plane type
structure Plane where
  -- Add necessary fields or leave it abstract

-- Define the perpendicular and parallel relations
def perpendicular (p q : Plane) : Prop := sorry

def parallel (p q : Plane) : Prop := sorry

-- State the theorem
theorem plane_perpendicular_parallel_implies_perpendicular 
  (α β γ : Plane) 
  (h1 : α ≠ β) (h2 : β ≠ γ) (h3 : α ≠ γ)
  (h4 : perpendicular α β) 
  (h5 : parallel β γ) : 
  perpendicular α γ := by sorry

end NUMINAMATH_CALUDE_plane_perpendicular_parallel_implies_perpendicular_l1568_156805


namespace NUMINAMATH_CALUDE_jims_remaining_distance_l1568_156802

/-- Calculates the remaining distance in a journey. -/
def remaining_distance (total : ℕ) (driven : ℕ) : ℕ :=
  total - driven

/-- Proves that for Jim's journey, the remaining distance is 1,068 miles. -/
theorem jims_remaining_distance :
  remaining_distance 2450 1382 = 1068 := by
  sorry

end NUMINAMATH_CALUDE_jims_remaining_distance_l1568_156802


namespace NUMINAMATH_CALUDE_monomial_division_equality_l1568_156891

theorem monomial_division_equality (x y : ℝ) (m n : ℤ) :
  (x^m * y^n) / ((1/4) * x^3 * y) = 4 * x^2 ↔ m = 5 ∧ n = 1 := by
  sorry

end NUMINAMATH_CALUDE_monomial_division_equality_l1568_156891


namespace NUMINAMATH_CALUDE_eduardo_flour_amount_l1568_156849

/-- Represents the number of cookies in the original recipe -/
def original_cookies : ℕ := 30

/-- Represents the amount of flour (in cups) needed for the original recipe -/
def original_flour : ℕ := 2

/-- Represents the number of cookies Eduardo wants to bake -/
def eduardo_cookies : ℕ := 90

/-- Calculates the amount of flour needed for a given number of cookies -/
def flour_needed (cookies : ℕ) : ℕ :=
  (cookies * original_flour) / original_cookies

theorem eduardo_flour_amount : flour_needed eduardo_cookies = 6 := by
  sorry

end NUMINAMATH_CALUDE_eduardo_flour_amount_l1568_156849


namespace NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_23_l1568_156884

-- Define the polynomial
def p (x : ℝ) : ℝ := 4 * (2 * x^8 + 5 * x^5 - 6) + 9 * (x^6 - 8 * x^3 + 4)

-- Theorem statement
theorem sum_of_coefficients_is_negative_23 :
  p 1 = -23 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_is_negative_23_l1568_156884


namespace NUMINAMATH_CALUDE_solve_sticker_price_l1568_156873

def sticker_price_problem (p : ℝ) : Prop :=
  let store_a_price := 1.08 * (0.8 * p - 120)
  let store_b_price := 1.08 * (0.7 * p + 50)
  store_b_price - store_a_price = 27 ∧ p = 1450

theorem solve_sticker_price : ∃ p : ℝ, sticker_price_problem p := by
  sorry

end NUMINAMATH_CALUDE_solve_sticker_price_l1568_156873


namespace NUMINAMATH_CALUDE_lg_24_in_terms_of_a_b_l1568_156894

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem lg_24_in_terms_of_a_b (a b : ℝ) (h1 : lg 6 = a) (h2 : lg 12 = b) :
  lg 24 = 2 * b - a := by
  sorry

end NUMINAMATH_CALUDE_lg_24_in_terms_of_a_b_l1568_156894


namespace NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l1568_156814

theorem lcm_from_hcf_and_product (a b : ℕ+) : 
  Nat.gcd a b = 9 → a * b = 1800 → Nat.lcm a b = 200 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_hcf_and_product_l1568_156814


namespace NUMINAMATH_CALUDE_intersection_y_diff_zero_l1568_156821

def f (x : ℝ) : ℝ := 2 - x^2 + x^4
def g (x : ℝ) : ℝ := -1 + x^2 + x^4

theorem intersection_y_diff_zero :
  let intersection_points := {x : ℝ | f x = g x}
  ∃ (x₁ x₂ : ℝ), x₁ ∈ intersection_points ∧ x₂ ∈ intersection_points ∧
    ∀ (y₁ y₂ : ℝ), y₁ = f x₁ ∧ y₂ = f x₂ → |y₁ - y₂| = 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_y_diff_zero_l1568_156821


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1568_156872

theorem triangle_abc_properties (A B C : Real) (m n : Real × Real) :
  -- Given conditions
  m = (Real.sin A, Real.sin B) →
  n = (Real.cos B, Real.cos A) →
  m.1 * n.1 + m.2 * n.2 = Real.sin (2 * C) →
  A + B + C = π →
  2 * Real.sin C = Real.sin A + Real.sin B →
  Real.sin A * Real.sin C * (Real.sin B - Real.sin A) = 18 →
  -- Conclusions
  C = π / 3 ∧ 
  2 * Real.sin A * Real.sin B * Real.cos C = 18 ∧
  Real.sin A * Real.sin B = 16 ∧
  Real.sin C = Real.sin A + Real.sin B - Real.sin A * Real.sin B / 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1568_156872


namespace NUMINAMATH_CALUDE_ferry_tourists_sum_l1568_156804

/-- The number of trips the ferry makes in a day -/
def num_trips : ℕ := 9

/-- The initial number of tourists on the first trip -/
def initial_tourists : ℕ := 120

/-- The decrease in number of tourists for each subsequent trip -/
def tourist_decrease : ℤ := 2

/-- The sum of an arithmetic sequence -/
def arithmetic_sum (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) / 2 * (2 * a₁ + (n - 1 : ℕ) * d)

theorem ferry_tourists_sum :
  arithmetic_sum initial_tourists (-tourist_decrease) num_trips = 1008 := by
  sorry

end NUMINAMATH_CALUDE_ferry_tourists_sum_l1568_156804


namespace NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l1568_156838

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

-- Define the theorem
theorem fibonacci_arithmetic_sequence (a b c : ℕ) :
  (fib a < fib b) ∧ (fib b < fib c) ∧  -- Fₐ, Fₑ, Fₒ form an increasing sequence
  (fib (a + 1) < fib (b + 1)) ∧ (fib (b + 1) < fib (c + 1)) ∧  -- Fₐ₊₁, Fₑ₊₁, Fₒ₊₁ form an increasing sequence
  (fib c - fib b = fib b - fib a) ∧  -- Arithmetic sequence condition
  (fib (c + 1) - fib (b + 1) = fib (b + 1) - fib (a + 1)) ∧  -- Arithmetic sequence condition for next terms
  (a + b + c = 3000) →  -- Sum condition
  a = 999 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_arithmetic_sequence_l1568_156838


namespace NUMINAMATH_CALUDE_P₁_subset_P₂_l1568_156819

/-- P₁ is the set of real numbers x such that x² + ax + 1 > 0 -/
def P₁ (a : ℝ) : Set ℝ := {x | x^2 + a*x + 1 > 0}

/-- P₂ is the set of real numbers x such that x² + ax + 2 > 0 -/
def P₂ (a : ℝ) : Set ℝ := {x | x^2 + a*x + 2 > 0}

/-- For all real numbers a, P₁(a) is a subset of P₂(a) -/
theorem P₁_subset_P₂ : ∀ a : ℝ, P₁ a ⊆ P₂ a := by
  sorry

end NUMINAMATH_CALUDE_P₁_subset_P₂_l1568_156819


namespace NUMINAMATH_CALUDE_hallway_floor_design_ratio_l1568_156839

/-- Given a rectangle with semicircles on either side, where the ratio of length to width
    is 4:1 and the width is 20 inches, the ratio of the area of the rectangle to the
    combined area of the semicircles is 16/π. -/
theorem hallway_floor_design_ratio : 
  ∀ (length width : ℝ),
  width = 20 →
  length = 4 * width →
  (length * width) / (π * (width / 2)^2) = 16 / π :=
by sorry

end NUMINAMATH_CALUDE_hallway_floor_design_ratio_l1568_156839


namespace NUMINAMATH_CALUDE_isabelle_concert_savings_l1568_156815

/-- The number of weeks Isabelle needs to work to afford concert tickets for herself and her brothers -/
def weeks_to_work (isabelle_ticket_cost : ℕ) (brother_ticket_cost : ℕ) (isabelle_savings : ℕ) (brothers_savings : ℕ) (weekly_pay : ℕ) : ℕ :=
  let total_cost := isabelle_ticket_cost + 2 * brother_ticket_cost
  let total_savings := isabelle_savings + brothers_savings
  let remaining_cost := total_cost - total_savings
  remaining_cost / weekly_pay

theorem isabelle_concert_savings : weeks_to_work 20 10 5 5 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_isabelle_concert_savings_l1568_156815


namespace NUMINAMATH_CALUDE_count_possible_sums_l1568_156893

/-- The set of integers from 1 to 150 -/
def S : Finset ℕ := Finset.range 150

/-- The size of subset C -/
def k : ℕ := 80

/-- The minimum possible sum of k elements from S -/
def min_sum : ℕ := k * (k + 1) / 2

/-- The maximum possible sum of k elements from S -/
def max_sum : ℕ := (Finset.sum S id - (150 - k) * (150 - k + 1) / 2)

/-- The number of possible values for the sum of k elements from S -/
def num_possible_sums : ℕ := max_sum - min_sum + 1

theorem count_possible_sums :
  num_possible_sums = 6844 := by sorry

end NUMINAMATH_CALUDE_count_possible_sums_l1568_156893


namespace NUMINAMATH_CALUDE_ship_speed_calculation_l1568_156840

/-- The speed of the train in km/h -/
def train_speed : ℝ := 48

/-- The total distance traveled in km -/
def total_distance : ℝ := 480

/-- The additional time taken by the train in hours -/
def additional_time : ℝ := 2

/-- The speed of the ship in km/h -/
def ship_speed : ℝ := 60

theorem ship_speed_calculation : 
  (total_distance / ship_speed) + additional_time = total_distance / train_speed := by
  sorry

#check ship_speed_calculation

end NUMINAMATH_CALUDE_ship_speed_calculation_l1568_156840


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l1568_156853

/-- Given an ellipse with equation x^2 + 9y^2 = 8100, 
    the distance between its foci is 120√2 -/
theorem ellipse_foci_distance : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, x^2 + 9*y^2 = 8100 → x^2/a^2 + y^2/b^2 = 1) ∧
    c^2 = a^2 - b^2 ∧
    2*c = 120*Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l1568_156853


namespace NUMINAMATH_CALUDE_neg_eight_celsius_meaning_l1568_156829

/-- Represents temperature in Celsius -/
structure Temperature where
  value : ℤ
  unit : String
  deriving Repr

/-- Converts a temperature to its representation relative to zero -/
def tempRelativeToZero (t : Temperature) : String :=
  if t.value > 0 then
    s!"{t.value}°C above zero"
  else if t.value < 0 then
    s!"{-t.value}°C below zero"
  else
    "0°C"

/-- The convention for representing temperatures -/
axiom temp_convention (t : Temperature) : 
  t.value > 0 → tempRelativeToZero t = s!"{t.value}°C above zero"

/-- Theorem: -8°C represents 8°C below zero -/
theorem neg_eight_celsius_meaning :
  let t : Temperature := ⟨-8, "C"⟩
  tempRelativeToZero t = "8°C below zero" := by
  sorry

end NUMINAMATH_CALUDE_neg_eight_celsius_meaning_l1568_156829


namespace NUMINAMATH_CALUDE_balloon_permutations_l1568_156820

def balloon_letters : Nat := 7
def l_count : Nat := 2
def o_count : Nat := 3

theorem balloon_permutations :
  (balloon_letters.factorial) / (l_count.factorial * o_count.factorial) = 420 := by
  sorry

end NUMINAMATH_CALUDE_balloon_permutations_l1568_156820


namespace NUMINAMATH_CALUDE_composition_equation_solution_l1568_156835

/-- Given functions f and g, and a condition on their composition, prove the value of a. -/
theorem composition_equation_solution (a : ℝ) : 
  (let f (x : ℝ) := (x + 4) / 7 + 2
   let g (x : ℝ) := 5 - 2 * x
   f (g a) = 8) → 
  a = -33/2 := by
sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l1568_156835


namespace NUMINAMATH_CALUDE_fred_movie_change_l1568_156824

theorem fred_movie_change 
  (ticket_price : ℚ)
  (num_tickets : ℕ)
  (borrowed_movie_price : ℚ)
  (paid_amount : ℚ)
  (h1 : ticket_price = 592/100)
  (h2 : num_tickets = 2)
  (h3 : borrowed_movie_price = 679/100)
  (h4 : paid_amount = 20) :
  paid_amount - (ticket_price * num_tickets + borrowed_movie_price) = 137/100 := by
  sorry

end NUMINAMATH_CALUDE_fred_movie_change_l1568_156824


namespace NUMINAMATH_CALUDE_cubic_inequality_l1568_156880

theorem cubic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 + b^3 + c^3 + 3*a*b*c > a*b*(a+b) + b*c*(b+c) + a*c*(a+c) := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1568_156880


namespace NUMINAMATH_CALUDE_expression_equality_l1568_156863

theorem expression_equality : (-1)^2023 - Real.sqrt 9 + |1 - Real.sqrt 2| - ((-8) ^ (1/3 : ℝ)) = Real.sqrt 2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l1568_156863


namespace NUMINAMATH_CALUDE_smallest_k_inequality_l1568_156822

theorem smallest_k_inequality (x y z : ℝ) :
  ∃ (k : ℝ), k = 3 ∧ (x^2 + y^2 + z^2)^2 ≤ k * (x^4 + y^4 + z^4) ∧
  ∀ (k' : ℝ), (∀ (a b c : ℝ), (a^2 + b^2 + c^2)^2 ≤ k' * (a^4 + b^4 + c^4)) → k' ≥ k :=
sorry

end NUMINAMATH_CALUDE_smallest_k_inequality_l1568_156822


namespace NUMINAMATH_CALUDE_weekly_jog_distance_l1568_156865

/-- The total distance jogged throughout the week in kilometers -/
def total_distance (mon tue wed thu fri_miles : ℝ) (mile_to_km : ℝ) : ℝ :=
  mon + tue + wed + thu + (fri_miles * mile_to_km)

/-- Theorem stating the total distance jogged throughout the week -/
theorem weekly_jog_distance :
  let mon := 3
  let tue := 5.5
  let wed := 9.7
  let thu := 10.8
  let fri_miles := 2
  let mile_to_km := 1.60934
  total_distance mon tue wed thu fri_miles mile_to_km = 32.21868 := by
  sorry

end NUMINAMATH_CALUDE_weekly_jog_distance_l1568_156865


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l1568_156886

-- Define the lines
def line1 (x : ℝ) : ℝ := 3 * x - 3
def line2 (x : ℝ) : ℝ := -2 * x + 14
def line3 : ℝ := 0
def line4 : ℝ := 5

-- Define the vertices of the quadrilateral
def vertex1 : ℝ × ℝ := (0, line1 0)
def vertex2 : ℝ × ℝ := (0, line2 0)
def vertex3 : ℝ × ℝ := (line4, line1 line4)
def vertex4 : ℝ × ℝ := (line4, line2 line4)

-- Define the area of the quadrilateral
def quadrilateralArea : ℝ := 80

-- Theorem statement
theorem area_of_quadrilateral :
  let vertices := [vertex1, vertex2, vertex3, vertex4]
  quadrilateralArea = 80 := by sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l1568_156886


namespace NUMINAMATH_CALUDE_coefficient_x3_is_80_l1568_156844

/-- The coefficient of x^3 in the expansion of (1+2x)^5 -/
def coefficient_x3 : ℕ :=
  Nat.choose 5 3 * 2^3

/-- Theorem stating that the coefficient of x^3 in (1+2x)^5 is 80 -/
theorem coefficient_x3_is_80 : coefficient_x3 = 80 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3_is_80_l1568_156844


namespace NUMINAMATH_CALUDE_complex_roots_circle_l1568_156890

theorem complex_roots_circle (z : ℂ) : 
  (z + 1)^6 = 243 * z^6 → Complex.abs (z - Complex.ofReal (1/8)) = 1/8 := by
  sorry

end NUMINAMATH_CALUDE_complex_roots_circle_l1568_156890


namespace NUMINAMATH_CALUDE_fraction_sum_l1568_156899

theorem fraction_sum (m n : ℚ) (h : n / m = 3 / 7) : (m + n) / m = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l1568_156899


namespace NUMINAMATH_CALUDE_first_line_time_l1568_156837

/-- Represents the productivity of a production line -/
structure ProductivityRate where
  rate : ℝ
  rate_pos : rate > 0

/-- Represents a production line -/
structure ProductionLine where
  productivity : ProductivityRate

/-- Represents a system of three production lines -/
structure ProductionSystem where
  line1 : ProductionLine
  line2 : ProductionLine
  line3 : ProductionLine
  combined_productivity : ProductivityRate
  first_second_productivity : ProductivityRate
  combined_vs_first_second : combined_productivity.rate = 1.5 * first_second_productivity.rate
  second_faster_than_first : line2.productivity.rate = line1.productivity.rate + (1 / 2)
  second_third_vs_first : 
    1 / line1.productivity.rate - (24 / 5) = 
    1 / (line2.productivity.rate + line3.productivity.rate)

theorem first_line_time (system : ProductionSystem) : 
  1 / system.line1.productivity.rate = 8 := by
  sorry

end NUMINAMATH_CALUDE_first_line_time_l1568_156837


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l1568_156847

theorem abs_sum_inequality : 
  (∃ (a : ℝ), ∀ (x : ℝ), |x - 4| + |x - 6| ≥ a) ∧ 
  (∀ (b : ℝ), (∀ (x : ℝ), |x - 4| + |x - 6| ≥ b) → b ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l1568_156847


namespace NUMINAMATH_CALUDE_analytic_method_characterization_l1568_156842

/-- Enumeration of proof methods --/
inductive ProofMethod
  | MathematicalInduction
  | ProofByContradiction
  | AnalyticMethod
  | SyntheticMethod

/-- Characteristic of a proof method --/
def isCharacterizedBy (m : ProofMethod) (c : String) : Prop :=
  match m with
  | ProofMethod.AnalyticMethod => c = "seeking the cause from the effect"
  | _ => c ≠ "seeking the cause from the effect"

/-- Theorem stating that the Analytic Method is characterized by "seeking the cause from the effect" --/
theorem analytic_method_characterization :
  isCharacterizedBy ProofMethod.AnalyticMethod "seeking the cause from the effect" :=
by sorry

end NUMINAMATH_CALUDE_analytic_method_characterization_l1568_156842


namespace NUMINAMATH_CALUDE_shaded_area_is_three_l1568_156809

def grid_area : ℕ := 3 * 2 + 4 * 6 + 5 * 3

def unshaded_triangle_area : ℕ := (14 * 6) / 2

def shaded_area : ℕ := grid_area - unshaded_triangle_area

theorem shaded_area_is_three : shaded_area = 3 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_three_l1568_156809


namespace NUMINAMATH_CALUDE_linear_equation_root_conditions_l1568_156885

/-- Conditions for roots of a linear equation -/
theorem linear_equation_root_conditions (a b : ℝ) :
  let x := -b / a
  (x > 0 ↔ a * b < 0) ∧
  (x < 0 ↔ a * b > 0) ∧
  (x = 0 ↔ b = 0 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_root_conditions_l1568_156885


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l1568_156864

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log (1 - x)}

-- Theorem to prove
theorem intersection_A_complement_B :
  A ∩ (U \ B) = {x : ℝ | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l1568_156864


namespace NUMINAMATH_CALUDE_prime_sum_implies_prime_exponent_l1568_156850

theorem prime_sum_implies_prime_exponent (p d : ℕ) : 
  Prime p → p = (10^d - 1) / 9 → Prime d := by
  sorry

end NUMINAMATH_CALUDE_prime_sum_implies_prime_exponent_l1568_156850


namespace NUMINAMATH_CALUDE_business_profit_calculation_l1568_156866

/-- Represents the total profit of a business partnership --/
def total_profit (a_investment b_investment : ℕ) (a_management_fee : ℚ) (a_total_received : ℕ) : ℚ :=
  let total_investment := a_investment + b_investment
  let remaining_profit_share := 1 - a_management_fee
  let a_profit_share := (a_investment : ℚ) / (total_investment : ℚ) * remaining_profit_share
  (a_total_received : ℚ) / (a_management_fee + a_profit_share)

/-- Theorem stating the total profit of the business partnership --/
theorem business_profit_calculation :
  total_profit 3500 2500 (1/10) 6000 = 9600 := by
  sorry

#eval total_profit 3500 2500 (1/10) 6000

end NUMINAMATH_CALUDE_business_profit_calculation_l1568_156866


namespace NUMINAMATH_CALUDE_range_of_a_l1568_156826

theorem range_of_a (a : ℝ) 
  (h1 : ∀ x : ℝ, x^2 - a ≥ 0)
  (h2 : ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) :
  a ≤ -2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1568_156826


namespace NUMINAMATH_CALUDE_line_intersection_range_l1568_156870

theorem line_intersection_range (a : ℝ) : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 3 ∧ 2 * x + (3 - a) = 0) ↔ 5 ≤ a ∧ a ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_range_l1568_156870


namespace NUMINAMATH_CALUDE_flight_time_estimate_l1568_156881

/-- The radius of the circular path in miles -/
def radius : ℝ := 3950

/-- The speed of the object in miles per hour -/
def speed : ℝ := 550

/-- The approximate value of π -/
def π_approx : ℝ := 3.14

/-- The theorem stating that the time taken to complete one revolution is approximately 45 hours -/
theorem flight_time_estimate :
  let circumference := 2 * π_approx * radius
  let exact_time := circumference / speed
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |exact_time - 45| < ε :=
sorry

end NUMINAMATH_CALUDE_flight_time_estimate_l1568_156881


namespace NUMINAMATH_CALUDE_price_change_equivalence_l1568_156825

theorem price_change_equivalence (initial_price : ℝ) (x : ℝ) 
  (h1 : initial_price > 0)
  (h2 : x > 0 ∧ x < 100) :
  (1.25 * initial_price) * (1 - x / 100) = 1.125 * initial_price → x = 10 := by
sorry

end NUMINAMATH_CALUDE_price_change_equivalence_l1568_156825


namespace NUMINAMATH_CALUDE_class_weighted_average_l1568_156877

/-- Calculates the weighted average score for a class with three groups of students -/
theorem class_weighted_average (total_students : ℕ) 
  (group1_count : ℕ) (group1_avg : ℚ)
  (group2_count : ℕ) (group2_avg : ℚ)
  (group3_count : ℕ) (group3_avg : ℚ)
  (h1 : total_students = group1_count + group2_count + group3_count)
  (h2 : total_students = 30)
  (h3 : group1_count = 12)
  (h4 : group2_count = 10)
  (h5 : group3_count = 8)
  (h6 : group1_avg = 72 / 100)
  (h7 : group2_avg = 85 / 100)
  (h8 : group3_avg = 92 / 100) :
  (group1_count * group1_avg + 2 * group2_count * group2_avg + group3_count * group3_avg) / 
  (group1_count + 2 * group2_count + group3_count) = 825 / 1000 := by
  sorry


end NUMINAMATH_CALUDE_class_weighted_average_l1568_156877


namespace NUMINAMATH_CALUDE_modular_power_congruence_l1568_156803

theorem modular_power_congruence (p : ℕ) (n : ℕ) (a b : ℤ) 
  (h_prime : Nat.Prime p) (h_cong : a ≡ b [ZMOD p^n]) :
  a^p ≡ b^p [ZMOD p^(n+1)] := by sorry

end NUMINAMATH_CALUDE_modular_power_congruence_l1568_156803
