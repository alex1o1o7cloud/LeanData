import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l1860_186004

theorem consecutive_integers_sum (x : ℤ) : 
  x + 1 < 20 → 
  x * (x + 1) + x + (x + 1) = 156 → 
  x + (x + 1) = 23 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l1860_186004


namespace NUMINAMATH_CALUDE_intersection_empty_implies_t_geq_one_l1860_186021

theorem intersection_empty_implies_t_geq_one (t : ℝ) : 
  let M : Set ℝ := {x | x ≤ 1}
  let P : Set ℝ := {x | x > t}
  (M ∩ P = ∅) → t ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_t_geq_one_l1860_186021


namespace NUMINAMATH_CALUDE_tangent_surface_area_l1860_186054

/-- Given a sphere of radius R and a point S at distance 2R from the center,
    the surface area formed by tangent lines from S to the sphere is 3πR^2/2 -/
theorem tangent_surface_area (R : ℝ) (h : R > 0) :
  let sphere_radius := R
  let point_distance := 2 * R
  let surface_area := (3 / 2) * π * R^2
  surface_area = (3 / 2) * π * sphere_radius^2 :=
by sorry

end NUMINAMATH_CALUDE_tangent_surface_area_l1860_186054


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1860_186088

theorem imaginary_part_of_z (z : ℂ) (h : z + Complex.abs z = 1 + 2*I) : 
  Complex.im z = 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1860_186088


namespace NUMINAMATH_CALUDE_quadratic_roots_l1860_186051

/-- A quadratic function passing through specific points has roots -4 and 1 -/
theorem quadratic_roots (a b c : ℝ) (h_a : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (f (-5) = 6) ∧ (f (-4) = 0) ∧ (f (-2) = -6) ∧ (f 0 = -4) ∧ (f 2 = 6) →
  (∀ x, f x = 0 ↔ x = -4 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l1860_186051


namespace NUMINAMATH_CALUDE_vector_square_difference_l1860_186009

theorem vector_square_difference (a b : ℝ × ℝ) 
  (h1 : a + b = (-3, 6)) 
  (h2 : a - b = (-3, 2)) 
  (h3 : a ≠ (0, 0)) 
  (h4 : b ≠ (0, 0)) : 
  (a.1^2 + a.2^2) - (b.1^2 + b.2^2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_vector_square_difference_l1860_186009


namespace NUMINAMATH_CALUDE_batsman_average_increase_l1860_186038

theorem batsman_average_increase 
  (innings : Nat) 
  (last_score : Nat) 
  (final_average : Nat) 
  (h1 : innings = 12) 
  (h2 : last_score = 75) 
  (h3 : final_average = 64) : 
  (final_average : ℚ) - (((innings : ℚ) * final_average - last_score) / (innings - 1)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l1860_186038


namespace NUMINAMATH_CALUDE_population_reaches_max_capacity_in_140_years_l1860_186027

def initial_year : ℕ := 1998
def initial_population : ℕ := 200
def total_land : ℕ := 32000
def space_per_person : ℕ := 2
def growth_period : ℕ := 20

def max_capacity : ℕ := total_land / space_per_person

def population_after_years (years : ℕ) : ℕ :=
  initial_population * 2^(years / growth_period)

theorem population_reaches_max_capacity_in_140_years :
  ∃ (y : ℕ), y = 140 ∧ 
  population_after_years y ≥ max_capacity ∧
  population_after_years (y - growth_period) < max_capacity :=
sorry

end NUMINAMATH_CALUDE_population_reaches_max_capacity_in_140_years_l1860_186027


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l1860_186050

theorem other_root_of_quadratic (a : ℝ) : 
  ((-1)^2 + a*(-1) - 2 = 0) → (2^2 + a*2 - 2 = 0) := by
  sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l1860_186050


namespace NUMINAMATH_CALUDE_natural_number_operations_l1860_186074

theorem natural_number_operations (x y : ℕ) (h1 : x > y) (h2 : x + y + (x - y) + x * y + x / y = 243) :
  (x = 54 ∧ y = 2) ∨ (x = 24 ∧ y = 8) := by
  sorry

end NUMINAMATH_CALUDE_natural_number_operations_l1860_186074


namespace NUMINAMATH_CALUDE_equation_solution_range_l1860_186066

theorem equation_solution_range (m : ℝ) :
  (∃ x : ℝ, 1 - 2 * Real.sin x ^ 2 + 2 * Real.cos x - m = 0) ↔ -3/2 ≤ m ∧ m ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_range_l1860_186066


namespace NUMINAMATH_CALUDE_weekend_grass_cutting_time_l1860_186017

/-- Calculates the total time Jason spends cutting grass over the weekend -/
def total_weekend_time (small_time medium_time large_time break_time weather_delay : ℕ)
  (saturday_small saturday_medium saturday_large : ℕ)
  (sunday_medium sunday_large : ℕ) : ℕ :=
  let saturday_time := 
    saturday_small * small_time + 
    saturday_medium * medium_time + 
    saturday_large * large_time + 
    (saturday_small + saturday_medium + saturday_large - 1) * break_time
  let sunday_time := 
    sunday_medium * (medium_time + weather_delay) + 
    sunday_large * (large_time + weather_delay) + 
    (sunday_medium + sunday_large - 1) * break_time
  saturday_time + sunday_time

theorem weekend_grass_cutting_time :
  total_weekend_time 25 30 40 5 10 2 4 2 6 2 = 11 * 60 := by
  sorry

end NUMINAMATH_CALUDE_weekend_grass_cutting_time_l1860_186017


namespace NUMINAMATH_CALUDE_m_equals_two_iff_parallel_l1860_186015

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := 2 * x - m * y - 1 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := (m - 1) * x - y + 1 = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := ∀ (x y : ℝ), l₁ m x y ↔ l₂ m x y

-- Theorem statement
theorem m_equals_two_iff_parallel :
  ∀ m : ℝ, m = 2 ↔ parallel m := by sorry

end NUMINAMATH_CALUDE_m_equals_two_iff_parallel_l1860_186015


namespace NUMINAMATH_CALUDE_birds_on_fence_l1860_186087

theorem birds_on_fence (initial_birds : ℕ) (new_birds : ℕ) : 
  initial_birds = 12 → new_birds = 8 → initial_birds + new_birds = 20 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l1860_186087


namespace NUMINAMATH_CALUDE_paco_marble_purchase_l1860_186094

theorem paco_marble_purchase : 
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_paco_marble_purchase_l1860_186094


namespace NUMINAMATH_CALUDE_no_real_solutions_l1860_186069

theorem no_real_solutions :
  ∀ x : ℝ, x ≠ 2 → (3 * x^2) / (x - 2) - (x + 4) / 4 + (5 - 3 * x) / (x - 2) + 2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_solutions_l1860_186069


namespace NUMINAMATH_CALUDE_library_book_distributions_l1860_186013

-- Define the total number of books
def total_books : ℕ := 8

-- Define the function that calculates the number of valid distributions
def valid_distributions (n : ℕ) : ℕ :=
  -- Count distributions where books in library range from 1 to (n - 2)
  (Finset.range (n - 2)).card

-- Theorem statement
theorem library_book_distributions : 
  valid_distributions total_books = 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_library_book_distributions_l1860_186013


namespace NUMINAMATH_CALUDE_range_of_a_l1860_186043

theorem range_of_a (a : ℝ) : (2 * a - 1) ^ 0 = 1 → a ≠ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1860_186043


namespace NUMINAMATH_CALUDE_crazy_silly_school_books_read_l1860_186059

/-- The number of books read in the 'Crazy Silly School' series -/
def books_read (total_books unread_books : ℕ) : ℕ :=
  total_books - unread_books

/-- Theorem stating that the number of books read is 33 -/
theorem crazy_silly_school_books_read :
  books_read 50 17 = 33 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_books_read_l1860_186059


namespace NUMINAMATH_CALUDE_jimmy_needs_four_packs_l1860_186032

/-- The number of packs of bread Jimmy needs to buy for his sandwiches -/
def breadPacksNeeded (sandwiches : ℕ) (slicesPerSandwich : ℕ) (slicesPerPack : ℕ) : ℕ :=
  (sandwiches * slicesPerSandwich + slicesPerPack - 1) / slicesPerPack

/-- Proof that Jimmy needs to buy 4 packs of bread -/
theorem jimmy_needs_four_packs :
  breadPacksNeeded 8 2 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_needs_four_packs_l1860_186032


namespace NUMINAMATH_CALUDE_min_value_geometric_sequence_l1860_186024

def geometric_sequence (a₁ : ℝ) (r : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => geometric_sequence a₁ r n * r

def expression (a₁ r : ℝ) : ℝ :=
  5 * (geometric_sequence a₁ r 1) + 6 * (geometric_sequence a₁ r 2)

theorem min_value_geometric_sequence :
  ∃ (min_val : ℝ), min_val = -25/12 ∧
  ∀ (r : ℝ), expression 2 r ≥ min_val :=
sorry

end NUMINAMATH_CALUDE_min_value_geometric_sequence_l1860_186024


namespace NUMINAMATH_CALUDE_ratio_problem_l1860_186033

theorem ratio_problem (a b c : ℝ) (h1 : b/a = 3) (h2 : c/b = 4) : (a + b) / (b + c) = 4/15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l1860_186033


namespace NUMINAMATH_CALUDE_possible_values_of_a_l1860_186085

theorem possible_values_of_a (x y a : ℝ) 
  (h1 : x + y = a) 
  (h2 : x^3 + y^3 = a) 
  (h3 : x^5 + y^5 = a) : 
  a ∈ ({-2, -1, 0, 1, 2} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l1860_186085


namespace NUMINAMATH_CALUDE_parallelogram_sum_xy_l1860_186001

/-- A parallelogram with sides measuring 10, 4x+2, 12y-2, and 10 units consecutively has x + y = 3 -/
theorem parallelogram_sum_xy (x y : ℝ) : 
  (10 : ℝ) = 4*x + 2 ∧ (10 : ℝ) = 12*y - 2 → x + y = 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_sum_xy_l1860_186001


namespace NUMINAMATH_CALUDE_closed_under_subtraction_l1860_186095

/-- A set of integers with special properties -/
structure SpecialIntegerSet where
  M : Set Int
  has_pos : ∃ x ∈ M, x > 0
  has_neg : ∃ x ∈ M, x < 0
  closed_double : ∀ a ∈ M, (2 * a) ∈ M
  closed_sum : ∀ a b, a ∈ M → b ∈ M → (a + b) ∈ M

/-- The main theorem: M is closed under subtraction -/
theorem closed_under_subtraction (S : SpecialIntegerSet) :
  ∀ a b, a ∈ S.M → b ∈ S.M → (a - b) ∈ S.M := by
  sorry

end NUMINAMATH_CALUDE_closed_under_subtraction_l1860_186095


namespace NUMINAMATH_CALUDE_terminal_side_in_quadrant_II_l1860_186045

def α : Real := 2

theorem terminal_side_in_quadrant_II :
  π / 2 < α ∧ α < π :=
sorry

end NUMINAMATH_CALUDE_terminal_side_in_quadrant_II_l1860_186045


namespace NUMINAMATH_CALUDE_weight_system_properties_l1860_186012

/-- Represents a set of weights -/
def Weights : List ℕ := [1, 3, 9, 27]

/-- The maximum weight that can be measured -/
def MaxWeight : ℕ := 40

/-- Checks if a weight can be represented by a combination of given weights -/
def isRepresentable (n : ℕ) (weights : List ℕ) : Prop :=
  ∃ (combination : List Bool), 
    combination.length = weights.length ∧ 
    (List.zip combination weights).foldl (λ sum (b, w) => sum + if b then w else 0) 0 = n

theorem weight_system_properties :
  (∀ n : ℕ, n ≤ MaxWeight → isRepresentable n Weights) ∧
  (∀ n : ℕ, n > MaxWeight → ¬ isRepresentable n Weights) :=
sorry

end NUMINAMATH_CALUDE_weight_system_properties_l1860_186012


namespace NUMINAMATH_CALUDE_frog_escape_probability_l1860_186047

/-- Probability of the frog surviving when starting at pad N -/
noncomputable def P (N : ℕ) : ℝ :=
  sorry

/-- The number of lilypads -/
def num_pads : ℕ := 21

/-- The starting position of the frog -/
def start_pos : ℕ := 3

theorem frog_escape_probability :
  (∀ N : ℕ, 0 < N → N < num_pads - 1 →
    P N = (2 * N : ℝ) / 20 * P (N - 1) + (1 - (2 * N : ℝ) / 20) * P (N + 1)) →
  P 0 = 0 →
  P (num_pads - 1) = 1 →
  P start_pos = 4 / 11 := by
  sorry

end NUMINAMATH_CALUDE_frog_escape_probability_l1860_186047


namespace NUMINAMATH_CALUDE_vaccine_effectiveness_theorem_l1860_186025

/-- Represents the data for a vaccine experiment -/
structure VaccineExperiment where
  total_participants : ℕ
  vaccinated_infected : ℕ
  placebo_infected : ℕ

/-- Calculates the vaccine effectiveness -/
def vaccine_effectiveness (exp : VaccineExperiment) : ℚ :=
  let p := exp.vaccinated_infected / (exp.total_participants / 2 : ℚ)
  let q := exp.placebo_infected / (exp.total_participants / 2 : ℚ)
  1 - p / q

/-- The main theorem about vaccine effectiveness -/
theorem vaccine_effectiveness_theorem (exp_A exp_B : VaccineExperiment) :
  exp_A.total_participants = 30000 →
  exp_A.vaccinated_infected = 50 →
  exp_A.placebo_infected = 500 →
  vaccine_effectiveness exp_A = 9/10 ∧
  ∃ (exp_B : VaccineExperiment),
    vaccine_effectiveness exp_B > 9/10 ∧
    exp_B.vaccinated_infected ≥ exp_A.vaccinated_infected :=
by sorry

end NUMINAMATH_CALUDE_vaccine_effectiveness_theorem_l1860_186025


namespace NUMINAMATH_CALUDE_correct_total_paid_l1860_186099

/-- Calculates the total amount paid after discount for a bulk purchase -/
def total_amount_paid (item_count : ℕ) (price_per_item : ℚ) (discount_amount : ℚ) (discount_threshold : ℚ) : ℚ :=
  let total_cost := item_count * price_per_item
  let discount_count := ⌊total_cost / discount_threshold⌋
  let total_discount := discount_count * discount_amount
  total_cost - total_discount

/-- Theorem stating the correct total amount paid for the given scenario -/
theorem correct_total_paid :
  total_amount_paid 400 (40/100) 2 10 = 128 := by
  sorry

end NUMINAMATH_CALUDE_correct_total_paid_l1860_186099


namespace NUMINAMATH_CALUDE_three_digit_integers_with_remainders_l1860_186068

theorem three_digit_integers_with_remainders : 
  let n : ℕ → Prop := λ x => 
    100 ≤ x ∧ x < 1000 ∧ 
    x % 7 = 3 ∧ 
    x % 8 = 4 ∧ 
    x % 10 = 6
  (∃! (l : List ℕ), l.length = 4 ∧ ∀ x ∈ l, n x) := by
  sorry

end NUMINAMATH_CALUDE_three_digit_integers_with_remainders_l1860_186068


namespace NUMINAMATH_CALUDE_sunglasses_cost_theorem_l1860_186019

/-- The cost of sunglasses for a vendor -/
theorem sunglasses_cost_theorem 
  (selling_price : ℝ) 
  (pairs_sold : ℕ) 
  (sign_cost : ℝ) 
  (h1 : selling_price = 30)
  (h2 : pairs_sold = 10)
  (h3 : sign_cost = 20)
  (h4 : (pairs_sold : ℝ) * (selling_price - cost_per_pair) / 2 = sign_cost) :
  cost_per_pair = 26 := by
  sorry

#check sunglasses_cost_theorem

end NUMINAMATH_CALUDE_sunglasses_cost_theorem_l1860_186019


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_with_foci_on_y_axis_l1860_186063

theorem curve_is_hyperbola_with_foci_on_y_axis (θ : Real) 
  (h1 : π < θ ∧ θ < 3*π/2) -- θ is in the third quadrant
  (h2 : ∀ x y : Real, x^2 + y^2 * Real.sin θ = Real.cos θ) -- curve equation
  : ∃ (a b : Real), 
    a > 0 ∧ b > 0 ∧ 
    (∀ x y : Real, y^2 / b^2 - x^2 / a^2 = 1) ∧ -- standard form of hyperbola with foci on y-axis
    (∃ c : Real, c > 0 ∧ c^2 = a^2 + b^2) -- condition for foci on y-axis
  := by sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_with_foci_on_y_axis_l1860_186063


namespace NUMINAMATH_CALUDE_michael_crayons_worth_l1860_186016

/-- Calculates the total worth of crayons after a purchase --/
def total_worth_after_purchase (initial_packs : ℕ) (additional_packs : ℕ) (cost_per_pack : ℚ) : ℚ :=
  (initial_packs + additional_packs) * cost_per_pack

/-- Proves that the total worth of crayons after Michael's purchase is $15 --/
theorem michael_crayons_worth :
  let initial_packs := 4
  let additional_packs := 2
  let cost_per_pack := 5/2
  total_worth_after_purchase initial_packs additional_packs cost_per_pack = 15 := by
  sorry

end NUMINAMATH_CALUDE_michael_crayons_worth_l1860_186016


namespace NUMINAMATH_CALUDE_equal_derivative_points_l1860_186023

theorem equal_derivative_points (x₀ : ℝ) : 
  (2 * x₀ = -3 * x₀^2) → (x₀ = 0 ∨ x₀ = -2/3) :=
by sorry

end NUMINAMATH_CALUDE_equal_derivative_points_l1860_186023


namespace NUMINAMATH_CALUDE_spying_arrangement_odd_l1860_186046

/-- A function representing the spying arrangement in a circular group -/
def spyingArrangement (n : ℕ) : ℕ → ℕ :=
  fun i => (i % n) + 1

/-- The theorem stating that the number of people in the spying arrangement must be odd -/
theorem spying_arrangement_odd (n : ℕ) (h : n > 0) :
  (∀ i : ℕ, i < n → spyingArrangement n (spyingArrangement n i) = (i + 2) % n + 1) →
  Odd n :=
sorry

end NUMINAMATH_CALUDE_spying_arrangement_odd_l1860_186046


namespace NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l1860_186086

/-- Definition of a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

/-- Theorem stating that (5, 12, 13) is a Pythagorean triple -/
theorem five_twelve_thirteen_pythagorean_triple :
  is_pythagorean_triple 5 12 13 := by
  sorry

end NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l1860_186086


namespace NUMINAMATH_CALUDE_area_triangle_PTU_l1860_186090

/-- Regular octagon with side length 3 -/
structure RegularOctagon :=
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Triangle formed by vertices P, T, and U in the regular octagon -/
def triangle_PTU (octagon : RegularOctagon) : Set (Fin 3 → ℝ × ℝ) := sorry

/-- Area of a triangle -/
def triangle_area (t : Set (Fin 3 → ℝ × ℝ)) : ℝ := sorry

/-- Main theorem: Area of triangle PTU in a regular octagon with side length 3 -/
theorem area_triangle_PTU (octagon : RegularOctagon) :
  triangle_area (triangle_PTU octagon) = (9 * Real.sqrt 2 + 9) / 2 := by sorry

end NUMINAMATH_CALUDE_area_triangle_PTU_l1860_186090


namespace NUMINAMATH_CALUDE_sales_revenue_error_l1860_186030

theorem sales_revenue_error (x z : ℕ) : 
  (10 ≤ x ∧ x ≤ 99) →
  (10 ≤ z ∧ z ≤ 99) →
  (1000 * z + 10 * x) - (1000 * x + 10 * z) = 2920 →
  z = x + 3 ∧ 10 ≤ x ∧ x ≤ 96 :=
by sorry

end NUMINAMATH_CALUDE_sales_revenue_error_l1860_186030


namespace NUMINAMATH_CALUDE_red_balls_count_l1860_186002

/-- Given a set of balls where the ratio of red balls to white balls is 4:5,
    and there are 20 white balls, prove that the number of red balls is 16. -/
theorem red_balls_count (total : ℕ) (red : ℕ) (white : ℕ) 
    (h1 : total = red + white)
    (h2 : red * 5 = white * 4)
    (h3 : white = 20) : red = 16 := by
  sorry

end NUMINAMATH_CALUDE_red_balls_count_l1860_186002


namespace NUMINAMATH_CALUDE_min_paper_length_l1860_186037

/-- Represents a binary message of length 2016 -/
def Message := Fin 2016 → Bool

/-- Represents a paper of length n with 10 pre-colored consecutive squares -/
structure Paper (n : ℕ) where
  squares : Fin n → Bool
  precolored_start : Fin (n - 9)
  precolored : Fin 10 → Bool

/-- A strategy for encoding and decoding messages -/
structure Strategy (n : ℕ) where
  encode : Message → Paper n → Paper n
  decode : Paper n → Message

/-- The strategy works with perfect accuracy -/
def perfect_accuracy (s : Strategy n) : Prop :=
  ∀ (m : Message) (p : Paper n), s.decode (s.encode m p) = m

/-- The main theorem: The minimum value of n for which a perfect strategy exists is 2026 -/
theorem min_paper_length :
  (∃ (s : Strategy 2026), perfect_accuracy s) ∧
  (∀ (n : ℕ), n < 2026 → ¬∃ (s : Strategy n), perfect_accuracy s) :=
sorry

end NUMINAMATH_CALUDE_min_paper_length_l1860_186037


namespace NUMINAMATH_CALUDE_two_white_balls_probability_l1860_186052

/-- The probability of drawing two white balls successively without replacement
    from a box containing 8 white balls and 9 black balls is 7/34. -/
theorem two_white_balls_probability :
  let total_balls : ℕ := 8 + 9
  let white_balls : ℕ := 8
  let black_balls : ℕ := 9
  let prob_first_white : ℚ := white_balls / total_balls
  let prob_second_white : ℚ := (white_balls - 1) / (total_balls - 1)
  prob_first_white * prob_second_white = 7 / 34 := by
  sorry

end NUMINAMATH_CALUDE_two_white_balls_probability_l1860_186052


namespace NUMINAMATH_CALUDE_solve_equation_and_expression_l1860_186075

theorem solve_equation_and_expression (x : ℝ) (h : 5 * x - 7 = 15 * x + 13) : 
  3 * (x - 4) + 2 = -16 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_and_expression_l1860_186075


namespace NUMINAMATH_CALUDE_basketball_team_score_l1860_186039

theorem basketball_team_score (two_pointers three_pointers free_throws : ℕ) : 
  (3 * three_pointers = 2 * (2 * two_pointers)) →
  (free_throws = 2 * two_pointers) →
  (2 * two_pointers + 3 * three_pointers + free_throws = 65) →
  free_throws = 18 := by
sorry

end NUMINAMATH_CALUDE_basketball_team_score_l1860_186039


namespace NUMINAMATH_CALUDE_field_division_l1860_186062

theorem field_division (total_area smaller_area larger_area : ℝ) : 
  total_area = 700 ∧ 
  smaller_area + larger_area = total_area ∧ 
  larger_area - smaller_area = (1 / 5) * ((smaller_area + larger_area) / 2) →
  smaller_area = 315 :=
by sorry

end NUMINAMATH_CALUDE_field_division_l1860_186062


namespace NUMINAMATH_CALUDE_probability_of_selecting_particular_student_l1860_186031

/-- The probability of selecting a particular student from an institute with multiple classes. -/
theorem probability_of_selecting_particular_student
  (total_classes : ℕ)
  (students_per_class : ℕ)
  (selected_students : ℕ)
  (h1 : total_classes = 8)
  (h2 : students_per_class = 40)
  (h3 : selected_students = 3)
  (h4 : selected_students ≤ total_classes) :
  (selected_students : ℚ) / (total_classes * students_per_class : ℚ) = 3 / 320 := by
  sorry

#check probability_of_selecting_particular_student

end NUMINAMATH_CALUDE_probability_of_selecting_particular_student_l1860_186031


namespace NUMINAMATH_CALUDE_sin_240_degrees_l1860_186084

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_240_degrees_l1860_186084


namespace NUMINAMATH_CALUDE_joan_spent_four_half_dollars_on_wednesday_l1860_186026

/-- The number of half-dollars Joan spent on Wednesday -/
def wednesday_half_dollars : ℕ := sorry

/-- The number of half-dollars Joan spent on Thursday -/
def thursday_half_dollars : ℕ := 14

/-- The value of a half-dollar in dollars -/
def half_dollar_value : ℚ := 1/2

/-- The total amount Joan spent in dollars -/
def total_spent : ℚ := 9

/-- Theorem: Joan spent 4 half-dollars on Wednesday -/
theorem joan_spent_four_half_dollars_on_wednesday :
  wednesday_half_dollars = 4 :=
by
  have h1 : (wednesday_half_dollars : ℚ) * half_dollar_value + 
            (thursday_half_dollars : ℚ) * half_dollar_value = total_spent :=
    sorry
  sorry

end NUMINAMATH_CALUDE_joan_spent_four_half_dollars_on_wednesday_l1860_186026


namespace NUMINAMATH_CALUDE_inequality_proof_l1860_186057

theorem inequality_proof (x₁ x₂ x₃ : ℝ) (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) :
  (x₁^2 + x₂^2 + x₃^2)^3 ≤ 3 * (x₁^3 + x₂^3 + x₃^3)^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1860_186057


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l1860_186020

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def N : Set ℕ := {4, 5, 6}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l1860_186020


namespace NUMINAMATH_CALUDE_water_needed_in_quarts_l1860_186097

/-- Represents the ratio of water to lemon juice in the lemonade mixture -/
def water_to_lemon_ratio : ℚ := 4

/-- Represents the total number of parts in the mixture -/
def total_parts : ℚ := water_to_lemon_ratio + 1

/-- Represents the total volume of the mixture in gallons -/
def total_volume : ℚ := 1

/-- Represents the number of quarts in a gallon -/
def quarts_per_gallon : ℚ := 4

/-- Theorem stating the amount of water needed in quarts -/
theorem water_needed_in_quarts : 
  (water_to_lemon_ratio / total_parts) * total_volume * quarts_per_gallon = 16/5 := by
  sorry

end NUMINAMATH_CALUDE_water_needed_in_quarts_l1860_186097


namespace NUMINAMATH_CALUDE_parabola_c_value_l1860_186006

/-- A parabola passing through two points -/
def Parabola (b c : ℝ) : ℝ → ℝ := fun x ↦ 2 * x^2 + b * x + c

theorem parabola_c_value :
  ∀ b c : ℝ, 
  Parabola b c 2 = 12 ∧ 
  Parabola b c 4 = 44 →
  c = -4 := by
sorry

end NUMINAMATH_CALUDE_parabola_c_value_l1860_186006


namespace NUMINAMATH_CALUDE_sum_of_integers_l1860_186071

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x.val^2 + y.val^2 = 130) 
  (h2 : x.val * y.val = 36) : 
  x.val + y.val = Real.sqrt 202 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l1860_186071


namespace NUMINAMATH_CALUDE_cos_240_degrees_l1860_186076

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l1860_186076


namespace NUMINAMATH_CALUDE_x_coordinate_of_Q_l1860_186060

theorem x_coordinate_of_Q (P Q : ℝ × ℝ) (α : ℝ) : 
  P = (3/5, 4/5) →
  (Q.1 < 0 ∧ Q.2 < 0) →
  Real.sqrt (Q.1^2 + Q.2^2) = 1 →
  α = Real.arccos (3/5) →
  α + 3 * Real.pi / 4 = Real.arccos Q.1 →
  Q.1 = -7 * Real.sqrt 2 / 10 :=
by sorry

end NUMINAMATH_CALUDE_x_coordinate_of_Q_l1860_186060


namespace NUMINAMATH_CALUDE_multiply_333_by_111_l1860_186003

theorem multiply_333_by_111 : 333 * 111 = 36963 := by
  sorry

end NUMINAMATH_CALUDE_multiply_333_by_111_l1860_186003


namespace NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l1860_186093

theorem smallest_integer_with_given_remainders :
  ∃ (a : ℕ), a > 0 ∧ a % 8 = 6 ∧ a % 9 = 5 ∧
  ∀ (b : ℕ), b > 0 → b % 8 = 6 → b % 9 = 5 → a ≤ b :=
by
  use 14
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_given_remainders_l1860_186093


namespace NUMINAMATH_CALUDE_river_width_is_8km_l1860_186053

/-- Represents the boat's journey across the river -/
structure RiverCrossing where
  boat_speed : ℝ
  current_speed : ℝ
  crossing_time : ℝ

/-- Calculates the width of the river based on the given conditions -/
def river_width (rc : RiverCrossing) : ℝ :=
  rc.boat_speed * rc.crossing_time

/-- Theorem stating that the width of the river is 8 km under the given conditions -/
theorem river_width_is_8km (rc : RiverCrossing) 
  (h1 : rc.boat_speed = 4)
  (h2 : rc.current_speed = 3)
  (h3 : rc.crossing_time = 2) : 
  river_width rc = 8 := by
  sorry

end NUMINAMATH_CALUDE_river_width_is_8km_l1860_186053


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1860_186041

theorem simplify_fraction_product : (125 : ℚ) / 5000 * 40 = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1860_186041


namespace NUMINAMATH_CALUDE_man_work_days_l1860_186091

/-- Proves that if a woman can do a piece of work in 40 days and a man is 25% more efficient,
    then the man can do the same piece of work in 32 days. -/
theorem man_work_days (woman_days : ℕ) (man_efficiency : ℚ) :
  woman_days = 40 →
  man_efficiency = 1.25 →
  (woman_days : ℚ) / man_efficiency = 32 :=
by sorry

end NUMINAMATH_CALUDE_man_work_days_l1860_186091


namespace NUMINAMATH_CALUDE_product_of_numbers_l1860_186098

theorem product_of_numbers (x y : ℝ) : 
  |x - y| = 12 → x^2 + y^2 = 245 → x * y = 50.30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1860_186098


namespace NUMINAMATH_CALUDE_school_fire_problem_l1860_186096

/-- Represents the initial state and changes in a school after a fire incident -/
structure SchoolState where
  initialClassCount : ℕ
  initialStudentsPerClass : ℕ
  firstUnusableClasses : ℕ
  firstAddedStudents : ℕ
  secondUnusableClasses : ℕ
  secondAddedStudents : ℕ

/-- Calculates the total number of students after the changes -/
def totalStudentsAfterChanges (s : SchoolState) : ℕ :=
  let remainingClasses := s.initialClassCount - s.firstUnusableClasses - s.secondUnusableClasses
  remainingClasses * (s.initialStudentsPerClass + s.firstAddedStudents + s.secondAddedStudents)

/-- Theorem stating that the initial number of students in the school was 900 -/
theorem school_fire_problem (s : SchoolState) 
  (h1 : s.firstUnusableClasses = 6)
  (h2 : s.firstAddedStudents = 5)
  (h3 : s.secondUnusableClasses = 10)
  (h4 : s.secondAddedStudents = 15)
  (h5 : totalStudentsAfterChanges s = s.initialClassCount * s.initialStudentsPerClass) :
  s.initialClassCount * s.initialStudentsPerClass = 900 := by
  sorry

end NUMINAMATH_CALUDE_school_fire_problem_l1860_186096


namespace NUMINAMATH_CALUDE_number_ratio_l1860_186072

theorem number_ratio (A B C : ℚ) (k : ℤ) (h1 : A = 2 * B) (h2 : A = k * C)
  (h3 : (A + B + C) / 3 = 88) (h4 : A - C = 96) : A / C = 15 / 7 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l1860_186072


namespace NUMINAMATH_CALUDE_average_weight_of_three_l1860_186070

/-- Given the weights of three people with specific relationships, prove their average weight. -/
theorem average_weight_of_three (ishmael ponce jalen : ℝ) : 
  ishmael = ponce + 20 →
  ponce = jalen - 10 →
  jalen = 160 →
  (ishmael + ponce + jalen) / 3 = 160 := by
sorry

end NUMINAMATH_CALUDE_average_weight_of_three_l1860_186070


namespace NUMINAMATH_CALUDE_phone_storage_theorem_l1860_186056

/-- Calculates the maximum number of songs that can be stored on a phone given the total storage, used storage, and size of each song. -/
def max_songs (total_storage : ℕ) (used_storage : ℕ) (song_size : ℕ) : ℕ :=
  ((total_storage - used_storage) * 1000) / song_size

/-- Theorem stating that given a phone with 16 GB total storage, 4 GB already used, and songs of 30 MB each, the maximum number of additional songs that can be stored is 400. -/
theorem phone_storage_theorem :
  max_songs 16 4 30 = 400 := by
  sorry

end NUMINAMATH_CALUDE_phone_storage_theorem_l1860_186056


namespace NUMINAMATH_CALUDE_equal_selection_probabilities_l1860_186010

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Calculate the probability of selection for a given sampling method -/
def probability_of_selection (method : SamplingMethod) (population_size : ℕ) (sample_size : ℕ) : ℚ :=
  match method with
  | SamplingMethod.SimpleRandom => sample_size / population_size
  | SamplingMethod.Systematic => sample_size / population_size
  | SamplingMethod.Stratified => sample_size / population_size

theorem equal_selection_probabilities (population_size : ℕ) (sample_size : ℕ)
    (h1 : population_size = 50)
    (h2 : sample_size = 10) :
    (probability_of_selection SamplingMethod.SimpleRandom population_size sample_size =
     probability_of_selection SamplingMethod.Systematic population_size sample_size) ∧
    (probability_of_selection SamplingMethod.SimpleRandom population_size sample_size =
     probability_of_selection SamplingMethod.Stratified population_size sample_size) ∧
    (probability_of_selection SamplingMethod.SimpleRandom population_size sample_size = 1/5) :=
  sorry

end NUMINAMATH_CALUDE_equal_selection_probabilities_l1860_186010


namespace NUMINAMATH_CALUDE_solution_set_is_open_ray_l1860_186014

def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f x > 2 * x + 4}

theorem solution_set_is_open_ray
  (f : ℝ → ℝ)
  (h1 : Differentiable ℝ f)
  (h2 : ∀ x, deriv f x > 2)
  (h3 : f (-1) = 2) :
  solution_set f = Set.Ioi (-1) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_is_open_ray_l1860_186014


namespace NUMINAMATH_CALUDE_birthday_gifts_l1860_186065

theorem birthday_gifts (gifts_12th : ℕ) (fewer_gifts : ℕ) : 
  gifts_12th = 20 → fewer_gifts = 8 → 
  gifts_12th + (gifts_12th - fewer_gifts) = 32 := by
  sorry

end NUMINAMATH_CALUDE_birthday_gifts_l1860_186065


namespace NUMINAMATH_CALUDE_sum_of_squares_greater_than_ten_l1860_186035

theorem sum_of_squares_greater_than_ten (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁₂ : |x₁ - x₂| > 1) (h₁₃ : |x₁ - x₃| > 1) (h₁₄ : |x₁ - x₄| > 1) (h₁₅ : |x₁ - x₅| > 1)
  (h₂₃ : |x₂ - x₃| > 1) (h₂₄ : |x₂ - x₄| > 1) (h₂₅ : |x₂ - x₅| > 1)
  (h₃₄ : |x₃ - x₄| > 1) (h₃₅ : |x₃ - x₅| > 1)
  (h₄₅ : |x₄ - x₅| > 1) :
  x₁^2 + x₂^2 + x₃^2 + x₄^2 + x₅^2 > 10 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_greater_than_ten_l1860_186035


namespace NUMINAMATH_CALUDE_units_digit_of_smallest_n_with_2016_digits_l1860_186061

theorem units_digit_of_smallest_n_with_2016_digits : ∃ n : ℕ,
  (∀ m : ℕ, 7 * m < 10^2015 → m < n) ∧
  7 * n ≥ 10^2015 ∧
  7 * n < 10^2016 ∧
  n % 10 = 6 :=
sorry

end NUMINAMATH_CALUDE_units_digit_of_smallest_n_with_2016_digits_l1860_186061


namespace NUMINAMATH_CALUDE_intersection_centroids_exist_l1860_186081

/-- Represents a point on the grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a line on the grid -/
inductive GridLine
  | Horizontal (y : Int)
  | Vertical (x : Int)

/-- The grid size -/
def gridSize : Nat := 4030

/-- The number of selected lines in each direction -/
def selectedLines : Nat := 2017

/-- Checks if a point is within the grid bounds -/
def isWithinGrid (p : GridPoint) : Prop :=
  -gridSize / 2 ≤ p.x ∧ p.x ≤ gridSize / 2 ∧
  -gridSize / 2 ≤ p.y ∧ p.y ≤ gridSize / 2

/-- Checks if a point is an intersection of selected lines -/
def isIntersection (p : GridPoint) (horizontalLines : List Int) (verticalLines : List Int) : Prop :=
  p.y ∈ horizontalLines ∧ p.x ∈ verticalLines

/-- Calculates the centroid of a triangle -/
def centroid (a b c : GridPoint) : GridPoint :=
  { x := (a.x + b.x + c.x) / 3
  , y := (a.y + b.y + c.y) / 3 }

/-- The main theorem -/
theorem intersection_centroids_exist 
  (horizontalLines : List Int) 
  (verticalLines : List Int) 
  (h1 : horizontalLines.length = selectedLines)
  (h2 : verticalLines.length = selectedLines)
  (h3 : ∀ y ∈ horizontalLines, -gridSize / 2 ≤ y ∧ y ≤ gridSize / 2)
  (h4 : ∀ x ∈ verticalLines, -gridSize / 2 ≤ x ∧ x ≤ gridSize / 2) :
  ∃ (a b c d e f : GridPoint),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    isWithinGrid a ∧ isWithinGrid b ∧ isWithinGrid c ∧
    isWithinGrid d ∧ isWithinGrid e ∧ isWithinGrid f ∧
    isIntersection a horizontalLines verticalLines ∧
    isIntersection b horizontalLines verticalLines ∧
    isIntersection c horizontalLines verticalLines ∧
    isIntersection d horizontalLines verticalLines ∧
    isIntersection e horizontalLines verticalLines ∧
    isIntersection f horizontalLines verticalLines ∧
    centroid a b c = { x := 0, y := 0 } ∧
    centroid d e f = { x := 0, y := 0 } :=
  by sorry


end NUMINAMATH_CALUDE_intersection_centroids_exist_l1860_186081


namespace NUMINAMATH_CALUDE_triangle_abc_is_right_triangle_l1860_186040

theorem triangle_abc_is_right_triangle (AB AC BC : ℝ) 
  (h1 : AB = 1) (h2 : AC = 2) (h3 : BC = Real.sqrt 5) : 
  AB ^ 2 + AC ^ 2 = BC ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_is_right_triangle_l1860_186040


namespace NUMINAMATH_CALUDE_grayson_speed_l1860_186083

/-- Grayson's motorboat trip -/
structure GraysonTrip where
  speed1 : ℝ  -- Speed during the first hour
  time1 : ℝ   -- Time of the first part (1 hour)
  speed2 : ℝ  -- Speed during the second part (20 mph)
  time2 : ℝ   -- Time of the second part (0.5 hours)

/-- Rudy's rowboat trip -/
structure RudyTrip where
  speed : ℝ   -- Rudy's speed (10 mph)
  time : ℝ    -- Rudy's travel time (3 hours)

/-- The main theorem -/
theorem grayson_speed (g : GraysonTrip) (r : RudyTrip) 
  (h1 : g.time1 = 1)
  (h2 : g.time2 = 0.5)
  (h3 : g.speed2 = 20)
  (h4 : r.speed = 10)
  (h5 : r.time = 3)
  (h6 : g.speed1 * g.time1 + g.speed2 * g.time2 = r.speed * r.time + 5) :
  g.speed1 = 25 := by
  sorry

end NUMINAMATH_CALUDE_grayson_speed_l1860_186083


namespace NUMINAMATH_CALUDE_point_N_coordinates_l1860_186034

def M : ℝ × ℝ := (5, -6)
def a : ℝ × ℝ := (1, -2)

theorem point_N_coordinates : 
  ∀ N : ℝ × ℝ, 
  (N.1 - M.1, N.2 - M.2) = (-3 * a.1, -3 * a.2) → 
  N = (2, 0) := by sorry

end NUMINAMATH_CALUDE_point_N_coordinates_l1860_186034


namespace NUMINAMATH_CALUDE_square_division_l1860_186028

theorem square_division (a n : ℕ) : 
  a > 0 → 
  n > 1 → 
  a^2 = 88 + n^2 → 
  (a = 13 ∧ n = 9) ∨ (a = 23 ∧ n = 21) :=
by sorry

end NUMINAMATH_CALUDE_square_division_l1860_186028


namespace NUMINAMATH_CALUDE_square_carpet_side_length_l1860_186058

theorem square_carpet_side_length 
  (floor_length : ℝ) 
  (floor_width : ℝ) 
  (uncovered_area : ℝ) 
  (h1 : floor_length = 10)
  (h2 : floor_width = 8)
  (h3 : uncovered_area = 64)
  : ∃ (side_length : ℝ), 
    side_length^2 = floor_length * floor_width - uncovered_area ∧ 
    side_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_carpet_side_length_l1860_186058


namespace NUMINAMATH_CALUDE_points_on_line_equidistant_l1860_186089

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the line 4x + 7y = 28 -/
def onLine (p : Point) : Prop :=
  4 * p.x + 7 * p.y = 28

/-- Defines the condition of being equidistant from coordinate axes -/
def equidistant (p : Point) : Prop :=
  |p.x| = |p.y|

/-- Defines the condition of being in quadrant I -/
def inQuadrantI (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Defines the condition of being in quadrant II -/
def inQuadrantII (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Defines the condition of being in quadrant III -/
def inQuadrantIII (p : Point) : Prop :=
  p.x < 0 ∧ p.y < 0

/-- Defines the condition of being in quadrant IV -/
def inQuadrantIV (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

theorem points_on_line_equidistant :
  ∀ p : Point, onLine p ∧ equidistant p →
    (inQuadrantI p ∨ inQuadrantII p) ∧
    ¬(inQuadrantIII p ∨ inQuadrantIV p) :=
by sorry

end NUMINAMATH_CALUDE_points_on_line_equidistant_l1860_186089


namespace NUMINAMATH_CALUDE_polynomial_arrangement_l1860_186036

-- Define the polynomial as a function
def polynomial (x y : ℝ) : ℝ := 2 * x^3 * y - 4 * y^2 + 5 * x^2

-- Define the arranged polynomial as a function
def arranged_polynomial (x y : ℝ) : ℝ := 5 * x^2 + 2 * x^3 * y - 4 * y^2

-- Theorem stating that the arranged polynomial is equal to the original polynomial
theorem polynomial_arrangement (x y : ℝ) : 
  polynomial x y = arranged_polynomial x y := by
  sorry

end NUMINAMATH_CALUDE_polynomial_arrangement_l1860_186036


namespace NUMINAMATH_CALUDE_all_cars_meet_time_prove_all_cars_meet_time_l1860_186092

/-- Represents a car on a circular track -/
structure Car where
  speed : ℝ
  direction : Bool -- true for clockwise, false for counterclockwise

/-- Represents the race scenario -/
structure RaceScenario where
  track_length : ℝ
  car_a : Car
  car_b : Car
  car_c : Car
  first_ac_meet : ℝ
  first_ab_meet : ℝ

/-- Theorem stating when all three cars meet for the first time -/
theorem all_cars_meet_time (race : RaceScenario) : ℝ :=
  let first_ac_meet := race.first_ac_meet
  let first_ab_meet := race.first_ab_meet
  371

#check all_cars_meet_time

/-- Main theorem proving the time when all three cars meet -/
theorem prove_all_cars_meet_time (race : RaceScenario) 
  (h1 : race.car_a.direction = true)
  (h2 : race.car_b.direction = true)
  (h3 : race.car_c.direction = false)
  (h4 : race.car_a.speed ≠ race.car_b.speed)
  (h5 : race.car_a.speed ≠ race.car_c.speed)
  (h6 : race.car_b.speed ≠ race.car_c.speed)
  (h7 : race.first_ac_meet = 7)
  (h8 : race.first_ab_meet = 53)
  : all_cars_meet_time race = 371 := by
  sorry

#check prove_all_cars_meet_time

end NUMINAMATH_CALUDE_all_cars_meet_time_prove_all_cars_meet_time_l1860_186092


namespace NUMINAMATH_CALUDE_ring_arrangements_count_l1860_186022

/-- The number of ways to arrange 6 rings out of 10 distinguishable rings on 4 fingers. -/
def ring_arrangements : ℕ :=
  (Nat.choose 10 6) * (Nat.factorial 6) * (Nat.choose 9 3)

/-- The correct number of arrangements is 12672000. -/
theorem ring_arrangements_count : ring_arrangements = 12672000 := by
  sorry

end NUMINAMATH_CALUDE_ring_arrangements_count_l1860_186022


namespace NUMINAMATH_CALUDE_point_on_line_l1860_186073

/-- A point (x, 3) lies on the straight line joining (1, 5) and (5, -3) if and only if x = 2 -/
theorem point_on_line (x : ℝ) : 
  (3 - 5) / (x - 1) = (-3 - 5) / (5 - 1) ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l1860_186073


namespace NUMINAMATH_CALUDE_trail_mix_portions_l1860_186049

theorem trail_mix_portions (nuts dried_fruit chocolate coconut : ℕ) 
  (h1 : nuts = 16) (h2 : dried_fruit = 6) (h3 : chocolate = 8) (h4 : coconut = 4) :
  Nat.gcd nuts (Nat.gcd dried_fruit (Nat.gcd chocolate coconut)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_portions_l1860_186049


namespace NUMINAMATH_CALUDE_two_numbers_between_4_and_16_l1860_186005

theorem two_numbers_between_4_and_16 :
  ∃ (a b : ℝ), 
    4 < a ∧ a < b ∧ b < 16 ∧
    (b - a = a - 4) ∧
    (b * b = a * 16) ∧
    a + b = 20 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_between_4_and_16_l1860_186005


namespace NUMINAMATH_CALUDE_sum_of_repeated_digit_numbers_theorem_l1860_186078

theorem sum_of_repeated_digit_numbers_theorem :
  ∃ (a b c : ℕ),
    (∃ (d : ℕ), a = d * 11111 ∧ d < 10) ∧
    (∃ (e : ℕ), b = e * 1111 ∧ e < 10) ∧
    (∃ (f : ℕ), c = f * 111 ∧ f < 10) ∧
    (10000 ≤ a + b + c ∧ a + b + c < 100000) ∧
    (∃ (v w x y z : ℕ),
      v < 10 ∧ w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧
      v ≠ w ∧ v ≠ x ∧ v ≠ y ∧ v ≠ z ∧
      w ≠ x ∧ w ≠ y ∧ w ≠ z ∧
      x ≠ y ∧ x ≠ z ∧
      y ≠ z ∧
      a + b + c = v * 10000 + w * 1000 + x * 100 + y * 10 + z) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeated_digit_numbers_theorem_l1860_186078


namespace NUMINAMATH_CALUDE_trees_cut_l1860_186080

theorem trees_cut (original : ℕ) (died : ℕ) (left : ℕ) (cut : ℕ) : 
  original = 86 → died = 15 → left = 48 → cut = original - died - left → cut = 23 := by
  sorry

end NUMINAMATH_CALUDE_trees_cut_l1860_186080


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_five_l1860_186018

theorem complex_magnitude_equals_five (m : ℝ) (hm : m > 0) :
  Complex.abs (Complex.mk (-1) (2 * m)) = 5 ↔ m = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_five_l1860_186018


namespace NUMINAMATH_CALUDE_food_combo_discount_percentage_l1860_186011

/-- Calculates the discount percentage on food combos during a special offer. -/
theorem food_combo_discount_percentage
  (evening_ticket_cost : ℚ)
  (food_combo_cost : ℚ)
  (ticket_discount_percent : ℚ)
  (total_savings : ℚ)
  (h1 : evening_ticket_cost = 10)
  (h2 : food_combo_cost = 10)
  (h3 : ticket_discount_percent = 20)
  (h4 : total_savings = 7) :
  (total_savings - ticket_discount_percent / 100 * evening_ticket_cost) / food_combo_cost * 100 = 50 := by
sorry

end NUMINAMATH_CALUDE_food_combo_discount_percentage_l1860_186011


namespace NUMINAMATH_CALUDE_vector_sum_equals_result_l1860_186048

def vector_a : ℝ × ℝ := (0, -1)
def vector_b : ℝ × ℝ := (3, 2)

theorem vector_sum_equals_result : 2 • vector_a + vector_b = (3, 0) := by sorry

end NUMINAMATH_CALUDE_vector_sum_equals_result_l1860_186048


namespace NUMINAMATH_CALUDE_jinas_mascots_l1860_186042

/-- The number of mascots Jina has -/
def total_mascots (x y z : ℕ) : ℕ := x + y + z

/-- The theorem stating the total number of Jina's mascots -/
theorem jinas_mascots :
  ∃ (x y z : ℕ),
    y = 3 * x ∧
    z = 2 * y ∧
    (x + (5/2 : ℚ) * y) / y = 3/7 ∧
    total_mascots x y z = 60 := by
  sorry


end NUMINAMATH_CALUDE_jinas_mascots_l1860_186042


namespace NUMINAMATH_CALUDE_eagles_score_is_24_l1860_186064

/-- The combined score of both teams -/
def total_score : ℕ := 56

/-- The margin by which the Falcons won -/
def winning_margin : ℕ := 8

/-- The score of the Eagles -/
def eagles_score : ℕ := total_score / 2 - winning_margin / 2

theorem eagles_score_is_24 : eagles_score = 24 := by
  sorry

end NUMINAMATH_CALUDE_eagles_score_is_24_l1860_186064


namespace NUMINAMATH_CALUDE_max_colored_pages_l1860_186082

/-- The cost in cents to print a colored page -/
def cost_per_page : ℕ := 4

/-- The budget in dollars -/
def budget : ℕ := 30

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The maximum number of colored pages that can be printed -/
def max_pages : ℕ := (budget * cents_per_dollar) / cost_per_page

theorem max_colored_pages : max_pages = 750 := by
  sorry

end NUMINAMATH_CALUDE_max_colored_pages_l1860_186082


namespace NUMINAMATH_CALUDE_asymptote_sum_l1860_186077

theorem asymptote_sum (A B C : ℤ) : 
  (∀ x : ℝ, x^3 + A*x^2 + B*x + C = 0 ↔ x = -3 ∨ x = 0 ∨ x = 2) → 
  A + B + C = -5 := by
  sorry

end NUMINAMATH_CALUDE_asymptote_sum_l1860_186077


namespace NUMINAMATH_CALUDE_grouping_schemes_l1860_186008

theorem grouping_schemes (drivers : Finset α) (ticket_sellers : Finset β) :
  (drivers.card = 4) → (ticket_sellers.card = 4) →
  (Finset.product drivers ticket_sellers).card = 24 := by
sorry

end NUMINAMATH_CALUDE_grouping_schemes_l1860_186008


namespace NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l1860_186007

theorem solution_set_reciprocal_inequality (x : ℝ) :
  1 / x ≤ 1 ↔ x ∈ Set.Iic 0 ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_CALUDE_solution_set_reciprocal_inequality_l1860_186007


namespace NUMINAMATH_CALUDE_max_value_theorem_l1860_186067

/-- The maximum value of ab/(a+b) + ac/(a+c) + bc/(b+c) given the conditions -/
theorem max_value_theorem (a b c : ℝ) 
  (h_nonneg_a : 0 ≤ a) (h_nonneg_b : 0 ≤ b) (h_nonneg_c : 0 ≤ c)
  (h_sum : a + b + c = 3)
  (h_product : a * b * c = 1) :
  (a * b / (a + b) + a * c / (a + c) + b * c / (b + c)) ≤ 3 / 2 ∧ 
  ∃ a b c, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 ∧ a * b * c = 1 ∧
    a * b / (a + b) + a * c / (a + c) + b * c / (b + c) = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1860_186067


namespace NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l1860_186029

theorem sum_of_coefficients_equals_one 
  (a a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x : ℝ, (2*x - 3)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) : 
  a + a₁ + a₂ + a₃ + a₄ = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_equals_one_l1860_186029


namespace NUMINAMATH_CALUDE_particle_diameter_scientific_notation_l1860_186055

/-- Converts a decimal number to scientific notation -/
def to_scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem particle_diameter_scientific_notation :
  to_scientific_notation 0.00000021 = (2.1, -7) :=
sorry

end NUMINAMATH_CALUDE_particle_diameter_scientific_notation_l1860_186055


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l1860_186000

/-- An arithmetic sequence -/
def arithmetic_sequence : ℕ → ℝ := sorry

/-- Sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The value of n that maximizes S n -/
def n_max : ℕ := sorry

theorem arithmetic_sequence_max_sum :
  (S 16 > 0) → (S 17 < 0) → n_max = 8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_sum_l1860_186000


namespace NUMINAMATH_CALUDE_horner_method_eval_l1860_186079

def f (x : ℝ) : ℝ := 12 + 35*x - 8*x^2 + 79*x^3 + 6*x^4 + 5*x^5 + 3*x^6

theorem horner_method_eval :
  f (-4) = 220 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_eval_l1860_186079


namespace NUMINAMATH_CALUDE_product_of_distinct_roots_l1860_186044

theorem product_of_distinct_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_distinct_roots_l1860_186044
