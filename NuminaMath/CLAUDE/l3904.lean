import Mathlib

namespace NUMINAMATH_CALUDE_units_digit_of_n_l3904_390469

/-- Given two natural numbers m and n, returns true if m has a units digit of 3 -/
def has_units_digit_3 (m : ℕ) : Prop :=
  m % 10 = 3

/-- Given a natural number n, returns its units digit -/
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 31^5) (h2 : has_units_digit_3 m) :
  units_digit n = 7 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3904_390469


namespace NUMINAMATH_CALUDE_sugar_purchase_efficiency_l3904_390435

/-- Proves that Xiao Li's method of buying sugar is more cost-effective than Xiao Wang's --/
theorem sugar_purchase_efficiency
  (n : ℕ) (a : ℕ → ℝ)
  (h_n : n > 1)
  (h_a : ∀ i, i ∈ Finset.range n → a i > 0) :
  (Finset.sum (Finset.range n) a) / n ≥ n / (Finset.sum (Finset.range n) (λ i => 1 / a i)) :=
by sorry

end NUMINAMATH_CALUDE_sugar_purchase_efficiency_l3904_390435


namespace NUMINAMATH_CALUDE_round_201949_to_two_sig_figs_l3904_390495

/-- Rounds a number to a specified number of significant figures in scientific notation -/
def roundToSignificantFigures (x : ℝ) (sigFigs : ℕ) : ℝ := sorry

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

theorem round_201949_to_two_sig_figs :
  let number : ℝ := 201949
  let rounded := roundToSignificantFigures number 2
  ∃ (sn : ScientificNotation), 
    sn.coefficient = 2.0 ∧ 
    sn.exponent = 5 ∧ 
    rounded = sn.coefficient * (10 : ℝ) ^ sn.exponent :=
sorry

end NUMINAMATH_CALUDE_round_201949_to_two_sig_figs_l3904_390495


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3904_390400

/-- A hyperbola with center at the origin, focus at (3,0), and a line passing through
    the focus intersecting the hyperbola at two points whose midpoint is (-12,-15) -/
structure Hyperbola where
  /-- The equation of the hyperbola in the form x²/a² - y²/b² = 1 -/
  equation : ℝ → ℝ → Prop
  /-- The center of the hyperbola is at the origin -/
  center_at_origin : equation 0 0
  /-- One focus of the hyperbola is at (3,0) -/
  focus_at_3_0 : ∃ (x y : ℝ), equation x y ∧ (x - 3)^2 + y^2 = (x + 3)^2 + y^2
  /-- There exists a line passing through (3,0) that intersects the hyperbola at two points -/
  intersecting_line : ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    equation x₁ y₁ ∧ equation x₂ y₂ ∧ 
    (y₁ - 0) / (x₁ - 3) = (y₂ - 0) / (x₂ - 3)
  /-- The midpoint of the two intersection points is (-12,-15) -/
  midpoint : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    equation x₁ y₁ ∧ equation x₂ y₂ ∧
    (x₁ + x₂) / 2 = -12 ∧ (y₁ + y₂) / 2 = -15

/-- The equation of the hyperbola is x²/4 - y²/5 = 1 -/
theorem hyperbola_equation (h : Hyperbola) : 
  h.equation = fun x y => x^2 / 4 - y^2 / 5 = 1 := by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3904_390400


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3904_390476

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_parallel (l m : Line) (α : Plane) :
  l ≠ m →  -- l and m are different lines
  perpendicular l α →
  perpendicular m α →
  parallel l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_l3904_390476


namespace NUMINAMATH_CALUDE_total_flowering_bulbs_l3904_390428

/-- Calculates the total number of small flowering bulbs that can be purchased given the costs and constraints. -/
theorem total_flowering_bulbs 
  (crocus_cost : ℚ)
  (daffodil_cost : ℚ)
  (total_budget : ℚ)
  (crocus_count : ℕ)
  (h1 : crocus_cost = 35/100)
  (h2 : daffodil_cost = 65/100)
  (h3 : total_budget = 2915/100)
  (h4 : crocus_count = 22) :
  ∃ (daffodil_count : ℕ), 
    (crocus_count : ℚ) * crocus_cost + (daffodil_count : ℚ) * daffodil_cost ≤ total_budget ∧
    crocus_count + daffodil_count = 55 :=
by sorry

end NUMINAMATH_CALUDE_total_flowering_bulbs_l3904_390428


namespace NUMINAMATH_CALUDE_faulty_clock_correct_time_fraction_l3904_390466

/-- Represents a faulty digital clock that displays '5' instead of '2' over a 24-hour period -/
structure FaultyClock where
  /-- The number of hours in a day -/
  hours_per_day : ℕ
  /-- The number of minutes in an hour -/
  minutes_per_hour : ℕ
  /-- The number of hours affected by the fault -/
  faulty_hours : ℕ
  /-- The number of minutes per hour affected by the fault -/
  faulty_minutes : ℕ

/-- The fraction of the day a faulty clock displays the correct time -/
def correct_time_fraction (c : FaultyClock) : ℚ :=
  ((c.hours_per_day - c.faulty_hours) / c.hours_per_day) *
  ((c.minutes_per_hour - c.faulty_minutes) / c.minutes_per_hour)

/-- Theorem stating that the fraction of the day the faulty clock displays the correct time is 9/16 -/
theorem faulty_clock_correct_time_fraction :
  ∃ (c : FaultyClock), c.hours_per_day = 24 ∧ c.minutes_per_hour = 60 ∧
  c.faulty_hours = 6 ∧ c.faulty_minutes = 15 ∧
  correct_time_fraction c = 9 / 16 :=
by
  sorry

end NUMINAMATH_CALUDE_faulty_clock_correct_time_fraction_l3904_390466


namespace NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3904_390444

/-- The sum of the coordinates of the midpoint of a line segment with endpoints (1, 4) and (7, 10) is 11. -/
theorem midpoint_coordinate_sum : 
  let x1 : ℝ := 1
  let y1 : ℝ := 4
  let x2 : ℝ := 7
  let y2 : ℝ := 10
  let midpoint_x : ℝ := (x1 + x2) / 2
  let midpoint_y : ℝ := (y1 + y2) / 2
  midpoint_x + midpoint_y = 11 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_coordinate_sum_l3904_390444


namespace NUMINAMATH_CALUDE_factors_of_2012_l3904_390460

theorem factors_of_2012 : Finset.card (Nat.divisors 2012) = 6 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_2012_l3904_390460


namespace NUMINAMATH_CALUDE_train_journey_properties_l3904_390445

/-- Represents the properties of a train's journey -/
structure TrainJourney where
  duration : Real
  hourly_distance : Real

/-- Defines the concept of constant speed -/
def constant_speed (journey : TrainJourney) : Prop :=
  ∀ t : Real, 0 < t → t ≤ journey.duration → 
    (t * journey.hourly_distance) / t = journey.hourly_distance

/-- Calculates the total distance traveled -/
def total_distance (journey : TrainJourney) : Real :=
  journey.duration * journey.hourly_distance

/-- Main theorem about the train's journey -/
theorem train_journey_properties (journey : TrainJourney) 
  (h1 : journey.duration = 5.5)
  (h2 : journey.hourly_distance = 100) : 
  constant_speed journey ∧ total_distance journey = 550 := by
  sorry

#check train_journey_properties

end NUMINAMATH_CALUDE_train_journey_properties_l3904_390445


namespace NUMINAMATH_CALUDE_unique_solution_iff_n_eleven_l3904_390416

/-- The equation x^2 - 3x + 5 = 0 has a unique solution in (ℤ_n, +, ·) if and only if n = 11 -/
theorem unique_solution_iff_n_eleven (n : ℕ) (hn : n ≥ 2) :
  (∃! x : ZMod n, x^2 - 3*x + 5 = 0) ↔ n = 11 := by sorry

end NUMINAMATH_CALUDE_unique_solution_iff_n_eleven_l3904_390416


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_shift_l3904_390491

theorem fixed_point_of_exponential_shift (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 4 + a^(x - 1)
  f 1 = 5 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_shift_l3904_390491


namespace NUMINAMATH_CALUDE_sector_arc_length_l3904_390467

/-- Given a sector with central angle π/3 and radius 3, its arc length is π. -/
theorem sector_arc_length (α : Real) (r : Real) (l : Real) : 
  α = π / 3 → r = 3 → l = r * α → l = π := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l3904_390467


namespace NUMINAMATH_CALUDE_maggie_earnings_l3904_390481

/-- The amount Maggie earns for each magazine subscription she sells -/
def earnings_per_subscription : ℝ := 5

/-- The number of subscriptions Maggie sold to her parents -/
def parents_subscriptions : ℕ := 4

/-- The number of subscriptions Maggie sold to her grandfather -/
def grandfather_subscriptions : ℕ := 1

/-- The number of subscriptions Maggie sold to the next-door neighbor -/
def neighbor_subscriptions : ℕ := 2

/-- The total amount Maggie earned from all subscriptions -/
def total_earnings : ℝ := 55

theorem maggie_earnings :
  earnings_per_subscription * (parents_subscriptions + grandfather_subscriptions + 
  neighbor_subscriptions + 2 * neighbor_subscriptions) = total_earnings :=
sorry

end NUMINAMATH_CALUDE_maggie_earnings_l3904_390481


namespace NUMINAMATH_CALUDE_parabola_intercepts_sum_l3904_390401

/-- Represents a parabola of the form x = 3y² - 9y + 5 --/
def Parabola (x y : ℝ) : Prop := x = 3 * y^2 - 9 * y + 5

/-- Theorem stating that for the given parabola, the sum of its x-intercept and y-intercepts is 8 --/
theorem parabola_intercepts_sum (a b c : ℝ) 
  (h_x_intercept : Parabola a 0)
  (h_y_intercept1 : Parabola 0 b)
  (h_y_intercept2 : Parabola 0 c)
  : a + b + c = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intercepts_sum_l3904_390401


namespace NUMINAMATH_CALUDE_candy_division_l3904_390492

theorem candy_division (total_candy : ℚ) (num_piles : ℕ) (piles_for_carlos : ℕ) :
  total_candy = 75 / 7 →
  num_piles = 5 →
  piles_for_carlos = 2 →
  piles_for_carlos * (total_candy / num_piles) = 30 / 7 := by
  sorry

end NUMINAMATH_CALUDE_candy_division_l3904_390492


namespace NUMINAMATH_CALUDE_pages_read_difference_l3904_390475

/-- Given a book with 270 pages, prove that reading 2/3 of it results in 90 more pages read than left to read. -/
theorem pages_read_difference (total_pages : ℕ) (fraction_read : ℚ) : 
  total_pages = 270 → 
  fraction_read = 2/3 →
  (fraction_read * total_pages : ℚ) - (total_pages - fraction_read * total_pages : ℚ) = 90 :=
by
  sorry

#check pages_read_difference

end NUMINAMATH_CALUDE_pages_read_difference_l3904_390475


namespace NUMINAMATH_CALUDE_burning_time_3x5_grid_l3904_390450

/-- Represents a rectangular grid of toothpicks -/
structure ToothpickGrid where
  rows : ℕ
  cols : ℕ
  toothpicks : ℕ

/-- Represents the burning properties of toothpicks -/
structure BurningProperties where
  burn_time : ℕ  -- Time for one toothpick to burn completely
  spread_speed : ℝ  -- Speed at which fire spreads (assumed constant)

/-- Calculates the maximum burning time for a toothpick grid -/
def max_burning_time (grid : ToothpickGrid) (props : BurningProperties) : ℕ :=
  sorry  -- The actual calculation would go here

/-- Theorem stating the maximum burning time for the specific grid -/
theorem burning_time_3x5_grid :
  let grid := ToothpickGrid.mk 3 5 38
  let props := BurningProperties.mk 10 1
  max_burning_time grid props = 65 :=
sorry

#check burning_time_3x5_grid

end NUMINAMATH_CALUDE_burning_time_3x5_grid_l3904_390450


namespace NUMINAMATH_CALUDE_english_spanish_difference_l3904_390443

/-- The number of hours Ryan spends learning English -/
def hours_english : ℕ := 7

/-- The number of hours Ryan spends learning Chinese -/
def hours_chinese : ℕ := 2

/-- The number of hours Ryan spends learning Spanish -/
def hours_spanish : ℕ := 4

/-- Theorem: Ryan spends 3 more hours on learning English than Spanish -/
theorem english_spanish_difference : hours_english - hours_spanish = 3 := by
  sorry

end NUMINAMATH_CALUDE_english_spanish_difference_l3904_390443


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l3904_390478

/-- Given a curve defined by 2x^2 - y = 0, prove that the midpoint of the line segment
    connecting (0, -1) and any point on the curve satisfies y = 4x^2 - 1/2 -/
theorem midpoint_trajectory (x₁ y₁ x y : ℝ) :
  (2 * x₁^2 = y₁) →  -- P(x₁, y₁) is on the curve
  (x = x₁ / 2) →     -- x-coordinate of midpoint
  (y = (y₁ - 1) / 2) -- y-coordinate of midpoint
  → y = 4 * x^2 - 1/2 := by
sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l3904_390478


namespace NUMINAMATH_CALUDE_divisor_pairs_count_l3904_390462

theorem divisor_pairs_count (n : ℕ) (h : n = 2^6 * 3^3) :
  (Finset.filter (fun p : ℕ × ℕ => p.1 * p.2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range (n+1)) (Finset.range (n+1)))).card = 28 :=
by sorry

end NUMINAMATH_CALUDE_divisor_pairs_count_l3904_390462


namespace NUMINAMATH_CALUDE_william_shared_three_marbles_l3904_390494

/-- The number of marbles William shared with Theresa -/
def marbles_shared (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

theorem william_shared_three_marbles :
  let initial := 10
  let remaining := 7
  marbles_shared initial remaining = 3 := by
  sorry

end NUMINAMATH_CALUDE_william_shared_three_marbles_l3904_390494


namespace NUMINAMATH_CALUDE_factorization_x2_4xy_4y2_l3904_390453

/-- Factorization of a polynomial x^2 - 4xy + 4y^2 --/
theorem factorization_x2_4xy_4y2 (x y : ℝ) :
  x^2 - 4*x*y + 4*y^2 = (x - 2*y)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_x2_4xy_4y2_l3904_390453


namespace NUMINAMATH_CALUDE_trig_identity_l3904_390405

theorem trig_identity (x y : ℝ) :
  (Real.sin x)^2 + (Real.sin (x + y + π/4))^2 - 
  2 * (Real.sin x) * (Real.sin (y + π/4)) * (Real.sin (x + y + π/4)) = 
  1 - (1/2) * (Real.sin y)^2 := by
sorry

end NUMINAMATH_CALUDE_trig_identity_l3904_390405


namespace NUMINAMATH_CALUDE_unique_solution_equation_l3904_390421

theorem unique_solution_equation :
  ∃! x : ℝ, x ≠ 2 ∧ x - 6 / (x - 2) = 4 - 6 / (x - 2) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_equation_l3904_390421


namespace NUMINAMATH_CALUDE_eric_erasers_friends_l3904_390472

theorem eric_erasers_friends (total_erasers : ℕ) (erasers_per_friend : ℕ) (h1 : total_erasers = 9306) (h2 : erasers_per_friend = 94) :
  total_erasers / erasers_per_friend = 99 := by
sorry

end NUMINAMATH_CALUDE_eric_erasers_friends_l3904_390472


namespace NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l3904_390430

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (q : ℝ) :
  is_geometric_sequence a q →
  (a 1 < 0 ∧ 0 < q ∧ q < 1) →
  (∀ n : ℕ, n > 0 → a (n + 1) > a n) ∧
  ¬(∀ n : ℕ, n > 0 → a (n + 1) > a n → (a 1 < 0 ∧ 0 < q ∧ q < 1)) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_increasing_condition_l3904_390430


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l3904_390471

theorem completing_square_equivalence (x : ℝ) : 
  (x^2 - 4*x + 2 = 0) ↔ ((x - 2)^2 = 2) := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l3904_390471


namespace NUMINAMATH_CALUDE_problem_statement_l3904_390455

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | x * (x - 4) * (x + 1) < 0}

def B (a : ℝ) : Set ℝ := {x | x^2 - 2*a*x + a^2 - 1 < 0}

theorem problem_statement :
  (∀ x : ℝ, x ∈ (A ∪ B 4) ↔ -1 < x ∧ x < 5) ∧
  (∀ a : ℝ, (∀ x : ℝ, x ∈ (U \ A) ↔ x ∈ (U \ B a)) ↔ 0 ≤ a ∧ a ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_problem_statement_l3904_390455


namespace NUMINAMATH_CALUDE_jeep_speed_problem_l3904_390477

theorem jeep_speed_problem (distance : ℝ) (original_time : ℝ) (new_time_factor : ℝ) :
  distance = 420 ∧ original_time = 7 ∧ new_time_factor = 3/2 →
  (distance / (new_time_factor * original_time)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_jeep_speed_problem_l3904_390477


namespace NUMINAMATH_CALUDE_symmetric_line_across_x_axis_l3904_390489

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Reflects a line across the x-axis -/
def reflectLineAcrossXAxis (l : Line2D) : Line2D :=
  { a := l.a, b := -l.b, c := l.c }

theorem symmetric_line_across_x_axis :
  let originalLine := Line2D.mk 2 (-3) 2
  let symmetricLine := Line2D.mk 2 3 2
  reflectLineAcrossXAxis originalLine = symmetricLine := by sorry

end NUMINAMATH_CALUDE_symmetric_line_across_x_axis_l3904_390489


namespace NUMINAMATH_CALUDE_james_container_capacity_l3904_390431

/-- Represents the capacity of different container types and their quantities --/
structure ContainerInventory where
  largeCaskCapacity : ℕ
  barrelCapacity : ℕ
  smallCaskCapacity : ℕ
  glassBottleCapacity : ℕ
  clayJugCapacity : ℕ
  barrelCount : ℕ
  largeCaskCount : ℕ
  smallCaskCount : ℕ
  glassBottleCount : ℕ
  clayJugCount : ℕ

/-- Calculates the total capacity of all containers --/
def totalCapacity (inv : ContainerInventory) : ℕ :=
  inv.barrelCapacity * inv.barrelCount +
  inv.largeCaskCapacity * inv.largeCaskCount +
  inv.smallCaskCapacity * inv.smallCaskCount +
  inv.glassBottleCapacity * inv.glassBottleCount +
  inv.clayJugCapacity * inv.clayJugCount

/-- Theorem stating that James' total container capacity is 318 gallons --/
theorem james_container_capacity :
  ∀ (inv : ContainerInventory),
    inv.largeCaskCapacity = 20 →
    inv.barrelCapacity = 2 * inv.largeCaskCapacity + 3 →
    inv.smallCaskCapacity = inv.largeCaskCapacity / 2 →
    inv.glassBottleCapacity = inv.smallCaskCapacity / 10 →
    inv.clayJugCapacity = 3 * inv.glassBottleCapacity →
    inv.barrelCount = 4 →
    inv.largeCaskCount = 3 →
    inv.smallCaskCount = 5 →
    inv.glassBottleCount = 12 →
    inv.clayJugCount = 8 →
    totalCapacity inv = 318 :=
by
  sorry


end NUMINAMATH_CALUDE_james_container_capacity_l3904_390431


namespace NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l3904_390496

/-- The surface area of a sphere that circumscribes a rectangular solid -/
theorem sphere_surface_area_from_rectangular_solid 
  (length width height : ℝ) 
  (h_length : length = 3) 
  (h_width : width = 2) 
  (h_height : height = 1) : 
  4 * Real.pi * ((length^2 + width^2 + height^2) / 4) = 14 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_from_rectangular_solid_l3904_390496


namespace NUMINAMATH_CALUDE_special_polynomial_derivative_theorem_l3904_390447

/-- A second-degree polynomial with roots in [-1, 1] and |f(x₀)| = 1 for some x₀ ∈ [-1, 1] -/
structure SpecialPolynomial where
  f : ℝ → ℝ
  degree_two : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  roots_in_interval : ∀ r, f r = 0 → r ∈ Set.Icc (-1 : ℝ) 1
  exists_unit_value : ∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, |f x₀| = 1

/-- The main theorem about special polynomials -/
theorem special_polynomial_derivative_theorem (p : SpecialPolynomial) :
  (∀ α ∈ Set.Icc (0 : ℝ) 1, ∃ ζ ∈ Set.Icc (-1 : ℝ) 1, |deriv p.f ζ| = α) ∧
  (¬∃ ζ ∈ Set.Icc (-1 : ℝ) 1, |deriv p.f ζ| > 1) :=
by sorry

end NUMINAMATH_CALUDE_special_polynomial_derivative_theorem_l3904_390447


namespace NUMINAMATH_CALUDE_feed_lasts_longer_when_selling_feed_lasts_shorter_when_buying_nils_has_300_geese_l3904_390479

/-- Represents the number of geese Nils currently has -/
def current_geese : ℕ := sorry

/-- Represents the number of days the feed lasts with the current number of geese -/
def current_feed_duration : ℕ := sorry

/-- Represents the amount of feed one goose consumes per day -/
def feed_per_goose_per_day : ℚ := sorry

/-- Represents the total amount of feed available -/
def total_feed : ℚ := sorry

/-- The feed lasts 20 days longer when 75 geese are sold -/
theorem feed_lasts_longer_when_selling : 
  total_feed / (feed_per_goose_per_day * (current_geese - 75)) = current_feed_duration + 20 := by sorry

/-- The feed lasts 15 days shorter when 100 geese are bought -/
theorem feed_lasts_shorter_when_buying : 
  total_feed / (feed_per_goose_per_day * (current_geese + 100)) = current_feed_duration - 15 := by sorry

/-- The main theorem proving that Nils has 300 geese -/
theorem nils_has_300_geese : current_geese = 300 := by sorry

end NUMINAMATH_CALUDE_feed_lasts_longer_when_selling_feed_lasts_shorter_when_buying_nils_has_300_geese_l3904_390479


namespace NUMINAMATH_CALUDE_ratio_problem_l3904_390480

theorem ratio_problem (a b c d : ℝ) 
  (h1 : b / a = 3)
  (h2 : c / b = 4)
  (h3 : d = 5 * b)
  : (a + b + d) / (b + c + d) = 19 / 30 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l3904_390480


namespace NUMINAMATH_CALUDE_work_completion_time_l3904_390422

/-- Represents the time it takes for A to complete the work alone -/
def time_A : ℝ := 15

/-- Represents the time it takes for B to complete the work alone -/
def time_B : ℝ := 27

/-- Represents the total amount of work -/
def total_work : ℝ := 1

/-- Represents the number of days A works before leaving -/
def days_A_worked : ℝ := 5

/-- Represents the number of days B works to complete the remaining work -/
def days_B_worked : ℝ := 18

theorem work_completion_time :
  (days_A_worked / time_A) + (days_B_worked / time_B) = total_work ∧
  time_A = total_work / ((total_work - (days_B_worked / time_B)) / days_A_worked) :=
sorry

end NUMINAMATH_CALUDE_work_completion_time_l3904_390422


namespace NUMINAMATH_CALUDE_technician_salary_l3904_390474

/-- Proves that the average salary of technicians is 12000 given the workshop conditions --/
theorem technician_salary (total_workers : ℕ) (technicians : ℕ) (avg_salary : ℕ) (non_tech_salary : ℕ) :
  total_workers = 21 →
  technicians = 7 →
  avg_salary = 8000 →
  non_tech_salary = 6000 →
  (avg_salary * total_workers = 12000 * technicians + non_tech_salary * (total_workers - technicians)) :=
by
  sorry

#check technician_salary

end NUMINAMATH_CALUDE_technician_salary_l3904_390474


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_times_14_l3904_390413

def B : Matrix (Fin 3) (Fin 3) ℝ := !![3, 1, 2; 0, 4, 1; 0, 0, 2]

theorem B_power_15_minus_3_times_14 :
  B^15 - 3 • (B^14) = !![0, 3, 1; 0, 4, 1; 0, 0, -2] := by
  sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_times_14_l3904_390413


namespace NUMINAMATH_CALUDE_counseling_rooms_count_l3904_390449

theorem counseling_rooms_count :
  ∃ (x : ℕ) (total_students : ℕ),
    (total_students = 20 * x + 32) ∧
    (total_students = 24 * (x - 1)) ∧
    (x = 14) := by
  sorry

end NUMINAMATH_CALUDE_counseling_rooms_count_l3904_390449


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3904_390485

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1)
  (h_sum : a * b + b * c + a * c = 1) :
  (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c)) ≥ (9 + 3 * Real.sqrt 3) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3904_390485


namespace NUMINAMATH_CALUDE_estate_value_l3904_390454

-- Define the estate and its components
def estate : ℝ := sorry
def older_child_share : ℝ := sorry
def younger_child_share : ℝ := sorry
def wife_share : ℝ := sorry
def charity_share : ℝ := 800

-- Define the conditions
axiom children_share : older_child_share + younger_child_share = 0.6 * estate
axiom children_ratio : older_child_share = (3/2) * younger_child_share
axiom wife_share_relation : wife_share = 4 * older_child_share
axiom total_distribution : estate = older_child_share + younger_child_share + wife_share + charity_share

-- Theorem to prove
theorem estate_value : estate = 1923 := by sorry

end NUMINAMATH_CALUDE_estate_value_l3904_390454


namespace NUMINAMATH_CALUDE_meal_prep_combinations_l3904_390458

def total_people : Nat := 6
def meal_preparers : Nat := 3

theorem meal_prep_combinations :
  Nat.choose total_people meal_preparers = 20 := by
  sorry

end NUMINAMATH_CALUDE_meal_prep_combinations_l3904_390458


namespace NUMINAMATH_CALUDE_absolute_value_equation_solutions_l3904_390484

theorem absolute_value_equation_solutions :
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
    (∀ x : ℝ, x ∈ s ↔ |x - 1| = |x - 2| + |x - 3| + |x - 4|) ∧
    (2 ∈ s ∧ 4 ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solutions_l3904_390484


namespace NUMINAMATH_CALUDE_friends_contribution_l3904_390486

/-- Represents the expenses of a group of friends -/
structure Expenses where
  num_friends : Nat
  total_amount : Rat

/-- Calculates the amount each friend should contribute -/
def calculate_contribution (e : Expenses) : Rat :=
  e.total_amount / e.num_friends

/-- Theorem: For 5 friends with total expenses of $61, each should contribute $12.20 -/
theorem friends_contribution :
  let e : Expenses := { num_friends := 5, total_amount := 61 }
  calculate_contribution e = 61 / 5 := by sorry

end NUMINAMATH_CALUDE_friends_contribution_l3904_390486


namespace NUMINAMATH_CALUDE_town_population_problem_l3904_390439

theorem town_population_problem (original_population : ℝ) : 
  (original_population * 1.15 * 0.87 = original_population - 50) → 
  original_population = 100000 := by
  sorry

end NUMINAMATH_CALUDE_town_population_problem_l3904_390439


namespace NUMINAMATH_CALUDE_largest_non_sum_30_composite_l3904_390415

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def is_sum_of_multiple_30_and_composite (n : ℕ) : Prop :=
  ∃ k m, k > 0 ∧ is_composite m ∧ n = 30 * k + m

theorem largest_non_sum_30_composite : 
  (∀ n > 93, is_sum_of_multiple_30_and_composite n) ∧
  ¬is_sum_of_multiple_30_and_composite 93 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_30_composite_l3904_390415


namespace NUMINAMATH_CALUDE_rhombus_diagonal_l3904_390426

/-- 
Given a rhombus with area 90 cm² and one diagonal of length 12 cm,
prove that the length of the other diagonal is 15 cm.
-/
theorem rhombus_diagonal (area : ℝ) (d1 : ℝ) (d2 : ℝ) : 
  area = 90 → d2 = 12 → area = (d1 * d2) / 2 → d1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_diagonal_l3904_390426


namespace NUMINAMATH_CALUDE_candidate_a_votes_l3904_390452

theorem candidate_a_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_a_percent : ℚ) : 
  total_votes = 560000 →
  invalid_percent = 15 / 100 →
  candidate_a_percent = 60 / 100 →
  ⌊(1 - invalid_percent) * candidate_a_percent * total_votes⌋ = 285600 :=
by sorry

end NUMINAMATH_CALUDE_candidate_a_votes_l3904_390452


namespace NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3904_390457

theorem sufficient_condition_for_inequality (x : ℝ) :
  1 < x ∧ x < 2 → (x + 1) / (x - 1) > 2 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_inequality_l3904_390457


namespace NUMINAMATH_CALUDE_max_xy_given_constraint_l3904_390423

theorem max_xy_given_constraint (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_constraint : x + 4 * y = 4) :
  ∀ x' y' : ℝ, 0 < x' → 0 < y' → x' + 4 * y' = 4 → x' * y' ≤ x * y ∧ x * y = 1 :=
sorry

end NUMINAMATH_CALUDE_max_xy_given_constraint_l3904_390423


namespace NUMINAMATH_CALUDE_lilliputian_matchboxes_in_gulliverian_l3904_390424

/-- Represents the scale factor between Lilliput and Gulliver's homeland -/
def scale_factor : ℝ := 12

/-- Calculates the volume of a matchbox given its dimensions -/
def matchbox_volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The number of Lilliputian matchboxes that fit into one Gulliverian matchbox is 1728 -/
theorem lilliputian_matchboxes_in_gulliverian (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0) :
  (matchbox_volume l w h) / (matchbox_volume (l / scale_factor) (w / scale_factor) (h / scale_factor)) = 1728 := by
  sorry

#check lilliputian_matchboxes_in_gulliverian

end NUMINAMATH_CALUDE_lilliputian_matchboxes_in_gulliverian_l3904_390424


namespace NUMINAMATH_CALUDE_exactly_one_survives_l3904_390402

theorem exactly_one_survives (p q : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1) :
  (p * (1 - q)) + ((1 - p) * q) = p + q - p * q := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_survives_l3904_390402


namespace NUMINAMATH_CALUDE_largest_angle_in_specific_pentagon_l3904_390425

/-- The measure of the largest angle in a pentagon with specific angle conditions -/
theorem largest_angle_in_specific_pentagon : 
  ∀ (A B C D E x : ℝ),
  -- Pentagon conditions
  A + B + C + D + E = 540 →
  -- Specific angle conditions
  A = 70 →
  B = 90 →
  C = D →
  E = 3 * x - 10 →
  C = x →
  -- Conclusion
  max A (max B (max C (max D E))) = 224 :=
by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_specific_pentagon_l3904_390425


namespace NUMINAMATH_CALUDE_abs_geq_ax_implies_a_in_range_l3904_390434

theorem abs_geq_ax_implies_a_in_range (a : ℝ) :
  (∀ x : ℝ, |x| ≥ a * x) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_geq_ax_implies_a_in_range_l3904_390434


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3904_390437

def M : Set ℝ := {x | x^2 + 2*x = 0}
def N : Set ℝ := {x | x^2 - 2*x = 0}

theorem union_of_M_and_N : M ∪ N = {-2, 0, 2} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3904_390437


namespace NUMINAMATH_CALUDE_consecutive_odd_product_ends_09_l3904_390429

theorem consecutive_odd_product_ends_09 (n : ℕ) (hn : n > 0) :
  ∃ k : ℕ, (10*n - 3) * (10*n - 1) * (10*n + 1) * (10*n + 3) = 100 * k + 9 :=
sorry

end NUMINAMATH_CALUDE_consecutive_odd_product_ends_09_l3904_390429


namespace NUMINAMATH_CALUDE_waiter_customers_l3904_390488

/-- Given a waiter with tables, each having a certain number of women and men,
    calculate the total number of customers. -/
theorem waiter_customers (tables women_per_table men_per_table : ℕ) :
  tables = 5 →
  women_per_table = 5 →
  men_per_table = 3 →
  tables * (women_per_table + men_per_table) = 40 := by
sorry

end NUMINAMATH_CALUDE_waiter_customers_l3904_390488


namespace NUMINAMATH_CALUDE_remainder_of_binary_div_eight_l3904_390417

/-- The binary number 101110110101₂ -/
def binary_number : Nat := 2981

/-- The divisor 8 -/
def divisor : Nat := 8

/-- Theorem stating that the remainder of the binary number divided by 8 is 5 -/
theorem remainder_of_binary_div_eight :
  binary_number % divisor = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_binary_div_eight_l3904_390417


namespace NUMINAMATH_CALUDE_radio_selling_price_l3904_390427

/-- Calculates the selling price of a radio given its purchase price, overhead expenses, and profit percentage. -/
def calculate_selling_price (purchase_price : ℚ) (overhead_expenses : ℚ) (profit_percentage : ℚ) : ℚ :=
  let total_cost := purchase_price + overhead_expenses
  let profit_amount := (profit_percentage / 100) * total_cost
  total_cost + profit_amount

/-- Theorem stating that the selling price of a radio with given parameters is 350 Rs. -/
theorem radio_selling_price :
  let purchase_price : ℚ := 225
  let overhead_expenses : ℚ := 15
  let profit_percentage : ℚ := 45833333333333314 / 1000000000000000
  calculate_selling_price purchase_price overhead_expenses profit_percentage = 350 := by
  sorry


end NUMINAMATH_CALUDE_radio_selling_price_l3904_390427


namespace NUMINAMATH_CALUDE_divisibility_problem_l3904_390461

theorem divisibility_problem (A B C : Nat) : 
  A < 10 → B < 10 → C < 10 →
  (7 * 1000000 + A * 100000 + 5 * 10000 + 1 * 1000 + B * 10 + 2) % 15 = 0 →
  (3 * 1000000 + 2 * 100000 + 6 * 10000 + A * 1000 + B * 100 + 4 * 10 + C) % 15 = 0 →
  C = 4 := by
sorry

end NUMINAMATH_CALUDE_divisibility_problem_l3904_390461


namespace NUMINAMATH_CALUDE_inscribed_square_area_l3904_390412

/-- A square inscribed in a semicircle with radius 1 -/
structure InscribedSquare where
  /-- The side length of the square -/
  side : ℝ
  /-- One side of the square is flush with the diameter of the semicircle -/
  flush_with_diameter : True
  /-- The square is inscribed in the semicircle -/
  inscribed : side^2 + (side/2)^2 = 1

/-- The area of an inscribed square is 4/5 -/
theorem inscribed_square_area (s : InscribedSquare) : s.side^2 = 4/5 := by
  sorry

#check inscribed_square_area

end NUMINAMATH_CALUDE_inscribed_square_area_l3904_390412


namespace NUMINAMATH_CALUDE_system_solution_l3904_390456

theorem system_solution :
  let s : Set (ℚ × ℚ) := {(1/2, 5), (1, 3), (3/2, 2), (5/2, 1)}
  ∀ x y : ℚ, (2*x + y + 2*x*y = 11 ∧ 2*x^2*y + x*y^2 = 15) ↔ (x, y) ∈ s := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3904_390456


namespace NUMINAMATH_CALUDE_bread_butter_price_ratio_l3904_390493

/-- Proves that the ratio of bread price to butter price is 1:2 given the problem conditions --/
theorem bread_butter_price_ratio : 
  ∀ (butter bread cheese tea : ℝ),
  butter + bread + cheese + tea = 21 →
  butter = 0.8 * cheese →
  tea = 2 * cheese →
  tea = 10 →
  bread / butter = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_bread_butter_price_ratio_l3904_390493


namespace NUMINAMATH_CALUDE_remaining_amount_proof_l3904_390404

def initial_amount : ℕ := 87
def spent_amount : ℕ := 64

theorem remaining_amount_proof :
  initial_amount - spent_amount = 23 :=
by sorry

end NUMINAMATH_CALUDE_remaining_amount_proof_l3904_390404


namespace NUMINAMATH_CALUDE_sugar_solution_volume_l3904_390420

/-- Given a sugar solution, prove that the initial volume was 3 liters -/
theorem sugar_solution_volume (V : ℝ) : 
  V > 0 → -- Initial volume is positive
  (0.4 * V) / (V + 1) = 0.30000000000000004 → -- New concentration after adding 1 liter of water
  V = 3 := by
sorry

end NUMINAMATH_CALUDE_sugar_solution_volume_l3904_390420


namespace NUMINAMATH_CALUDE_valid_numbers_l3904_390414

def is_valid (n : ℕ+) : Prop :=
  ∀ a : ℕ+, (a ≤ 1 + Real.sqrt n.val) → (Nat.gcd a.val n.val = 1) →
    ∃ x : ℤ, (a.val : ℤ) ≡ x^2 [ZMOD n.val]

theorem valid_numbers : {n : ℕ+ | is_valid n} = {1, 2, 12} := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l3904_390414


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3904_390438

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 6 = 2 →
  a 8 = 4 →
  a 10 + a 4 = 6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3904_390438


namespace NUMINAMATH_CALUDE_average_study_time_difference_l3904_390441

-- Define the list of daily differences
def daily_differences : List Int := [15, -5, 25, 0, -15, 10, 20]

-- Define the number of days
def num_days : Nat := 7

-- Theorem to prove
theorem average_study_time_difference :
  (daily_differences.sum : ℚ) / num_days = 50 / 7 := by
  sorry

end NUMINAMATH_CALUDE_average_study_time_difference_l3904_390441


namespace NUMINAMATH_CALUDE_trains_passing_time_l3904_390432

theorem trains_passing_time (train_length : ℝ) (train_speed : ℝ) : 
  train_length = 500 →
  train_speed = 30 →
  (2 * train_length) / (2 * train_speed * (5/18)) = 60 :=
by
  sorry

#check trains_passing_time

end NUMINAMATH_CALUDE_trains_passing_time_l3904_390432


namespace NUMINAMATH_CALUDE_percentage_equation_l3904_390436

theorem percentage_equation (x : ℝ) : (65 / 100 * x = 20 / 100 * 747.50) → x = 230 := by
  sorry

end NUMINAMATH_CALUDE_percentage_equation_l3904_390436


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3904_390419

theorem expand_and_simplify (x : ℝ) : (17 * x - 9) * (3 * x) = 51 * x^2 - 27 * x := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3904_390419


namespace NUMINAMATH_CALUDE_koala_fiber_intake_l3904_390482

/-- Represents the amount of fiber a koala eats and absorbs in a day. -/
structure KoalaFiber where
  eaten : ℝ
  absorbed : ℝ
  absorption_rate : ℝ
  absorption_equation : absorbed = absorption_rate * eaten

/-- Theorem: If a koala absorbs 20% of the fiber it eats and absorbed 12 ounces
    of fiber in one day, then it ate 60 ounces of fiber that day. -/
theorem koala_fiber_intake (k : KoalaFiber) 
    (h1 : k.absorption_rate = 0.20)
    (h2 : k.absorbed = 12) : 
    k.eaten = 60 := by
  sorry

end NUMINAMATH_CALUDE_koala_fiber_intake_l3904_390482


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3904_390464

def U : Set ℕ := {x | x < 8}

def A : Set ℕ := {x | (x - 1) * (x - 3) * (x - 4) * (x - 7) = 0}

theorem complement_of_A_in_U : U \ A = {0, 2, 5, 6} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3904_390464


namespace NUMINAMATH_CALUDE_possible_ad_values_l3904_390470

/-- Represents a point on a line -/
structure Point :=
  (x : ℝ)

/-- The distance between two points -/
def distance (p q : Point) : ℝ := |p.x - q.x|

/-- Theorem: Possible values of AD given AB = 1, BC = 2, CD = 4 -/
theorem possible_ad_values (A B C D : Point) 
  (h1 : distance A B = 1)
  (h2 : distance B C = 2)
  (h3 : distance C D = 4) :
  (distance A D = 1) ∨ (distance A D = 3) ∨ (distance A D = 5) ∨ (distance A D = 7) :=
sorry

end NUMINAMATH_CALUDE_possible_ad_values_l3904_390470


namespace NUMINAMATH_CALUDE_eighteenth_term_of_equally_summed_sequence_l3904_390468

/-- An Equally Summed Sequence is a sequence where the sum of each term and its subsequent term is always constant. -/
def EquallyStandardSequence (a : ℕ → ℝ) (c : ℝ) :=
  ∀ n, a n + a (n + 1) = c

theorem eighteenth_term_of_equally_summed_sequence
  (a : ℕ → ℝ)
  (h1 : EquallyStandardSequence a 5)
  (h2 : a 1 = 2) :
  a 18 = 3 := by
sorry

end NUMINAMATH_CALUDE_eighteenth_term_of_equally_summed_sequence_l3904_390468


namespace NUMINAMATH_CALUDE_abs_neg_five_plus_three_l3904_390498

theorem abs_neg_five_plus_three : |(-5 : ℤ) + 3| = 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_plus_three_l3904_390498


namespace NUMINAMATH_CALUDE_tangent_line_and_extreme_values_l3904_390473

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x + 2 / x

theorem tangent_line_and_extreme_values (a : ℝ) :
  (∀ x : ℝ, x > 0 → f a x = Real.log x - a * x + 2 / x) →
  (a = 1 → ∀ x y : ℝ, y = f 1 x → (x = 1 ∧ y = 1) → 2 * x + y - 3 = 0) ∧
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧
    (∀ x : ℝ, x > 0 → f a x ≤ f a x₁ ∧ f a x ≤ f a x₂) ↔ 0 < a ∧ a < 1/8) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_and_extreme_values_l3904_390473


namespace NUMINAMATH_CALUDE_power_of_ten_negative_y_l3904_390463

theorem power_of_ten_negative_y (y : ℝ) (h : (10 : ℝ) ^ (2 * y) = 25) : (10 : ℝ) ^ (-y) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_ten_negative_y_l3904_390463


namespace NUMINAMATH_CALUDE_larger_number_problem_l3904_390499

theorem larger_number_problem (x y : ℝ) (h1 : x > y) (h2 : x + y = 30) (h3 : 2 * y - x = 6) : x = 18 := by
  sorry

end NUMINAMATH_CALUDE_larger_number_problem_l3904_390499


namespace NUMINAMATH_CALUDE_rectangle_area_doubling_l3904_390406

theorem rectangle_area_doubling (l w : ℝ) (h1 : l > 0) (h2 : w > 0) :
  let new_length := 1.4 * l
  let new_width := (10/7) * w
  let original_area := l * w
  let new_area := new_length * new_width
  new_area = 2 * original_area := by sorry

end NUMINAMATH_CALUDE_rectangle_area_doubling_l3904_390406


namespace NUMINAMATH_CALUDE_regular_pyramid_volume_l3904_390403

theorem regular_pyramid_volume (b : ℝ) (h : b = 2) :
  ∀ V : ℝ, V ≤ (16 * Real.pi) / (9 * Real.sqrt 3) → V < 3.25 := by sorry

end NUMINAMATH_CALUDE_regular_pyramid_volume_l3904_390403


namespace NUMINAMATH_CALUDE_problem_solution_l3904_390465

theorem problem_solution (a n : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * n * 45 * 49) : n = 125 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3904_390465


namespace NUMINAMATH_CALUDE_gcd_210_162_l3904_390408

theorem gcd_210_162 : Nat.gcd 210 162 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_210_162_l3904_390408


namespace NUMINAMATH_CALUDE_farmer_tomato_rows_l3904_390407

/-- The number of tomato plants in each row -/
def plants_per_row : ℕ := 10

/-- The number of tomatoes yielded by each plant -/
def tomatoes_per_plant : ℕ := 20

/-- The total number of tomatoes harvested by the farmer -/
def total_tomatoes : ℕ := 6000

/-- The number of rows of tomatoes planted by the farmer -/
def rows_of_tomatoes : ℕ := total_tomatoes / (plants_per_row * tomatoes_per_plant)

theorem farmer_tomato_rows : rows_of_tomatoes = 30 := by
  sorry

end NUMINAMATH_CALUDE_farmer_tomato_rows_l3904_390407


namespace NUMINAMATH_CALUDE_invalid_external_diagonals_l3904_390459

/-- Checks if three numbers can be the lengths of external diagonals of a right regular prism -/
def are_valid_external_diagonals (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ a^2 + c^2 > b^2

/-- Theorem stating that {5, 7, 9} cannot be the external diagonals of a right regular prism -/
theorem invalid_external_diagonals :
  ¬(are_valid_external_diagonals 5 7 9) := by
  sorry

end NUMINAMATH_CALUDE_invalid_external_diagonals_l3904_390459


namespace NUMINAMATH_CALUDE_building_shadow_length_l3904_390410

/-- Given a flagpole and a building under similar conditions, 
    calculate the length of the shadow cast by the building. -/
theorem building_shadow_length 
  (flagpole_height : ℝ) 
  (flagpole_shadow : ℝ) 
  (building_height : ℝ) 
  (h1 : flagpole_height = 18)
  (h2 : flagpole_shadow = 45)
  (h3 : building_height = 22) :
  (building_height * flagpole_shadow) / flagpole_height = 55 := by
sorry

end NUMINAMATH_CALUDE_building_shadow_length_l3904_390410


namespace NUMINAMATH_CALUDE_smallest_integer_above_root_sum_power_l3904_390487

theorem smallest_integer_above_root_sum_power :
  ∃ n : ℕ, (n = 3323 ∧ (∀ m : ℕ, m < n → m ≤ (Real.sqrt 5 + Real.sqrt 3)^6) ∧
            (∀ k : ℕ, k > (Real.sqrt 5 + Real.sqrt 3)^6 → k ≥ n)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_above_root_sum_power_l3904_390487


namespace NUMINAMATH_CALUDE_rational_function_value_at_zero_l3904_390418

/-- A rational function with specific properties -/
structure RationalFunction where
  r : ℝ → ℝ  -- Numerator polynomial
  s : ℝ → ℝ  -- Denominator polynomial
  is_quadratic_r : ∃ a b c : ℝ, ∀ x, r x = a * x^2 + b * x + c
  is_quadratic_s : ∃ a b c : ℝ, ∀ x, s x = a * x^2 + b * x + c
  horizontal_asymptote : ∀ x, abs x > 1 → |r x / s x - 3| < 1
  vertical_asymptote_neg3 : s (-3) = 0
  vertical_asymptote_1 : s 1 = 0
  hole_at_2 : r 2 = 0 ∧ s 2 = 0

/-- Theorem stating that r(0)/s(0) = -1 for the given rational function -/
theorem rational_function_value_at_zero (f : RationalFunction) : f.r 0 / f.s 0 = -1 := by
  sorry

end NUMINAMATH_CALUDE_rational_function_value_at_zero_l3904_390418


namespace NUMINAMATH_CALUDE_function_determination_l3904_390490

theorem function_determination (f : ℝ → ℝ) 
  (h0 : f 0 = 1) 
  (h1 : ∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2) : 
  ∀ x : ℝ, f x = x + 1 := by
sorry

end NUMINAMATH_CALUDE_function_determination_l3904_390490


namespace NUMINAMATH_CALUDE_chord_midpoint_line_l3904_390440

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/4 = 1

-- Define point P
def P : ℝ × ℝ := (2, 1)

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x + 2*y - 4 = 0

-- Theorem statement
theorem chord_midpoint_line :
  ∀ (A B : ℝ × ℝ),
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 →  -- A and B are on the ellipse
  P = ((A.1 + B.1)/2, (A.2 + B.2)/2) →  -- P is the midpoint of AB
  line_equation A.1 A.2 ∧ line_equation B.1 B.2  -- A and B satisfy the line equation
  := by sorry

end NUMINAMATH_CALUDE_chord_midpoint_line_l3904_390440


namespace NUMINAMATH_CALUDE_books_at_end_of_month_l3904_390409

/-- Given a special collection of books, calculate the number of books at the end of the month. -/
theorem books_at_end_of_month 
  (initial_books : ℕ) 
  (loaned_books : ℕ) 
  (return_rate : ℚ) 
  (h1 : initial_books = 75)
  (h2 : loaned_books = 40)
  (h3 : return_rate = 65 / 100) : 
  initial_books - loaned_books + (return_rate * loaned_books).floor = 61 := by
  sorry

#check books_at_end_of_month

end NUMINAMATH_CALUDE_books_at_end_of_month_l3904_390409


namespace NUMINAMATH_CALUDE_min_value_abc_min_value_abc_achievable_l3904_390448

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 64) :
  a^2 + 6*a*b + 9*b^2 + 3*c^2 ≥ 192 :=
by sorry

theorem min_value_abc_achievable :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 64 ∧ a^2 + 6*a*b + 9*b^2 + 3*c^2 = 192 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_min_value_abc_achievable_l3904_390448


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3904_390442

-- Problem 1
theorem problem_1 : 25 - 9 + (-12) - (-7) = 4 := by sorry

-- Problem 2
theorem problem_2 : 1/9 * (-2)^3 / (2/3)^2 = -2 := by sorry

-- Problem 3
theorem problem_3 : (5/12 + 2/3 - 3/4) * (-12) = -4 := by sorry

-- Problem 4
theorem problem_4 : -1^4 + (-2) / (-1/3) - |(-9)| = -4 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3904_390442


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3904_390483

theorem quadratic_factorization (t : ℝ) : t^2 - 10*t + 25 = (t - 5)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3904_390483


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_times_i_l3904_390433

-- Define the complex number z
def z : ℂ := 4 - 8 * Complex.I

-- State the theorem
theorem imaginary_part_of_z_times_i : Complex.im (z * Complex.I) = 4 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_times_i_l3904_390433


namespace NUMINAMATH_CALUDE_total_shared_amount_l3904_390451

theorem total_shared_amount 
  (T a b c d : ℚ) 
  (h1 : a = (1/3) * (b + c + d))
  (h2 : b = (2/7) * (a + c + d))
  (h3 : c = (4/9) * (a + b + d))
  (h4 : d = (5/11) * (a + b + c))
  (h5 : a = b + 20)
  (h6 : c = d - 15)
  (h7 : T = a + b + c + d)
  (h8 : ∃ k : ℤ, T = 10 * k) :
  T = 1330 := by
sorry

end NUMINAMATH_CALUDE_total_shared_amount_l3904_390451


namespace NUMINAMATH_CALUDE_parallel_lines_D_eq_18_l3904_390446

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes {m1 m2 b1 b2 : ℝ} :
  (∀ x y : ℝ, y = m1 * x + b1 ↔ y = m2 * x + b2) ↔ m1 = m2

/-- The first line equation -/
def line1 (x y : ℝ) : Prop := x + 2 * y + 1 = 0

/-- The second line equation -/
def line2 (D : ℝ) (x y : ℝ) : Prop := 9 * x + D * y + 1 = 0

/-- The main theorem: if the two lines are parallel, then D = 18 -/
theorem parallel_lines_D_eq_18 :
  (∃ D : ℝ, ∀ x y : ℝ, (line1 x y ↔ line2 D x y) → D = 18) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_D_eq_18_l3904_390446


namespace NUMINAMATH_CALUDE_regular_polygon_perimeter_l3904_390411

/-- Given a regular polygon with central angle 45° and side length 5, its perimeter is 40. -/
theorem regular_polygon_perimeter (central_angle : ℝ) (side_length : ℝ) :
  central_angle = 45 →
  side_length = 5 →
  (360 / central_angle) * side_length = 40 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_perimeter_l3904_390411


namespace NUMINAMATH_CALUDE_num_al_sandwiches_l3904_390497

/-- Represents the number of different types of bread available. -/
def num_breads : Nat := 5

/-- Represents the number of different types of meat available. -/
def num_meats : Nat := 6

/-- Represents the number of different types of cheese available. -/
def num_cheeses : Nat := 6

/-- Represents the number of forbidden combinations. -/
def num_forbidden : Nat := 3

/-- Represents the number of overcounted combinations. -/
def num_overcounted : Nat := 1

/-- Calculates the total number of possible sandwich combinations. -/
def total_combinations : Nat := num_breads * num_meats * num_cheeses

/-- Calculates the number of forbidden sandwich combinations. -/
def forbidden_combinations : Nat :=
  num_breads + num_cheeses + num_cheeses - num_overcounted

/-- Theorem stating the number of different sandwiches Al can order. -/
theorem num_al_sandwiches : 
  total_combinations - forbidden_combinations = 164 := by
  sorry

end NUMINAMATH_CALUDE_num_al_sandwiches_l3904_390497
