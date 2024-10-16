import Mathlib

namespace NUMINAMATH_CALUDE_inequality_equivalence_l2367_236781

theorem inequality_equivalence (x : ℝ) : 
  (5 ≤ x / (2 * x - 6) ∧ x / (2 * x - 6) < 10) ↔ (3 < x ∧ x < 60 / 19) :=
by sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2367_236781


namespace NUMINAMATH_CALUDE_average_problem_l2367_236747

theorem average_problem (y : ℝ) : (15 + 26 + y) / 3 = 23 → y = 28 := by
  sorry

end NUMINAMATH_CALUDE_average_problem_l2367_236747


namespace NUMINAMATH_CALUDE_logarithm_comparison_l2367_236759

theorem logarithm_comparison : ∃ (a b c : ℝ),
  a = Real.log 2 / Real.log 3 ∧
  b = Real.log 2 / Real.log 5 ∧
  c = Real.log 3 / Real.log 2 ∧
  c > a ∧ a > b := by
  sorry

end NUMINAMATH_CALUDE_logarithm_comparison_l2367_236759


namespace NUMINAMATH_CALUDE_marketing_cost_per_book_l2367_236738

/-- The marketing cost per book for a publishing company --/
theorem marketing_cost_per_book 
  (fixed_cost : ℝ) 
  (selling_price : ℝ) 
  (break_even_quantity : ℕ) 
  (h1 : fixed_cost = 50000)
  (h2 : selling_price = 9)
  (h3 : break_even_quantity = 10000) :
  (selling_price * break_even_quantity - fixed_cost) / break_even_quantity = 4 := by
sorry


end NUMINAMATH_CALUDE_marketing_cost_per_book_l2367_236738


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2367_236710

theorem polynomial_divisibility (x : ℝ) : 
  let f : ℝ → ℝ := λ x => -x^4 - x^3 - x + 1
  ∃ (u v : ℝ → ℝ), 
    f x = (x^2 + 1) * (u x) ∧ 
    f x + 1 = (x^3 + x^2 + 1) * (v x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2367_236710


namespace NUMINAMATH_CALUDE_vector_properties_l2367_236731

def a : ℝ × ℝ := (2, 0)
def b : ℝ × ℝ := (-1, -1)

theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 ≠ 2) ∧
  (∃ (k : ℝ), a ≠ k • b) ∧
  (b.1 * (a.1 + b.1) + b.2 * (a.2 + b.2) = 0) ∧
  (a.1^2 + a.2^2 ≠ b.1^2 + b.2^2) :=
by sorry

end NUMINAMATH_CALUDE_vector_properties_l2367_236731


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_y_l2367_236712

noncomputable def y (x : ℝ) : ℝ := (1/3) * Real.cos x - 1

theorem sum_of_max_and_min_y : 
  (⨆ (x : ℝ), y x) + (⨅ (x : ℝ), y x) = -2 :=
sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_y_l2367_236712


namespace NUMINAMATH_CALUDE_tess_decoration_l2367_236751

/-- The number of heart stickers Tess has -/
def heart_stickers : ℕ := 120

/-- The number of star stickers Tess has -/
def star_stickers : ℕ := 81

/-- The number of smiley stickers Tess has -/
def smiley_stickers : ℕ := 45

/-- The greatest number of pages Tess can decorate -/
def max_pages : ℕ := Nat.gcd (Nat.gcd heart_stickers star_stickers) smiley_stickers

theorem tess_decoration :
  max_pages = 3 ∧
  heart_stickers % max_pages = 0 ∧
  star_stickers % max_pages = 0 ∧
  smiley_stickers % max_pages = 0 ∧
  ∀ n : ℕ, n > max_pages →
    (heart_stickers % n = 0 ∧ star_stickers % n = 0 ∧ smiley_stickers % n = 0) → False :=
by sorry

end NUMINAMATH_CALUDE_tess_decoration_l2367_236751


namespace NUMINAMATH_CALUDE_notebook_savings_theorem_l2367_236789

/-- Calculates the savings when buying notebooks on sale compared to regular price -/
def calculate_savings (original_price : ℚ) (regular_quantity : ℕ) (sale_quantity : ℕ) 
  (sale_discount : ℚ) (extra_discount : ℚ) : ℚ :=
  let regular_cost := original_price * regular_quantity
  let discounted_price := original_price * (1 - sale_discount)
  let sale_cost := if sale_quantity > 10
    then discounted_price * sale_quantity * (1 - extra_discount)
    else discounted_price * sale_quantity
  regular_cost - sale_cost

theorem notebook_savings_theorem : 
  let original_price : ℚ := 3
  let regular_quantity : ℕ := 8
  let sale_quantity : ℕ := 12
  let sale_discount : ℚ := 1/4
  let extra_discount : ℚ := 1/20
  calculate_savings original_price regular_quantity sale_quantity sale_discount extra_discount = 10.35 := by
  sorry

end NUMINAMATH_CALUDE_notebook_savings_theorem_l2367_236789


namespace NUMINAMATH_CALUDE_monic_quartic_value_l2367_236716

def is_monic_quartic (f : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, ∀ x, f x = x^4 + a*x^3 + b*x^2 + c*x + d

theorem monic_quartic_value (f : ℝ → ℝ) :
  is_monic_quartic f →
  f (-2) = -4 →
  f 1 = -1 →
  f (-3) = -9 →
  f 5 = -25 →
  f 2 = -64 := by
  sorry

end NUMINAMATH_CALUDE_monic_quartic_value_l2367_236716


namespace NUMINAMATH_CALUDE_bird_nest_theorem_l2367_236743

/-- Represents a bird's trip information -/
structure BirdTrip where
  trips_to_x : ℕ
  trips_to_y : ℕ
  trips_to_z : ℕ
  distance_to_x : ℕ
  distance_to_y : ℕ
  distance_to_z : ℕ
  time_to_x : ℕ
  time_to_y : ℕ
  time_to_z : ℕ

def bird_a : BirdTrip :=
  { trips_to_x := 15
  , trips_to_y := 0
  , trips_to_z := 10
  , distance_to_x := 300
  , distance_to_y := 0
  , distance_to_z := 400
  , time_to_x := 30
  , time_to_y := 0
  , time_to_z := 40 }

def bird_b : BirdTrip :=
  { trips_to_x := 0
  , trips_to_y := 20
  , trips_to_z := 5
  , distance_to_x := 0
  , distance_to_y := 500
  , distance_to_z := 600
  , time_to_x := 0
  , time_to_y := 60
  , time_to_z := 50 }

def total_distance (bird : BirdTrip) : ℕ :=
  2 * (bird.trips_to_x * bird.distance_to_x +
       bird.trips_to_y * bird.distance_to_y +
       bird.trips_to_z * bird.distance_to_z)

def total_time (bird : BirdTrip) : ℕ :=
  bird.trips_to_x * bird.time_to_x +
  bird.trips_to_y * bird.time_to_y +
  bird.trips_to_z * bird.time_to_z

theorem bird_nest_theorem :
  total_distance bird_a + total_distance bird_b = 43000 ∧
  total_time bird_a + total_time bird_b = 2300 := by
  sorry

end NUMINAMATH_CALUDE_bird_nest_theorem_l2367_236743


namespace NUMINAMATH_CALUDE_fraction_equality_l2367_236746

theorem fraction_equality (a b : ℝ) (h : 1/a - 1/b = 4) : 
  (a - 2*a*b - b) / (2*a + 7*a*b - 2*b) = 6 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2367_236746


namespace NUMINAMATH_CALUDE_ella_age_l2367_236763

/-- Given the ages of Sam, Tim, and Ella, prove that Ella is 15 years old. -/
theorem ella_age (s t e : ℕ) : 
  (s + t + e) / 3 = 12 →  -- The average of their ages is 12
  e - 5 = s →             -- Five years ago, Ella was the same age as Sam is now
  t + 4 = (3 * (s + 4)) / 4 →  -- In 4 years, Tim's age will be 3/4 of Sam's age at that time
  e = 15 := by
sorry


end NUMINAMATH_CALUDE_ella_age_l2367_236763


namespace NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l2367_236777

/-- The number of ways to distribute indistinguishable balls into indistinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The number of ways to distribute 5 balls into 3 boxes with one box always empty -/
theorem distribute_five_balls_three_boxes :
  distribute_balls 5 3 = 3 :=
sorry

end NUMINAMATH_CALUDE_distribute_five_balls_three_boxes_l2367_236777


namespace NUMINAMATH_CALUDE_average_of_sample_l2367_236748

def sample_average (x : Fin 10 → ℝ) (a b : ℝ) : Prop :=
  (x 0 + x 1 + x 2) / 3 = a ∧
  (x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9) / 7 = b

theorem average_of_sample (x : Fin 10 → ℝ) (a b : ℝ) 
  (h : sample_average x a b) : 
  (x 0 + x 1 + x 2 + x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9) / 10 = (3 * a + 7 * b) / 10 := by
  sorry

end NUMINAMATH_CALUDE_average_of_sample_l2367_236748


namespace NUMINAMATH_CALUDE_area_enclosed_by_curve_and_tangent_l2367_236734

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 1

-- Define the tangent line passing through (1, 0)
def tangent_line (x : ℝ) : ℝ := -x + 1

-- Theorem statement
theorem area_enclosed_by_curve_and_tangent :
  ∃ (a b : ℝ), a < b ∧ 
  (∀ x, a ≤ x ∧ x ≤ b → f x ≤ tangent_line x ∨ tangent_line x ≤ f x) ∧
  ∫ x in a..b, |f x - tangent_line x| = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_area_enclosed_by_curve_and_tangent_l2367_236734


namespace NUMINAMATH_CALUDE_total_games_is_62_l2367_236772

/-- Represents a baseball league with its characteristics and calculates the total number of games played -/
structure BaseballLeague where
  teams : Nat
  games_per_team_per_month : Nat
  season_months : Nat
  playoff_rounds : Nat
  games_per_playoff_round : Nat

/-- Calculates the total number of games played in the season, including playoffs -/
def BaseballLeague.total_games (league : BaseballLeague) : Nat :=
  let regular_season_games := (league.teams / 2) * league.games_per_team_per_month * league.season_months
  let playoff_games := league.playoff_rounds * league.games_per_playoff_round
  regular_season_games + playoff_games

/-- The specific baseball league described in the problem -/
def specific_league : BaseballLeague :=
  { teams := 8
  , games_per_team_per_month := 7
  , season_months := 2
  , playoff_rounds := 3
  , games_per_playoff_round := 2
  }

/-- Theorem stating that the total number of games in the specific league is 62 -/
theorem total_games_is_62 : specific_league.total_games = 62 := by
  sorry


end NUMINAMATH_CALUDE_total_games_is_62_l2367_236772


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2367_236799

theorem fraction_subtraction : (15 : ℚ) / 45 - (1 + 2 / 9) = -8 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2367_236799


namespace NUMINAMATH_CALUDE_fraction_simplification_l2367_236711

theorem fraction_simplification (a b c : ℝ) (h : a + b + c ≠ 0) :
  (a^2 + 3*a*b + b^2 - c^2) / (a^2 + 3*a*c + c^2 - b^2) = (a + b - c) / (a - b + c) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2367_236711


namespace NUMINAMATH_CALUDE_fuel_mixture_problem_l2367_236753

/-- Proves that the amount of fuel A added to the tank is 122 gallons -/
theorem fuel_mixture_problem (tank_capacity : ℝ) (ethanol_a : ℝ) (ethanol_b : ℝ) (total_ethanol : ℝ)
  (h1 : tank_capacity = 218)
  (h2 : ethanol_a = 0.12)
  (h3 : ethanol_b = 0.16)
  (h4 : total_ethanol = 30) :
  ∃ (fuel_a : ℝ), fuel_a = 122 ∧ 
    ethanol_a * fuel_a + ethanol_b * (tank_capacity - fuel_a) = total_ethanol :=
by sorry

end NUMINAMATH_CALUDE_fuel_mixture_problem_l2367_236753


namespace NUMINAMATH_CALUDE_sheridan_cats_l2367_236782

def current_cats : ℕ := sorry
def needed_cats : ℕ := 32
def total_cats : ℕ := 43

theorem sheridan_cats : current_cats = 11 := by
  sorry

end NUMINAMATH_CALUDE_sheridan_cats_l2367_236782


namespace NUMINAMATH_CALUDE_bakery_storage_l2367_236779

theorem bakery_storage (sugar flour baking_soda : ℝ) 
  (h1 : sugar / flour = 5 / 4)
  (h2 : flour / baking_soda = 10 / 1)
  (h3 : flour / (baking_soda + 60) = 8 / 1) :
  sugar = 3000 := by
sorry

end NUMINAMATH_CALUDE_bakery_storage_l2367_236779


namespace NUMINAMATH_CALUDE_vector_computation_l2367_236758

theorem vector_computation : 
  (4 : ℝ) • (![2, -9] : Fin 2 → ℝ) - (3 : ℝ) • (![(-1), -6] : Fin 2 → ℝ) = ![11, -18] :=
by sorry

end NUMINAMATH_CALUDE_vector_computation_l2367_236758


namespace NUMINAMATH_CALUDE_fred_found_43_seashells_l2367_236700

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := 15

/-- The total number of seashells found by Tom and Fred -/
def total_seashells : ℕ := 58

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := total_seashells - tom_seashells

theorem fred_found_43_seashells : fred_seashells = 43 := by
  sorry

end NUMINAMATH_CALUDE_fred_found_43_seashells_l2367_236700


namespace NUMINAMATH_CALUDE_ball_ratio_proof_l2367_236724

/-- Given that Robert initially had 25 balls, Tim initially had 40 balls,
    and Robert ended up with 45 balls after Tim gave him some balls,
    prove that the ratio of the number of balls Tim gave to Robert
    to the number of balls Tim had initially is 1:2. -/
theorem ball_ratio_proof (robert_initial : ℕ) (tim_initial : ℕ) (robert_final : ℕ)
    (h1 : robert_initial = 25)
    (h2 : tim_initial = 40)
    (h3 : robert_final = 45) :
    (robert_final - robert_initial) * 2 = tim_initial := by
  sorry

end NUMINAMATH_CALUDE_ball_ratio_proof_l2367_236724


namespace NUMINAMATH_CALUDE_mistaken_multiplication_l2367_236774

/-- Given a polynomial P(x) such that (-3x) * P(x) = 3x³ - 3x² + 3x,
    prove that P(x) - 3x = -x² - 2x - 1 -/
theorem mistaken_multiplication (P : ℝ → ℝ) :
  (∀ x, (-3 * x) * P x = 3 * x^3 - 3 * x^2 + 3 * x) →
  (∀ x, P x - 3 * x = -x^2 - 2 * x - 1) :=
by sorry

end NUMINAMATH_CALUDE_mistaken_multiplication_l2367_236774


namespace NUMINAMATH_CALUDE_mrs_hilt_pennies_l2367_236767

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a penny in cents -/
def penny_value : ℕ := 1

/-- Mrs. Hilt's coins without pennies -/
def mrs_hilt_coins : ℕ := 2 * dime_value + 2 * nickel_value

/-- Jacob's coins -/
def jacob_coins : ℕ := 4 * penny_value + nickel_value + dime_value

/-- The difference in their amounts in cents -/
def difference : ℕ := 13

/-- Theorem: Mrs. Hilt has 2 pennies -/
theorem mrs_hilt_pennies :
  ∃ (x : ℕ), mrs_hilt_coins + x * penny_value - jacob_coins = difference ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_mrs_hilt_pennies_l2367_236767


namespace NUMINAMATH_CALUDE_square_side_length_l2367_236728

theorem square_side_length (rectangle_width : ℝ) (rectangle_length : ℝ) (square_side : ℝ) : 
  rectangle_width = 4 →
  rectangle_length = 16 →
  square_side ^ 2 = rectangle_width * rectangle_length →
  square_side = 8 := by
sorry

end NUMINAMATH_CALUDE_square_side_length_l2367_236728


namespace NUMINAMATH_CALUDE_cloth_sale_problem_l2367_236721

/-- Proves that the number of metres of cloth sold is 200 given the specified conditions -/
theorem cloth_sale_problem (total_selling_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ) : 
  total_selling_price = 12000 →
  loss_per_metre = 12 →
  cost_price_per_metre = 72 →
  (total_selling_price / (cost_price_per_metre - loss_per_metre) : ℕ) = 200 :=
by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_problem_l2367_236721


namespace NUMINAMATH_CALUDE_inequality_proof_l2367_236786

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (a b c : ℝ)
  (ha : Real.sqrt a = x * (y - z)^2)
  (hb : Real.sqrt b = y * (z - x)^2)
  (hc : Real.sqrt c = z * (x - y)^2) :
  a^2 + b^2 + c^2 ≥ 2 * (a * b + b * c + c * a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2367_236786


namespace NUMINAMATH_CALUDE_original_number_proof_l2367_236787

theorem original_number_proof (x : ℤ) : (x + 2)^2 = x^2 - 2016 → x = -505 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l2367_236787


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2367_236717

/-- Given a quadratic equation 3x^2 + mx - 7 = 0 where -1 is one root, 
    prove that the other root is 7/3 -/
theorem other_root_of_quadratic (m : ℝ) : 
  (∃ x, 3 * x^2 + m * x - 7 = 0 ∧ x = -1) → 
  (∃ y, 3 * y^2 + m * y - 7 = 0 ∧ y = 7/3) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2367_236717


namespace NUMINAMATH_CALUDE_circle_symmetric_about_center_circle_symmetric_about_diameter_circle_is_symmetrical_l2367_236791

/-- Definition of a circle in a plane -/
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

/-- Definition of symmetry for a set about a point -/
def IsSymmetricAbout (S : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  ∀ x ∈ S, ∃ y ∈ S, p.1 = (x.1 + y.1) / 2 ∧ p.2 = (x.2 + y.2) / 2

/-- Theorem: Any circle is symmetric about its center -/
theorem circle_symmetric_about_center (center : ℝ × ℝ) (radius : ℝ) :
  IsSymmetricAbout (Circle center radius) center := by
  sorry

/-- Theorem: Any circle is symmetric about any of its diameters -/
theorem circle_symmetric_about_diameter (center : ℝ × ℝ) (radius : ℝ) (a b : ℝ × ℝ) 
  (ha : a ∈ Circle center radius) (hb : b ∈ Circle center radius)
  (hdiameter : (a.1 - b.1)^2 + (a.2 - b.2)^2 = 4 * radius^2) :
  IsSymmetricAbout (Circle center radius) ((a.1 + b.1) / 2, (a.2 + b.2) / 2) := by
  sorry

/-- Main theorem: Any circle is a symmetrical figure -/
theorem circle_is_symmetrical (center : ℝ × ℝ) (radius : ℝ) :
  ∃ p, IsSymmetricAbout (Circle center radius) p := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetric_about_center_circle_symmetric_about_diameter_circle_is_symmetrical_l2367_236791


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2367_236736

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∃ r₁ r₂ : ℝ, (r₁ ≠ r₂) ∧ 
    (∀ x : ℝ, x^2 + p*x + m = 0 ↔ x = r₁ ∨ x = r₂) ∧
    (∀ x : ℝ, x^2 + m*x + n = 0 ↔ x = r₁/2 ∨ x = r₂/2)) →
  n / p = 1 / 8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2367_236736


namespace NUMINAMATH_CALUDE_right_triangles_with_sqrt1001_leg_l2367_236744

theorem right_triangles_with_sqrt1001_leg :
  ∃! (n : ℕ), n > 0 ∧ n = (Finset.filter 
    (fun t : ℕ × ℕ × ℕ => 
      t.1 * t.1 + 1001 = t.2.2 * t.2.2 ∧ 
      t.2.1 * t.2.1 = 1001 ∧
      t.1 > 0 ∧ t.2.1 > 0 ∧ t.2.2 > 0)
    (Finset.product (Finset.range 1000) (Finset.product (Finset.range 1000) (Finset.range 1000)))).card ∧
  n = 4 :=
sorry

end NUMINAMATH_CALUDE_right_triangles_with_sqrt1001_leg_l2367_236744


namespace NUMINAMATH_CALUDE_ratio_of_21_to_reversed_l2367_236735

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem ratio_of_21_to_reversed : 
  let original := 21
  let reversed := reverse_digits original
  (original : ℚ) / reversed = 7 / 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_21_to_reversed_l2367_236735


namespace NUMINAMATH_CALUDE_percentage_of_boys_with_dogs_proof_percentage_of_boys_with_dogs_l2367_236768

theorem percentage_of_boys_with_dogs 
  (total_students : ℕ) 
  (girls : ℕ) 
  (boys : ℕ) 
  (girls_with_dogs : ℕ) 
  (total_with_dogs : ℕ) : Prop :=
  total_students = 100 →
  girls = total_students / 2 →
  boys = total_students / 2 →
  girls_with_dogs = girls * 20 / 100 →
  total_with_dogs = 15 →
  (total_with_dogs - girls_with_dogs) * 100 / boys = 10

theorem proof_percentage_of_boys_with_dogs : 
  percentage_of_boys_with_dogs 100 50 50 10 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_boys_with_dogs_proof_percentage_of_boys_with_dogs_l2367_236768


namespace NUMINAMATH_CALUDE_maci_red_pens_l2367_236795

/-- The number of blue pens Maci needs -/
def blue_pens : ℕ := 10

/-- The cost of a blue pen in cents -/
def blue_pen_cost : ℕ := 10

/-- The cost of a red pen in cents -/
def red_pen_cost : ℕ := 2 * blue_pen_cost

/-- The total cost of all pens in cents -/
def total_cost : ℕ := 400

/-- The number of red pens Maci needs -/
def red_pens : ℕ := 15

theorem maci_red_pens :
  blue_pens * blue_pen_cost + red_pens * red_pen_cost = total_cost :=
sorry

end NUMINAMATH_CALUDE_maci_red_pens_l2367_236795


namespace NUMINAMATH_CALUDE_special_function_property_l2367_236762

/-- A differentiable function f satisfying f(x) - f''(x) > 0 for all x --/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ 
  (∀ x : ℝ, Differentiable ℝ (deriv f)) ∧
  (∀ x : ℝ, f x - (deriv (deriv f)) x > 0)

/-- Theorem stating that for a special function f, ef(2015) > f(2016) --/
theorem special_function_property (f : ℝ → ℝ) (hf : SpecialFunction f) : 
  Real.exp 1 * f 2015 > f 2016 := by
  sorry

end NUMINAMATH_CALUDE_special_function_property_l2367_236762


namespace NUMINAMATH_CALUDE_people_in_line_l2367_236742

theorem people_in_line (people_between : ℕ) (h : people_between = 5) : 
  people_between + 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_people_in_line_l2367_236742


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l2367_236740

theorem cyclic_sum_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l2367_236740


namespace NUMINAMATH_CALUDE_square_root_cube_root_relation_l2367_236702

theorem square_root_cube_root_relation (x : ℝ) : 
  (∃ y : ℝ, y^2 = x ∧ (y = 8 ∨ y = -8)) → x^(1/3) = 4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_cube_root_relation_l2367_236702


namespace NUMINAMATH_CALUDE_opposite_roots_imply_k_value_l2367_236730

theorem opposite_roots_imply_k_value (k : ℝ) : 
  (∃ x : ℝ, x^2 + (k^2 - 4)*x + (k - 1) = 0 ∧ 
             ∃ y : ℝ, y^2 + (k^2 - 4)*y + (k - 1) = 0 ∧ 
             x = -y ∧ x ≠ y) → 
  k = -2 :=
by sorry

end NUMINAMATH_CALUDE_opposite_roots_imply_k_value_l2367_236730


namespace NUMINAMATH_CALUDE_x_values_theorem_l2367_236726

theorem x_values_theorem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h1 : x + 1 / y = 10) (h2 : y + 1 / x = 5 / 12) :
  x = 4 ∨ x = 6 :=
by sorry

end NUMINAMATH_CALUDE_x_values_theorem_l2367_236726


namespace NUMINAMATH_CALUDE_helmet_safety_analysis_l2367_236780

/-- Data for people not wearing helmets over 4 years -/
def helmet_data : List (Nat × Nat) := [(1, 1250), (2, 1050), (3, 1000), (4, 900)]

/-- Contingency table for helmet wearing and casualties -/
def contingency_table : Matrix (Fin 2) (Fin 2) Nat :=
  ![![7, 3],
    ![13, 27]]

/-- Calculate the regression line equation coefficients -/
def regression_line (data : List (Nat × Nat)) : ℝ × ℝ :=
  sorry

/-- Estimate the number of people not wearing helmets for a given year -/
def estimate_no_helmet (coef : ℝ × ℝ) (year : Nat) : ℝ :=
  sorry

/-- Calculate the K^2 statistic for a 2x2 contingency table -/
def k_squared (table : Matrix (Fin 2) (Fin 2) Nat) : ℝ :=
  sorry

theorem helmet_safety_analysis :
  let (b, a) := regression_line helmet_data
  (b = -110 ∧ a = 1325) ∧
  estimate_no_helmet (b, a) 5 = 775 ∧
  k_squared contingency_table > 3.841 :=
sorry

end NUMINAMATH_CALUDE_helmet_safety_analysis_l2367_236780


namespace NUMINAMATH_CALUDE_symmetric_complex_sum_third_quadrant_l2367_236704

/-- Given two complex numbers symmetric with respect to the imaginary axis,
    prove that their sum with one divided by its modulus squared is in the third quadrant -/
theorem symmetric_complex_sum_third_quadrant (z₁ z : ℂ) : 
  z₁ = 2 - I →
  z = -Complex.re z₁ + Complex.im z₁ * I → 
  let w := z₁ / Complex.normSq z₁ + z
  Complex.re w < 0 ∧ Complex.im w < 0 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_complex_sum_third_quadrant_l2367_236704


namespace NUMINAMATH_CALUDE_horse_food_calculation_l2367_236798

/-- Calculates the total food needed for a given number of horses over a specified number of days -/
def total_food_needed (num_horses : ℕ) (oats_per_meal : ℕ) (oats_meals_per_day : ℕ) (grain_per_day : ℕ) (num_days : ℕ) : ℕ :=
  num_horses * (oats_per_meal * oats_meals_per_day + grain_per_day) * num_days

/-- Proves that 4 horses need 132 pounds of food for 3 days given the specified feeding regimen -/
theorem horse_food_calculation :
  total_food_needed 4 4 2 3 3 = 132 := by
  sorry

end NUMINAMATH_CALUDE_horse_food_calculation_l2367_236798


namespace NUMINAMATH_CALUDE_fundraiser_results_l2367_236766

/-- Represents the sales data for Markeesha's fundraiser --/
structure FundraiserSales where
  M : ℕ  -- Number of boxes sold on Monday
  regular_price : ℕ := 3  -- Price of regular crackers
  whole_grain_price : ℕ := 6  -- Price of whole grain crackers

/-- Calculates the total profit for the week --/
def total_profit (sales : FundraiserSales) : ℝ :=
  (4 * sales.M + 205 : ℝ) * 4.5

/-- Calculates the total number of boxes sold for the week --/
def total_boxes_sold (sales : FundraiserSales) : ℕ :=
  4 * sales.M + 205

/-- Represents the sales for each day of the week --/
def daily_sales (sales : FundraiserSales) : Fin 7 → ℕ
| 0 => sales.M  -- Monday
| 1 => sales.M + 10  -- Tuesday
| 2 => sales.M + 20  -- Wednesday
| 3 => sales.M + 30  -- Thursday
| 4 => 30  -- Friday
| 5 => 60  -- Saturday
| 6 => 45  -- Sunday

/-- Theorem stating the main results of the fundraiser --/
theorem fundraiser_results (sales : FundraiserSales) :
  (total_profit sales = (4 * sales.M + 205 : ℝ) * 4.5) ∧
  (total_boxes_sold sales = 4 * sales.M + 205) ∧
  (∀ i : Fin 7, daily_sales sales i ≤ daily_sales sales 5) :=
sorry

end NUMINAMATH_CALUDE_fundraiser_results_l2367_236766


namespace NUMINAMATH_CALUDE_matching_socks_probability_theorem_l2367_236707

/-- The number of different pairs of socks -/
def num_pairs : ℕ := 5

/-- The number of days socks are selected -/
def num_days : ℕ := 5

/-- The probability of wearing matching socks on both the third and fifth day -/
def matching_socks_probability : ℚ := 1 / 63

/-- Theorem stating the probability of wearing matching socks on both the third and fifth day -/
theorem matching_socks_probability_theorem :
  let total_socks := 2 * num_pairs
  let favorable_outcomes := num_pairs * (num_pairs - 1) * (Nat.choose (total_socks - 4) 2) * (Nat.choose (total_socks - 6) 2) * (Nat.choose (total_socks - 8) 2)
  let total_outcomes := (Nat.choose total_socks 2) * (Nat.choose (total_socks - 2) 2) * (Nat.choose (total_socks - 4) 2) * (Nat.choose (total_socks - 6) 2) * (Nat.choose (total_socks - 8) 2)
  (favorable_outcomes : ℚ) / total_outcomes = matching_socks_probability :=
by sorry

#check matching_socks_probability_theorem

end NUMINAMATH_CALUDE_matching_socks_probability_theorem_l2367_236707


namespace NUMINAMATH_CALUDE_equation_solutions_l2367_236765

theorem equation_solutions :
  ∃! (x y : ℝ), y = (x + 2)^2 ∧ x * y + 2 * y = 2 ∧
  ∃ (a b c d : ℂ), a ≠ x ∧ c ≠ x ∧
    (a, b) ≠ (c, d) ∧
    b = (a + 2)^2 ∧ a * b + 2 * b = 2 ∧
    d = (c + 2)^2 ∧ c * d + 2 * d = 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2367_236765


namespace NUMINAMATH_CALUDE_difference_of_percentages_l2367_236792

theorem difference_of_percentages : (0.7 * 40) - ((4 / 5) * 25) = 8 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_percentages_l2367_236792


namespace NUMINAMATH_CALUDE_monthly_payment_difference_l2367_236713

/-- The cost of the house in dollars -/
def house_cost : ℕ := 480000

/-- The cost of the trailer in dollars -/
def trailer_cost : ℕ := 120000

/-- The number of months over which the loans are paid -/
def loan_duration_months : ℕ := 240

/-- The monthly payment for the house -/
def house_monthly_payment : ℚ := house_cost / loan_duration_months

/-- The monthly payment for the trailer -/
def trailer_monthly_payment : ℚ := trailer_cost / loan_duration_months

/-- Theorem stating the difference in monthly payments -/
theorem monthly_payment_difference :
  house_monthly_payment - trailer_monthly_payment = 1500 := by
  sorry


end NUMINAMATH_CALUDE_monthly_payment_difference_l2367_236713


namespace NUMINAMATH_CALUDE_arithmetic_sequence_solve_y_l2367_236741

/-- Given that 1/3, y-2, and 4y are consecutive terms of an arithmetic sequence, prove that y = -13/6 -/
theorem arithmetic_sequence_solve_y (y : ℚ) : 
  (y - 2 - (1/3 : ℚ) = 4*y - (y - 2)) → y = -13/6 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_solve_y_l2367_236741


namespace NUMINAMATH_CALUDE_binomial_60_3_l2367_236705

theorem binomial_60_3 : Nat.choose 60 3 = 34220 := by
  sorry

end NUMINAMATH_CALUDE_binomial_60_3_l2367_236705


namespace NUMINAMATH_CALUDE_tank_flow_rate_l2367_236775

/-- Represents the flow rate problem for a water tank -/
theorem tank_flow_rate 
  (tank_capacity : ℝ) 
  (initial_level : ℝ) 
  (fill_time : ℝ) 
  (drain1_rate : ℝ) 
  (drain2_rate : ℝ) 
  (h1 : tank_capacity = 8000)
  (h2 : initial_level = tank_capacity / 2)
  (h3 : fill_time = 48)
  (h4 : drain1_rate = 1000 / 4)
  (h5 : drain2_rate = 1000 / 6)
  : ∃ (flow_rate : ℝ), 
    flow_rate = 500 ∧ 
    (flow_rate - (drain1_rate + drain2_rate)) * fill_time = tank_capacity - initial_level :=
by sorry


end NUMINAMATH_CALUDE_tank_flow_rate_l2367_236775


namespace NUMINAMATH_CALUDE_square_field_area_l2367_236749

/-- Represents the properties of a square field with barbed wire fencing --/
structure SquareField where
  side : ℝ
  wireRate : ℝ
  gateWidth : ℝ
  gateCount : ℕ
  totalCost : ℝ

/-- Calculates the area of the square field --/
def fieldArea (field : SquareField) : ℝ :=
  field.side * field.side

/-- Calculates the length of barbed wire needed --/
def wireLength (field : SquareField) : ℝ :=
  4 * field.side - field.gateWidth * field.gateCount

/-- Theorem stating the area of the square field given the conditions --/
theorem square_field_area (field : SquareField)
  (h1 : field.wireRate = 1)
  (h2 : field.gateWidth = 1)
  (h3 : field.gateCount = 2)
  (h4 : field.totalCost = 666)
  (h5 : wireLength field * field.wireRate = field.totalCost) :
  fieldArea field = 27889 := by
  sorry

#eval 167 * 167  -- To verify the result

end NUMINAMATH_CALUDE_square_field_area_l2367_236749


namespace NUMINAMATH_CALUDE_valid_seating_arrangements_count_l2367_236784

-- Define the people
inductive Person : Type
| Alice : Person
| Bob : Person
| Carla : Person
| Derek : Person
| Eric : Person

-- Define a seating arrangement as a function from position to person
def SeatingArrangement := Fin 5 → Person

-- Define the condition that two people cannot sit next to each other
def CannotSitNextTo (p1 p2 : Person) (arrangement : SeatingArrangement) : Prop :=
  ∀ i : Fin 4, arrangement i ≠ p1 ∨ arrangement (Fin.succ i) ≠ p2

-- Define a valid seating arrangement
def ValidArrangement (arrangement : SeatingArrangement) : Prop :=
  (CannotSitNextTo Person.Alice Person.Bob arrangement) ∧
  (CannotSitNextTo Person.Alice Person.Carla arrangement) ∧
  (CannotSitNextTo Person.Carla Person.Bob arrangement) ∧
  (CannotSitNextTo Person.Carla Person.Derek arrangement) ∧
  (CannotSitNextTo Person.Derek Person.Eric arrangement)

-- The main theorem
theorem valid_seating_arrangements_count :
  ∃ arrangements : Finset SeatingArrangement,
    (∀ arr ∈ arrangements, ValidArrangement arr) ∧
    (∀ arr, ValidArrangement arr → arr ∈ arrangements) ∧
    arrangements.card = 12 :=
sorry

end NUMINAMATH_CALUDE_valid_seating_arrangements_count_l2367_236784


namespace NUMINAMATH_CALUDE_dog_movement_area_calculation_l2367_236725

/-- Represents the dimensions and constraints of a dog tied to a square doghouse --/
structure DogHouseSetup where
  side_length : ℝ
  tie_point_distance : ℝ
  chain_length : ℝ

/-- Calculates the area in which the dog can move --/
def dog_movement_area (setup : DogHouseSetup) : ℝ :=
  sorry

/-- Theorem stating the area in which the dog can move for the given setup --/
theorem dog_movement_area_calculation (ε : ℝ) (h_ε : ε > 0) :
  ∃ (setup : DogHouseSetup),
    setup.side_length = 1.2 ∧
    setup.tie_point_distance = 0.3 ∧
    setup.chain_length = 3 ∧
    |dog_movement_area setup - 23.693| < ε :=
  sorry

end NUMINAMATH_CALUDE_dog_movement_area_calculation_l2367_236725


namespace NUMINAMATH_CALUDE_fence_area_calculation_l2367_236761

/-- The time (in hours) it takes the first painter to paint the entire fence alone -/
def painter1_time : ℝ := 12

/-- The time (in hours) it takes the second painter to paint the entire fence alone -/
def painter2_time : ℝ := 15

/-- The reduction in combined painting speed (in square feet per hour) when the painters work together -/
def speed_reduction : ℝ := 5

/-- The time (in hours) it takes both painters to paint the fence together -/
def combined_time : ℝ := 7

/-- The total area of the fence in square feet -/
def fence_area : ℝ := 700

theorem fence_area_calculation :
  (combined_time * (fence_area / painter1_time + fence_area / painter2_time - speed_reduction) = fence_area) :=
sorry

end NUMINAMATH_CALUDE_fence_area_calculation_l2367_236761


namespace NUMINAMATH_CALUDE_vector_scalar_mult_and_add_l2367_236773

theorem vector_scalar_mult_and_add :
  (3 : ℝ) • ((-3 : ℝ), (2 : ℝ), (-5 : ℝ)) + ((1 : ℝ), (7 : ℝ), (-3 : ℝ)) = ((-8 : ℝ), (13 : ℝ), (-18 : ℝ)) := by
  sorry

end NUMINAMATH_CALUDE_vector_scalar_mult_and_add_l2367_236773


namespace NUMINAMATH_CALUDE_min_sum_of_product_2004_l2367_236701

theorem min_sum_of_product_2004 (x y z : ℕ+) (h : x * y * z = 2004) :
  ∃ (a b c : ℕ+), a * b * c = 2004 ∧ a + b + c ≤ x + y + z ∧ a + b + c = 174 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_2004_l2367_236701


namespace NUMINAMATH_CALUDE_average_income_b_and_c_l2367_236788

/-- Proves that given the conditions, the average monthly income of B and C is 6250 --/
theorem average_income_b_and_c (income_a income_b income_c : ℝ) : 
  (income_a + income_b) / 2 = 5050 →
  (income_a + income_c) / 2 = 5200 →
  income_a = 4000 →
  (income_b + income_c) / 2 = 6250 := by
sorry

end NUMINAMATH_CALUDE_average_income_b_and_c_l2367_236788


namespace NUMINAMATH_CALUDE_stratified_sample_size_l2367_236755

/-- Represents the ratio of students in each grade -/
structure GradeRatio where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the sample size and number of third grade students in the sample -/
structure Sample where
  size : ℕ
  thirdGrade : ℕ

/-- Theorem stating the sample size given the conditions -/
theorem stratified_sample_size 
  (ratio : GradeRatio) 
  (sample : Sample) 
  (h1 : ratio.first = 4)
  (h2 : ratio.second = 3)
  (h3 : ratio.third = 2)
  (h4 : sample.thirdGrade = 10) :
  (ratio.third : ℚ) / (ratio.first + ratio.second + ratio.third : ℚ) = 
  (sample.thirdGrade : ℚ) / (sample.size : ℚ) → 
  sample.size = 45 := by
sorry

end NUMINAMATH_CALUDE_stratified_sample_size_l2367_236755


namespace NUMINAMATH_CALUDE_x_intercept_of_specific_line_l2367_236709

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line -/
def x_intercept (l : Line) : ℝ := sorry

/-- The line passing through (4, 6) and (8, 2) -/
def specific_line : Line := { x₁ := 4, y₁ := 6, x₂ := 8, y₂ := 2 }

theorem x_intercept_of_specific_line :
  x_intercept specific_line = 10 := by sorry

end NUMINAMATH_CALUDE_x_intercept_of_specific_line_l2367_236709


namespace NUMINAMATH_CALUDE_angle_construction_error_bound_l2367_236719

/-- Represents a 4-digit trigonometric table -/
structure TrigTable :=
  (sin : ℚ → ℚ)
  (cos : ℚ → ℚ)
  (precision : ℕ := 4)

/-- Represents the construction of a regular polygon -/
structure RegularPolygon :=
  (sides : ℕ)
  (centralAngle : ℚ)

/-- The error bound for angle construction using a 4-digit trig table -/
def angleErrorBound (p : RegularPolygon) (t : TrigTable) : ℚ := sorry

theorem angle_construction_error_bound 
  (p : RegularPolygon) 
  (t : TrigTable) 
  (h1 : p.sides = 18) 
  (h2 : p.centralAngle = 20) 
  (h3 : t.precision = 4) :
  angleErrorBound p t < 21 / 3600 := by sorry

end NUMINAMATH_CALUDE_angle_construction_error_bound_l2367_236719


namespace NUMINAMATH_CALUDE_product_inequality_l2367_236706

theorem product_inequality : 
  (190 * 80 = 19 * 800) → 
  (190 * 80 = 19 * 8 * 100) → 
  (19 * 8 * 10 ≠ 190 * 80) := by
sorry

end NUMINAMATH_CALUDE_product_inequality_l2367_236706


namespace NUMINAMATH_CALUDE_calculation_proof_l2367_236752

theorem calculation_proof : 5 * 7 * 11 + 21 / 7 - 3 = 385 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2367_236752


namespace NUMINAMATH_CALUDE_thanksgiving_turkey_cost_johns_thanksgiving_cost_l2367_236771

/-- Calculates the total cost of John's Thanksgiving turkey surprise for his employees. -/
theorem thanksgiving_turkey_cost 
  (num_employees : ℕ) 
  (turkey_cost : ℝ) 
  (discount_rate : ℝ) 
  (discount_threshold : ℕ) 
  (delivery_flat_fee : ℝ) 
  (delivery_per_turkey : ℝ) 
  (sales_tax_rate : ℝ) : ℝ :=
  let discounted_turkey_cost := 
    if num_employees > discount_threshold
    then num_employees * turkey_cost * (1 - discount_rate)
    else num_employees * turkey_cost
  let delivery_cost := delivery_flat_fee + num_employees * delivery_per_turkey
  let total_before_tax := discounted_turkey_cost + delivery_cost
  let total_cost := total_before_tax * (1 + sales_tax_rate)
  total_cost

/-- The total cost for John's Thanksgiving surprise is $2,188.35. -/
theorem johns_thanksgiving_cost :
  thanksgiving_turkey_cost 85 25 0.15 50 50 2 0.08 = 2188.35 := by
  sorry

end NUMINAMATH_CALUDE_thanksgiving_turkey_cost_johns_thanksgiving_cost_l2367_236771


namespace NUMINAMATH_CALUDE_stamp_collection_value_l2367_236729

/-- Given a collection of stamps with equal individual value, 
    calculate the total value of the collection. -/
theorem stamp_collection_value 
  (total_stamps : ℕ) 
  (sample_stamps : ℕ) 
  (sample_value : ℕ) 
  (h1 : total_stamps = 21)
  (h2 : sample_stamps = 7)
  (h3 : sample_value = 28) :
  (total_stamps : ℚ) * (sample_value : ℚ) / (sample_stamps : ℚ) = 84 := by
  sorry

end NUMINAMATH_CALUDE_stamp_collection_value_l2367_236729


namespace NUMINAMATH_CALUDE_triangle_area_l2367_236764

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  A + B + C = π →
  a = 2 * Real.sqrt 3 →
  b + c = 4 →
  Real.cos B * Real.cos C - Real.sin B * Real.sin C = 1/2 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2367_236764


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l2367_236769

/-- The distance from point A(-2, 1) to the y-axis is 2 -/
theorem distance_to_y_axis : 
  let A : ℝ × ℝ := (-2, 1)
  abs A.1 = 2 := by sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l2367_236769


namespace NUMINAMATH_CALUDE_zoo_sandwiches_l2367_236723

theorem zoo_sandwiches (people : ℝ) (sandwiches_per_person : ℝ) :
  people = 219.0 →
  sandwiches_per_person = 3.0 →
  people * sandwiches_per_person = 657.0 := by
  sorry

end NUMINAMATH_CALUDE_zoo_sandwiches_l2367_236723


namespace NUMINAMATH_CALUDE_polynomial_root_product_l2367_236794

theorem polynomial_root_product (d e : ℝ) : 
  (∀ x : ℝ, x^2 + d*x + e = 0 ↔ x = Real.cos (π/9) ∨ x = Real.cos (2*π/9)) →
  d * e = -5/64 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_product_l2367_236794


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_roots_l2367_236754

theorem sum_of_reciprocals_roots (x : ℝ) : 
  (x^2 - 13*x + 4 = 0) → 
  (∃ r₁ r₂ : ℝ, r₁ ≠ r₂ ∧ x^2 - 13*x + 4 = (x - r₁) * (x - r₂) ∧ 
    (1 / r₁ + 1 / r₂ = 13 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_roots_l2367_236754


namespace NUMINAMATH_CALUDE_sallys_quarters_l2367_236703

/-- Given that Sally had 760 quarters initially and spent 418 quarters,
    prove that she now has 342 quarters. -/
theorem sallys_quarters (initial : ℕ) (spent : ℕ) (remaining : ℕ) 
    (h1 : initial = 760)
    (h2 : spent = 418)
    (h3 : remaining = initial - spent) :
  remaining = 342 := by
  sorry

end NUMINAMATH_CALUDE_sallys_quarters_l2367_236703


namespace NUMINAMATH_CALUDE_square_area_error_percentage_l2367_236796

theorem square_area_error_percentage (s : ℝ) (h : s > 0) :
  let measured_side := s * 1.01
  let actual_area := s^2
  let calculated_area := measured_side^2
  let area_error := calculated_area - actual_area
  let error_percentage := (area_error / actual_area) * 100
  error_percentage = 2.01 := by
sorry


end NUMINAMATH_CALUDE_square_area_error_percentage_l2367_236796


namespace NUMINAMATH_CALUDE_cubic_foot_to_cubic_inch_l2367_236797

theorem cubic_foot_to_cubic_inch :
  (1 : ℝ) * (foot ^ 3) = 1728 * (inch ^ 3) :=
by
  -- Define the relationship between foot and inch
  have foot_to_inch : (1 : ℝ) * foot = 12 * inch := sorry
  
  -- Cube both sides of the equation
  have cubed_equality : ((1 : ℝ) * foot) ^ 3 = (12 * inch) ^ 3 := sorry
  
  -- Simplify the left side
  have left_side : ((1 : ℝ) * foot) ^ 3 = (1 : ℝ) * (foot ^ 3) := sorry
  
  -- Simplify the right side
  have right_side : (12 * inch) ^ 3 = 1728 * (inch ^ 3) := sorry
  
  -- Combine the steps to prove the theorem
  sorry

end NUMINAMATH_CALUDE_cubic_foot_to_cubic_inch_l2367_236797


namespace NUMINAMATH_CALUDE_fraction_irreducible_l2367_236785

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducible_l2367_236785


namespace NUMINAMATH_CALUDE_rachel_songs_total_l2367_236739

theorem rachel_songs_total (albums : ℕ) (songs_per_album : ℕ) (h1 : albums = 8) (h2 : songs_per_album = 2) :
  albums * songs_per_album = 16 := by
  sorry

end NUMINAMATH_CALUDE_rachel_songs_total_l2367_236739


namespace NUMINAMATH_CALUDE_max_boxes_A_l2367_236718

def price_A : ℝ := 24
def price_B : ℝ := 16
def total_boxes : ℕ := 200
def max_cost : ℝ := 3920

theorem max_boxes_A : 
  price_A + 2 * price_B = 56 →
  2 * price_A + price_B = 64 →
  (∀ m : ℕ, m ≤ total_boxes → 
    price_A * m + price_B * (total_boxes - m) ≤ max_cost →
    m ≤ 90) ∧
  (∃ m : ℕ, m = 90 ∧ 
    price_A * m + price_B * (total_boxes - m) ≤ max_cost) :=
by sorry

end NUMINAMATH_CALUDE_max_boxes_A_l2367_236718


namespace NUMINAMATH_CALUDE_johns_age_l2367_236737

theorem johns_age (john : ℕ) (matt : ℕ) : 
  matt = 4 * john - 3 → 
  john + matt = 52 → 
  john = 11 := by
sorry

end NUMINAMATH_CALUDE_johns_age_l2367_236737


namespace NUMINAMATH_CALUDE_jamal_has_one_black_marble_l2367_236750

/-- Represents the bag of marbles with different colors. -/
structure MarbleBag where
  yellow : ℕ
  blue : ℕ
  green : ℕ
  black : ℕ

/-- The probability of drawing a black marble. -/
def blackProbability : ℚ := 1 / 28

/-- Jamal's bag of marbles. -/
def jamalsBag : MarbleBag := {
  yellow := 12,
  blue := 10,
  green := 5,
  black := 1  -- We'll prove this is correct
}

/-- The total number of marbles in the bag. -/
def totalMarbles (bag : MarbleBag) : ℕ :=
  bag.yellow + bag.blue + bag.green + bag.black

/-- Theorem stating that Jamal's bag contains exactly one black marble. -/
theorem jamal_has_one_black_marble :
  jamalsBag.black = 1 ∧
  (jamalsBag.black : ℚ) / (totalMarbles jamalsBag : ℚ) = blackProbability :=
by sorry

end NUMINAMATH_CALUDE_jamal_has_one_black_marble_l2367_236750


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2367_236745

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x, (1 - 2*x)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ = 510 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2367_236745


namespace NUMINAMATH_CALUDE_fertilizer_production_l2367_236770

/-- 
Given:
- m: Initial production in the first quarter (in tons)
- x: Percentage increase in production each quarter (as a decimal)
- n: Production in the third quarter (in tons)

Prove that the production in the third quarter (n) is equal to the initial production (m) 
multiplied by (1 + x)^2.
-/
theorem fertilizer_production (m n : ℝ) (x : ℝ) (h_positive : 0 < x) : 
  m * (1 + x)^2 = n → True :=
by
  sorry

end NUMINAMATH_CALUDE_fertilizer_production_l2367_236770


namespace NUMINAMATH_CALUDE_ellipse_center_locus_l2367_236790

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents an ellipse -/
structure Ellipse where
  center : Point
  semi_major_axis : ℝ
  semi_minor_axis : ℝ

/-- Represents a right angle -/
structure RightAngle where
  vertex : Point

/-- Predicate to check if an ellipse touches both sides of a right angle -/
def touches_right_angle (e : Ellipse) (ra : RightAngle) : Prop :=
  sorry

/-- The locus of the center of the ellipse -/
def center_locus (ra : RightAngle) (a b : ℝ) : Set Point :=
  {p : Point | ∃ e : Ellipse, e.center = p ∧ e.semi_major_axis = a ∧ e.semi_minor_axis = b ∧ touches_right_angle e ra}

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if a point is on an arc of a circle -/
def on_circle_arc (p : Point) (c : Circle) : Prop :=
  sorry

theorem ellipse_center_locus (ra : RightAngle) (a b : ℝ) :
  ∃ c : Circle, c.center = ra.vertex ∧ ∀ p ∈ center_locus ra a b, on_circle_arc p c :=
sorry

end NUMINAMATH_CALUDE_ellipse_center_locus_l2367_236790


namespace NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l2367_236756

theorem tan_eleven_pi_fourths : Real.tan (11 * π / 4) = -1 := by sorry

end NUMINAMATH_CALUDE_tan_eleven_pi_fourths_l2367_236756


namespace NUMINAMATH_CALUDE_volcano_eruption_percentage_l2367_236783

theorem volcano_eruption_percentage (total_volcanoes : ℕ) 
  (intact_volcanoes : ℕ) (mid_year_percentage : ℝ) 
  (end_year_percentage : ℝ) :
  total_volcanoes = 200 →
  intact_volcanoes = 48 →
  mid_year_percentage = 0.4 →
  end_year_percentage = 0.5 →
  ∃ (x : ℝ),
    x ≥ 0 ∧ x ≤ 100 ∧
    (total_volcanoes : ℝ) * (1 - x / 100) * (1 - mid_year_percentage) * (1 - end_year_percentage) = intact_volcanoes ∧
    x = 20 := by
  sorry

end NUMINAMATH_CALUDE_volcano_eruption_percentage_l2367_236783


namespace NUMINAMATH_CALUDE_composition_ratio_l2367_236708

def f (x : ℝ) : ℝ := 3 * x + 5

def g (x : ℝ) : ℝ := 4 * x - 3

theorem composition_ratio :
  (f (g (f (g 3)))) / (g (f (g (f 3)))) = 380 / 653 := by
  sorry

end NUMINAMATH_CALUDE_composition_ratio_l2367_236708


namespace NUMINAMATH_CALUDE_correct_mean_after_error_correction_l2367_236760

theorem correct_mean_after_error_correction (n : ℕ) (incorrect_mean correct_value incorrect_value : ℝ) :
  n = 30 →
  incorrect_mean = 250 →
  correct_value = 165 →
  incorrect_value = 135 →
  (n : ℝ) * incorrect_mean + (correct_value - incorrect_value) = n * 251 := by
  sorry

end NUMINAMATH_CALUDE_correct_mean_after_error_correction_l2367_236760


namespace NUMINAMATH_CALUDE_exact_division_condition_l2367_236727

-- Define the polynomial x^4 + 1
def f (x : ℂ) : ℂ := x^4 + 1

-- Define the trinomial x^2 + px + q
def g (p q x : ℂ) : ℂ := x^2 + p*x + q

-- Define the condition for exact division
def is_exact_division (p q : ℂ) : Prop :=
  ∃ (h : ℂ → ℂ), ∀ x, f x = (g p q x) * (h x)

-- State the theorem
theorem exact_division_condition :
  ∀ p q : ℂ, is_exact_division p q ↔ 
    ((p = 0 ∧ q = Complex.I) ∨ 
     (p = 0 ∧ q = -Complex.I) ∨ 
     (p = Real.sqrt 2 ∧ q = 1) ∨ 
     (p = -Real.sqrt 2 ∧ q = 1)) :=
by sorry

end NUMINAMATH_CALUDE_exact_division_condition_l2367_236727


namespace NUMINAMATH_CALUDE_gcf_180_270_l2367_236776

theorem gcf_180_270 : Nat.gcd 180 270 = 90 := by
  sorry

end NUMINAMATH_CALUDE_gcf_180_270_l2367_236776


namespace NUMINAMATH_CALUDE_algebraic_equality_l2367_236714

theorem algebraic_equality (m n : ℝ) : 4*m + 2*n - (n - m) = 5*m + n := by
  sorry

end NUMINAMATH_CALUDE_algebraic_equality_l2367_236714


namespace NUMINAMATH_CALUDE_unique_integer_function_l2367_236720

def IntegerFunction (f : ℤ → ℚ) : Prop :=
  ∀ (x y z : ℤ), 
    (∀ (c : ℚ), f x < c ∧ c < f y → ∃ (w : ℤ), f w = c) ∧
    (x + y + z = 0 → f x + f y + f z = f x * f y * f z)

theorem unique_integer_function : 
  ∃! (f : ℤ → ℚ), IntegerFunction f ∧ (∀ x : ℤ, f x = 0) :=
sorry

end NUMINAMATH_CALUDE_unique_integer_function_l2367_236720


namespace NUMINAMATH_CALUDE_smallest_sum_with_factors_and_perfect_square_l2367_236793

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_sum_with_factors_and_perfect_square :
  ∃ (a b : ℕ+),
    num_factors a = 15 ∧
    num_factors b = 20 ∧
    is_perfect_square (a.val + b.val) ∧
    ∀ (c d : ℕ+),
      num_factors c = 15 →
      num_factors d = 20 →
      is_perfect_square (c.val + d.val) →
      a.val + b.val ≤ c.val + d.val ∧
      a.val + b.val = 576 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_with_factors_and_perfect_square_l2367_236793


namespace NUMINAMATH_CALUDE_brand_preference_survey_l2367_236732

theorem brand_preference_survey (total : ℕ) (ratio : ℚ) (brand_x : ℕ) : 
  total = 250 → 
  ratio = 4/1 → 
  brand_x = total * (ratio / (1 + ratio)) → 
  brand_x = 200 := by
sorry

end NUMINAMATH_CALUDE_brand_preference_survey_l2367_236732


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2367_236757

theorem polynomial_expansion (x : ℝ) : 
  (7 * x^2 + 5 - 3 * x) * (4 * x^3) = 28 * x^5 - 12 * x^4 + 20 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2367_236757


namespace NUMINAMATH_CALUDE_second_number_value_l2367_236722

theorem second_number_value (x y : ℚ) 
  (h1 : (1 : ℚ) / 5 * x = (5 : ℚ) / 8 * y) 
  (h2 : x + 35 = 4 * y) : 
  y = 40 := by
sorry

end NUMINAMATH_CALUDE_second_number_value_l2367_236722


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2367_236715

theorem geometric_sequence_sum (a q : ℝ) (h1 : a + a * q = 7) (h2 : a * (q^6 - 1) / (q - 1) = 91) :
  a * (1 + q + q^2 + q^3) = 28 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2367_236715


namespace NUMINAMATH_CALUDE_rower_downstream_speed_l2367_236778

/-- Calculates the downstream speed of a rower given their upstream speed and still water speed. -/
def downstream_speed (upstream_speed still_water_speed : ℝ) : ℝ :=
  2 * still_water_speed - upstream_speed

/-- Theorem stating that a rower with an upstream speed of 12 kmph and a still water speed of 25 kmph
    will have a downstream speed of 38 kmph. -/
theorem rower_downstream_speed :
  downstream_speed 12 25 = 38 := by
  sorry

end NUMINAMATH_CALUDE_rower_downstream_speed_l2367_236778


namespace NUMINAMATH_CALUDE_decimal_51_to_binary_l2367_236733

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec to_binary_aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else to_binary_aux (m / 2) ((m % 2) :: acc)
    to_binary_aux n []

theorem decimal_51_to_binary :
  decimal_to_binary 51 = [1, 1, 0, 0, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_decimal_51_to_binary_l2367_236733
