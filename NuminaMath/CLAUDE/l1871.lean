import Mathlib

namespace NUMINAMATH_CALUDE_cos_supplementary_angles_l1871_187174

theorem cos_supplementary_angles (α β : Real) (h : α + β = Real.pi) : 
  Real.cos α = Real.cos β := by
  sorry

end NUMINAMATH_CALUDE_cos_supplementary_angles_l1871_187174


namespace NUMINAMATH_CALUDE_frog_climb_time_l1871_187178

/-- Represents the frog's climbing problem -/
structure FrogClimb where
  well_depth : ℕ
  climb_distance : ℕ
  slip_distance : ℕ
  climb_time : ℚ
  slip_time : ℚ
  intermediate_time : ℕ
  intermediate_distance : ℕ

/-- Theorem stating the time taken for the frog to climb out of the well -/
theorem frog_climb_time (f : FrogClimb)
  (h1 : f.well_depth = 12)
  (h2 : f.climb_distance = 3)
  (h3 : f.slip_distance = 1)
  (h4 : f.slip_time = f.climb_time / 3)
  (h5 : f.intermediate_time = 17)
  (h6 : f.intermediate_distance = f.well_depth - 3)
  (h7 : f.climb_time = 1) :
  ∃ (total_time : ℕ), total_time = 22 ∧ 
  (∃ (cycles : ℕ), 
    cycles * (f.climb_distance - f.slip_distance) + 
    (total_time - cycles * (f.climb_time + f.slip_time)) * f.climb_distance / f.climb_time = f.well_depth) :=
sorry

end NUMINAMATH_CALUDE_frog_climb_time_l1871_187178


namespace NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l1871_187173

-- Define the quadratic equations
def equation1 (x : ℝ) : Prop := 2 * x^2 - 5 * x + 1 = 0
def equation2 (x : ℝ) : Prop := (2 * x - 1)^2 - x^2 = 0

-- Theorem for the solutions of equation1
theorem solutions_equation1 :
  ∃ x₁ x₂ : ℝ, x₁ = (5 + Real.sqrt 17) / 4 ∧
              x₂ = (5 - Real.sqrt 17) / 4 ∧
              equation1 x₁ ∧
              equation1 x₂ ∧
              ∀ x : ℝ, equation1 x → (x = x₁ ∨ x = x₂) :=
sorry

-- Theorem for the solutions of equation2
theorem solutions_equation2 :
  ∃ x₁ x₂ : ℝ, x₁ = 1/3 ∧
              x₂ = 1 ∧
              equation2 x₁ ∧
              equation2 x₂ ∧
              ∀ x : ℝ, equation2 x → (x = x₁ ∨ x = x₂) :=
sorry

end NUMINAMATH_CALUDE_solutions_equation1_solutions_equation2_l1871_187173


namespace NUMINAMATH_CALUDE_rabbits_eaten_potatoes_l1871_187191

/-- The number of potatoes eaten by rabbits -/
def potatoesEaten (initial remaining : ℕ) : ℕ := initial - remaining

/-- Theorem: The number of potatoes eaten by rabbits is equal to the difference
    between the initial number of potatoes and the remaining number of potatoes -/
theorem rabbits_eaten_potatoes (initial remaining : ℕ) (h : remaining ≤ initial) :
  potatoesEaten initial remaining = initial - remaining := by
  sorry

#eval potatoesEaten 8 5  -- Should evaluate to 3

end NUMINAMATH_CALUDE_rabbits_eaten_potatoes_l1871_187191


namespace NUMINAMATH_CALUDE_intersection_point_exists_in_interval_l1871_187195

theorem intersection_point_exists_in_interval :
  ∃! x : ℝ, 3 < x ∧ x < 4 ∧ Real.log x = 7 - 2 * x := by sorry

end NUMINAMATH_CALUDE_intersection_point_exists_in_interval_l1871_187195


namespace NUMINAMATH_CALUDE_function_lower_bound_l1871_187135

open Real

theorem function_lower_bound (a x : ℝ) (ha : a > 0) : 
  let f : ℝ → ℝ := λ x => a * (exp x + a) - x
  f x > 2 * log a + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_function_lower_bound_l1871_187135


namespace NUMINAMATH_CALUDE_square_area_proof_l1871_187150

theorem square_area_proof (x : ℝ) : 
  (6 * x - 27 = 30 - 2 * x) → 
  ((6 * x - 27) * (6 * x - 27) = 248.0625) := by
  sorry

end NUMINAMATH_CALUDE_square_area_proof_l1871_187150


namespace NUMINAMATH_CALUDE_equal_bills_l1871_187169

/-- Given Linda's and Mark's tips and tip percentages, prove their bills are equal -/
theorem equal_bills (linda_tip mark_tip : ℝ) (linda_percent mark_percent : ℝ) 
  (h1 : linda_tip = 5)
  (h2 : mark_tip = 3)
  (h3 : linda_percent = 0.25)
  (h4 : mark_percent = 0.15)
  (h5 : linda_tip = linda_percent * (linda_tip / linda_percent))
  (h6 : mark_tip = mark_percent * (mark_tip / mark_percent)) :
  linda_tip / linda_percent = mark_tip / mark_percent := by
  sorry

end NUMINAMATH_CALUDE_equal_bills_l1871_187169


namespace NUMINAMATH_CALUDE_product_mod_500_l1871_187176

theorem product_mod_500 : (2367 * 1023) % 500 = 41 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_500_l1871_187176


namespace NUMINAMATH_CALUDE_min_sum_of_product_l1871_187106

theorem min_sum_of_product (a b : ℤ) (h : a * b = 144) : 
  ∀ x y : ℤ, x * y = 144 → a + b ≤ x + y ∧ ∃ a₀ b₀ : ℤ, a₀ * b₀ = 144 ∧ a₀ + b₀ = -145 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l1871_187106


namespace NUMINAMATH_CALUDE_not_perfect_square_l1871_187165

theorem not_perfect_square : 
  (∃ x : ℕ, 6^2024 = x^2) ∧ 
  (∀ y : ℕ, 7^2025 ≠ y^2) ∧ 
  (∃ z : ℕ, 8^2026 = z^2) ∧ 
  (∃ w : ℕ, 9^2027 = w^2) ∧ 
  (∃ v : ℕ, 10^2028 = v^2) := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_l1871_187165


namespace NUMINAMATH_CALUDE_square_area_decrease_l1871_187175

theorem square_area_decrease (areaI areaIII areaII : ℝ) (decrease_percent : ℝ) :
  areaI = 18 * Real.sqrt 3 →
  areaIII = 50 * Real.sqrt 3 →
  areaII = 72 →
  decrease_percent = 20 →
  let side_length := Real.sqrt areaII
  let new_side_length := side_length * (1 - decrease_percent / 100)
  let new_area := new_side_length ^ 2
  (areaII - new_area) / areaII * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_square_area_decrease_l1871_187175


namespace NUMINAMATH_CALUDE_certain_number_problem_l1871_187198

theorem certain_number_problem (x : ℝ) (h : x + 33 + 333 + 33.3 = 399.6) : x = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1871_187198


namespace NUMINAMATH_CALUDE_polynomial_expansion_equality_l1871_187127

/-- Proves that the expansion of (3y-2)*(5y^12+3y^11+5y^10+3y^9) equals 15y^13 - y^12 + 9y^11 - y^10 + 6y^9 for all real y. -/
theorem polynomial_expansion_equality (y : ℝ) : 
  (3*y - 2) * (5*y^12 + 3*y^11 + 5*y^10 + 3*y^9) = 
  15*y^13 - y^12 + 9*y^11 - y^10 + 6*y^9 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_equality_l1871_187127


namespace NUMINAMATH_CALUDE_decimal_difference_l1871_187199

-- Define the repeating decimal 0.72̄
def repeating_decimal : ℚ := 72 / 99

-- Define the terminating decimal 0.726
def terminating_decimal : ℚ := 726 / 1000

-- Theorem statement
theorem decimal_difference :
  repeating_decimal - terminating_decimal = 14 / 11000 := by
  sorry

end NUMINAMATH_CALUDE_decimal_difference_l1871_187199


namespace NUMINAMATH_CALUDE_houses_per_block_l1871_187196

/-- Given that each block receives 32 pieces of junk mail and each house in a block receives 8 pieces of mail, 
    prove that the number of houses in a block is 4. -/
theorem houses_per_block (mail_per_block : ℕ) (mail_per_house : ℕ) (h1 : mail_per_block = 32) (h2 : mail_per_house = 8) :
  mail_per_block / mail_per_house = 4 := by
  sorry

end NUMINAMATH_CALUDE_houses_per_block_l1871_187196


namespace NUMINAMATH_CALUDE_total_donation_is_375_l1871_187128

/- Define the donation amounts for each company -/
def foster_farms_donation : ℕ := 45
def american_summits_donation : ℕ := 2 * foster_farms_donation
def hormel_donation : ℕ := 3 * foster_farms_donation
def boudin_butchers_donation : ℕ := hormel_donation / 3
def del_monte_foods_donation : ℕ := american_summits_donation - 30

/- Define the total donation -/
def total_donation : ℕ := 
  foster_farms_donation + 
  american_summits_donation + 
  hormel_donation + 
  boudin_butchers_donation + 
  del_monte_foods_donation

/- Theorem stating that the total donation is 375 -/
theorem total_donation_is_375 : total_donation = 375 := by
  sorry

end NUMINAMATH_CALUDE_total_donation_is_375_l1871_187128


namespace NUMINAMATH_CALUDE_absolute_value_implication_l1871_187193

theorem absolute_value_implication (x : ℝ) : |x - 1| < 2 → x < 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_implication_l1871_187193


namespace NUMINAMATH_CALUDE_slide_boys_count_l1871_187143

/-- The number of boys who went down the slide initially -/
def initial_boys : ℕ := 22

/-- The number of additional boys who went down the slide -/
def additional_boys : ℕ := 13

/-- The total number of boys who went down the slide -/
def total_boys : ℕ := initial_boys + additional_boys

theorem slide_boys_count : total_boys = 35 := by
  sorry

end NUMINAMATH_CALUDE_slide_boys_count_l1871_187143


namespace NUMINAMATH_CALUDE_arccos_sqrt2_over_2_l1871_187157

theorem arccos_sqrt2_over_2 : Real.arccos (Real.sqrt 2 / 2) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arccos_sqrt2_over_2_l1871_187157


namespace NUMINAMATH_CALUDE_greater_eighteen_league_games_l1871_187131

/-- Represents a hockey league with the given specifications -/
structure HockeyLeague where
  divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculates the total number of games in the hockey league -/
def total_games (league : HockeyLeague) : Nat :=
  let total_teams := league.divisions * league.teams_per_division
  let games_per_team := (league.teams_per_division - 1) * league.intra_division_games + 
                        (total_teams - league.teams_per_division) * league.inter_division_games
  total_teams * games_per_team / 2

/-- Theorem stating that the total number of games in the specified league is 351 -/
theorem greater_eighteen_league_games : 
  total_games { divisions := 3
              , teams_per_division := 6
              , intra_division_games := 3
              , inter_division_games := 2 } = 351 := by
  sorry

end NUMINAMATH_CALUDE_greater_eighteen_league_games_l1871_187131


namespace NUMINAMATH_CALUDE_sum_equals_fourteen_thousand_minus_m_l1871_187148

theorem sum_equals_fourteen_thousand_minus_m (M : ℕ) : 
  1989 + 1991 + 1993 + 1995 + 1997 + 1999 + 2001 = 14000 - M → M = 35 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_fourteen_thousand_minus_m_l1871_187148


namespace NUMINAMATH_CALUDE_perpendicular_bisector_is_diameter_l1871_187184

/-- A circle in a plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A chord of a circle. -/
structure Chord (c : Circle) where
  endpoint1 : ℝ × ℝ
  endpoint2 : ℝ × ℝ

/-- A line in a plane. -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Predicate to check if a line is perpendicular to a chord. -/
def isPerpendicular (l : Line) (ch : Chord c) : Prop := sorry

/-- Predicate to check if a line bisects a chord. -/
def bisectsChord (l : Line) (ch : Chord c) : Prop := sorry

/-- Predicate to check if a line bisects the arcs subtended by a chord. -/
def bisectsArcs (l : Line) (ch : Chord c) : Prop := sorry

/-- Predicate to check if a line is a diameter of a circle. -/
def isDiameter (l : Line) (c : Circle) : Prop := sorry

/-- Theorem: A line perpendicular to a chord that bisects the chord and the arcs
    subtended by the chord is a diameter of the circle. -/
theorem perpendicular_bisector_is_diameter
  (c : Circle) (ch : Chord c) (l : Line)
  (h1 : isPerpendicular l ch)
  (h2 : bisectsChord l ch)
  (h3 : bisectsArcs l ch) :
  isDiameter l c := by sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_is_diameter_l1871_187184


namespace NUMINAMATH_CALUDE_antonella_toonies_l1871_187154

/-- Represents the number of coins of each type -/
structure CoinCount where
  loonies : ℕ
  toonies : ℕ

/-- Calculates the total value of coins in dollars -/
def totalValue (coins : CoinCount) : ℕ :=
  coins.loonies + 2 * coins.toonies

/-- Represents Antonella's coin situation -/
def antonellasCoins (coins : CoinCount) : Prop :=
  coins.loonies + coins.toonies = 10 ∧
  totalValue coins = 14

theorem antonella_toonies :
  ∃ (coins : CoinCount), antonellasCoins coins ∧ coins.toonies = 4 := by
  sorry

end NUMINAMATH_CALUDE_antonella_toonies_l1871_187154


namespace NUMINAMATH_CALUDE_eight_people_lineup_permutations_l1871_187156

theorem eight_people_lineup_permutations : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_eight_people_lineup_permutations_l1871_187156


namespace NUMINAMATH_CALUDE_jane_inspected_five_eighths_l1871_187101

/-- Represents the fraction of products inspected by Jane given the total rejection rate,
    John's rejection rate, and Jane's rejection rate. -/
def jane_inspection_fraction (total_rejection_rate john_rejection_rate jane_rejection_rate : ℚ) : ℚ :=
  5 / 8

/-- Theorem stating that given the specified rejection rates, Jane inspected 5/8 of the products. -/
theorem jane_inspected_five_eighths
  (total_rejection_rate : ℚ)
  (john_rejection_rate : ℚ)
  (jane_rejection_rate : ℚ)
  (h_total : total_rejection_rate = 75 / 10000)
  (h_john : john_rejection_rate = 5 / 1000)
  (h_jane : jane_rejection_rate = 9 / 1000) :
  jane_inspection_fraction total_rejection_rate john_rejection_rate jane_rejection_rate = 5 / 8 := by
  sorry

#eval jane_inspection_fraction (75/10000) (5/1000) (9/1000)

end NUMINAMATH_CALUDE_jane_inspected_five_eighths_l1871_187101


namespace NUMINAMATH_CALUDE_square_coverage_l1871_187192

theorem square_coverage (k n : ℕ) : k > 1 → (k ^ 2 = 2 ^ (n + 1) * n + 1) → (k = 7 ∧ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_square_coverage_l1871_187192


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_fifth_minus_n_l1871_187141

theorem largest_divisor_of_n_fifth_minus_n (n : ℤ) : 
  ∃ (d : ℕ), d = 30 ∧ (∀ (m : ℤ), (m^5 - m) % d = 0) ∧ 
  (∀ (k : ℕ), k > d → ∃ (l : ℤ), (l^5 - l) % k ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_fifth_minus_n_l1871_187141


namespace NUMINAMATH_CALUDE_unique_prime_twice_squares_l1871_187117

theorem unique_prime_twice_squares : 
  ∃! (p : ℕ), 
    Prime p ∧ 
    (∃ (x : ℕ), p + 1 = 2 * x^2) ∧ 
    (∃ (y : ℕ), p^2 + 1 = 2 * y^2) ∧ 
    p = 7 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_twice_squares_l1871_187117


namespace NUMINAMATH_CALUDE_bird_population_theorem_l1871_187181

/-- 
Given a population of birds consisting of robins and bluejays, 
if 1/3 of robins are female, 2/3 of bluejays are female, 
and the overall fraction of male birds is 7/15, 
then the fraction of birds that are robins is 2/5.
-/
theorem bird_population_theorem (total_birds : ℕ) (robins : ℕ) (bluejays : ℕ) 
  (h1 : robins + bluejays = total_birds)
  (h2 : (2 : ℚ) / 3 * robins + (1 : ℚ) / 3 * bluejays = (7 : ℚ) / 15 * total_birds) :
  (robins : ℚ) / total_birds = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_bird_population_theorem_l1871_187181


namespace NUMINAMATH_CALUDE_simplify_and_sum_exponents_l1871_187160

variables (a b c : ℝ)

theorem simplify_and_sum_exponents :
  ∃ (x y z : ℕ) (w : ℝ),
    (40 * a^7 * b^9 * c^14)^(1/3) = 2 * a^x * b^y * c^z * w^(1/3) ∧
    w = 5 * a * c^2 ∧
    x + y + z = 9 := by sorry

end NUMINAMATH_CALUDE_simplify_and_sum_exponents_l1871_187160


namespace NUMINAMATH_CALUDE_expand_and_simplify_l1871_187164

theorem expand_and_simplify (x : ℝ) : (2*x + 6)*(x + 10) = 2*x^2 + 26*x + 60 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l1871_187164


namespace NUMINAMATH_CALUDE_set_b_forms_triangle_l1871_187111

/-- Triangle Inequality Theorem: The sum of the lengths of any two sides of a triangle 
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function to check if three line segments can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set of line segments (5, 6, 10) can form a triangle -/
theorem set_b_forms_triangle : can_form_triangle 5 6 10 := by
  sorry


end NUMINAMATH_CALUDE_set_b_forms_triangle_l1871_187111


namespace NUMINAMATH_CALUDE_money_distribution_l1871_187166

/-- Given a sum of money distributed among four people in a specific proportion,
    where one person receives a fixed amount more than another,
    prove that a particular person's share is as stated. -/
theorem money_distribution (total : ℝ) (a b c d : ℝ) : 
  a + b + c + d = total →
  5 * b = 3 * a →
  5 * c = 2 * a →
  5 * d = 3 * a →
  a = b + 1000 →
  c = 1000 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l1871_187166


namespace NUMINAMATH_CALUDE_complex_equation_sum_l1871_187162

theorem complex_equation_sum (a b : ℝ) :
  (3 + b * I) / (1 - I) = a + b * I → a + b = 3 :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l1871_187162


namespace NUMINAMATH_CALUDE_odd_function_conditions_l1871_187151

noncomputable def f (a b x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + a)

theorem odd_function_conditions (a b : ℝ) :
  (∀ x, f a b x = -f a b (-x)) →
  (a = 2 ∧ b = 1 ∧
   ∀ x y, x < y → f 2 1 x > f 2 1 y ∧
   ∀ t k, f 2 1 (t^2 - 2*t) + f 2 1 (2*t^2 - k) < 0 → k < -1/3) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_conditions_l1871_187151


namespace NUMINAMATH_CALUDE_school_route_time_difference_l1871_187163

theorem school_route_time_difference :
  let first_route_uphill_time : ℕ := 6
  let first_route_path_time : ℕ := 2 * first_route_uphill_time
  let first_route_first_two_stages : ℕ := first_route_uphill_time + first_route_path_time
  let first_route_final_time : ℕ := first_route_first_two_stages / 3
  let first_route_total_time : ℕ := first_route_first_two_stages + first_route_final_time

  let second_route_flat_time : ℕ := 14
  let second_route_final_time : ℕ := 2 * second_route_flat_time
  let second_route_total_time : ℕ := second_route_flat_time + second_route_final_time

  second_route_total_time - first_route_total_time = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_school_route_time_difference_l1871_187163


namespace NUMINAMATH_CALUDE_map_distance_theorem_l1871_187183

/-- Represents the scale of a map in feet per inch -/
def map_scale : ℝ := 700

/-- Represents the length of a line on the map in inches -/
def map_line_length : ℝ := 5.5

/-- Calculates the actual distance represented by a line on the map -/
def actual_distance (scale : ℝ) (map_length : ℝ) : ℝ :=
  scale * map_length

/-- Proves that a 5.5-inch line on a map with a scale of 1 inch = 700 feet 
    represents 3850 feet in reality -/
theorem map_distance_theorem : 
  actual_distance map_scale map_line_length = 3850 := by
  sorry

end NUMINAMATH_CALUDE_map_distance_theorem_l1871_187183


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1871_187120

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n, p n) ↔ (∀ n, ¬ p n) :=
by sorry

theorem negation_of_proposition :
  (¬ ∃ n : ℕ, n^2 > 4^n) ↔ (∀ n : ℕ, n^2 ≤ 4^n) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_proposition_l1871_187120


namespace NUMINAMATH_CALUDE_jim_total_cars_l1871_187189

/-- The number of model cars Jim has -/
structure ModelCars where
  buicks : ℕ
  fords : ℕ
  chevys : ℕ

/-- Jim's collection of model cars satisfying the given conditions -/
def jim_collection : ModelCars :=
  { buicks := 220,
    fords := 55,
    chevys := 26 }

/-- Theorem stating the total number of model cars Jim has -/
theorem jim_total_cars :
  jim_collection.buicks = 220 ∧
  jim_collection.buicks = 4 * jim_collection.fords ∧
  jim_collection.fords = 2 * jim_collection.chevys + 3 →
  jim_collection.buicks + jim_collection.fords + jim_collection.chevys = 301 := by
  sorry

#eval jim_collection.buicks + jim_collection.fords + jim_collection.chevys

end NUMINAMATH_CALUDE_jim_total_cars_l1871_187189


namespace NUMINAMATH_CALUDE_student_ticket_price_l1871_187139

/-- Calculates the price of a student ticket given the total number of tickets sold,
    the total amount collected, the price of an adult ticket, and the number of student tickets sold. -/
theorem student_ticket_price
  (total_tickets : ℕ)
  (total_amount : ℚ)
  (adult_price : ℚ)
  (student_tickets : ℕ)
  (h1 : total_tickets = 59)
  (h2 : total_amount = 222.5)
  (h3 : adult_price = 4)
  (h4 : student_tickets = 9) :
  (total_amount - (adult_price * (total_tickets - student_tickets))) / student_tickets = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_student_ticket_price_l1871_187139


namespace NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l1871_187179

/-- An isosceles triangle with two angles in the ratio 1:4 has a vertex angle of either 20 or 120 degrees. -/
theorem isosceles_triangle_vertex_angle (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Angles are positive
  a + b + c = 180 →  -- Sum of angles in a triangle
  a = b →  -- Isosceles triangle condition
  (c = a ∧ b = 4 * a) ∨ (a = 4 * c ∧ b = 4 * c) →  -- Ratio condition
  c = 20 ∨ c = 120 := by
sorry


end NUMINAMATH_CALUDE_isosceles_triangle_vertex_angle_l1871_187179


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l1871_187136

theorem quadratic_roots_properties (r1 r2 : ℝ) : 
  r1 ≠ r2 → 
  r1^2 - 5*r1 + 6 = 0 → 
  r2^2 - 5*r2 + 6 = 0 → 
  (|r1 + r2| ≤ 6) ∧ 
  (|r1 * r2| ≤ 3 ∨ |r1 * r2| ≥ 8) ∧ 
  (r1 ≥ 0 ∨ r2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l1871_187136


namespace NUMINAMATH_CALUDE_halloween_cleaning_time_l1871_187113

/-- Calculates the total cleaning time for Halloween pranks -/
theorem halloween_cleaning_time 
  (egg_cleaning_time : ℕ) 
  (tp_cleaning_time : ℕ) 
  (num_eggs : ℕ) 
  (num_tp : ℕ) : 
  egg_cleaning_time = 15 ∧ 
  tp_cleaning_time = 30 ∧ 
  num_eggs = 60 ∧ 
  num_tp = 7 → 
  (num_eggs * egg_cleaning_time) / 60 + num_tp * tp_cleaning_time = 225 := by
  sorry

#check halloween_cleaning_time

end NUMINAMATH_CALUDE_halloween_cleaning_time_l1871_187113


namespace NUMINAMATH_CALUDE_mineral_age_arrangements_eq_60_l1871_187103

/-- The number of arrangements for a six-digit number using 2, 2, 4, 4, 7, 9, starting with an odd digit -/
def mineral_age_arrangements : ℕ :=
  let digits : List ℕ := [2, 2, 4, 4, 7, 9]
  let odd_digits : List ℕ := digits.filter (λ d => d % 2 = 1)
  let remaining_digits : ℕ := digits.length - 1
  let repeated_digits : List ℕ := [2, 4]
  odd_digits.length * (remaining_digits.factorial / (repeated_digits.map (λ d => (digits.count d).factorial)).prod)

/-- Theorem stating that the number of possible arrangements is 60 -/
theorem mineral_age_arrangements_eq_60 : mineral_age_arrangements = 60 := by
  sorry

end NUMINAMATH_CALUDE_mineral_age_arrangements_eq_60_l1871_187103


namespace NUMINAMATH_CALUDE_square_difference_existence_l1871_187146

theorem square_difference_existence (n : ℤ) : 
  (∃ a b : ℤ, n + a^2 = b^2) ↔ n % 4 ≠ 2 := by sorry

end NUMINAMATH_CALUDE_square_difference_existence_l1871_187146


namespace NUMINAMATH_CALUDE_exponential_decreasing_condition_l1871_187194

theorem exponential_decreasing_condition (a : ℝ) :
  (((a / (a - 1) ≤ 0) → (0 ≤ a ∧ a < 1)) ∧
   (∃ a, 0 ≤ a ∧ a < 1 ∧ a / (a - 1) > 0) ∧
   (∀ x y : ℝ, x < y → a^x > a^y ↔ 0 < a ∧ a < 1)) ↔
  (((a / (a - 1) ≤ 0) → (∀ x y : ℝ, x < y → a^x > a^y)) ∧
   (¬∀ a : ℝ, (a / (a - 1) ≤ 0) → (∀ x y : ℝ, x < y → a^x > a^y))) :=
by sorry

end NUMINAMATH_CALUDE_exponential_decreasing_condition_l1871_187194


namespace NUMINAMATH_CALUDE_equation_solution_l1871_187133

theorem equation_solution : ∃ x : ℝ, (2 / x = 3 / (x + 1)) ∧ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1871_187133


namespace NUMINAMATH_CALUDE_reflection_of_C_l1871_187159

/-- Reflects a point over the line y = x -/
def reflect_over_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

theorem reflection_of_C : 
  let C : ℝ × ℝ := (2, 2)
  reflect_over_y_eq_x C = C :=
by sorry

end NUMINAMATH_CALUDE_reflection_of_C_l1871_187159


namespace NUMINAMATH_CALUDE_total_eggs_proof_l1871_187110

/-- The total number of eggs used by Molly's employees at the Wafting Pie Company -/
def total_eggs (morning_eggs afternoon_eggs : ℕ) : ℕ :=
  morning_eggs + afternoon_eggs

/-- Proof that the total number of eggs used is 1339 -/
theorem total_eggs_proof (morning_eggs afternoon_eggs : ℕ) 
  (h1 : morning_eggs = 816) 
  (h2 : afternoon_eggs = 523) : 
  total_eggs morning_eggs afternoon_eggs = 1339 := by
  sorry

#eval total_eggs 816 523

end NUMINAMATH_CALUDE_total_eggs_proof_l1871_187110


namespace NUMINAMATH_CALUDE_matrix_equation_proof_l1871_187137

theorem matrix_equation_proof :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -5; 4, -3]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![-21, 19; 15, -13]
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![-1, -5; 0.5, 3.5]
  M * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_proof_l1871_187137


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1871_187115

/-- An isosceles triangle with sides 4 and 9 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 4 ∧ b = 9 ∧ c = 9 →  -- Two sides are 9, one side is 4
  a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
  a + b + c = 22 :=  -- Perimeter is 22
by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1871_187115


namespace NUMINAMATH_CALUDE_unique_polygon_pair_existence_l1871_187180

theorem unique_polygon_pair_existence : 
  ∃! (n₁ n₂ : ℕ), 
    n₁ > 0 ∧ n₂ > 0 ∧
    ∃ x : ℝ, x > 0 ∧
      (180 - 360 / n₁ : ℝ) = x ∧
      (180 - 360 / n₂ : ℝ) = x / 2 :=
by sorry

end NUMINAMATH_CALUDE_unique_polygon_pair_existence_l1871_187180


namespace NUMINAMATH_CALUDE_luncheon_cost_proof_l1871_187152

/-- Represents the cost of items in a luncheon -/
structure LuncheonCost where
  sandwich : ℝ
  coffee : ℝ
  pie : ℝ

/-- Calculates the total cost of a luncheon given quantities and prices -/
def luncheonTotal (cost : LuncheonCost) (sandwiches coffee pie : ℕ) : ℝ :=
  cost.sandwich * sandwiches + cost.coffee * coffee + cost.pie * pie

theorem luncheon_cost_proof (cost : LuncheonCost) 
  (h1 : luncheonTotal cost 2 5 2 = 3.50)
  (h2 : luncheonTotal cost 3 7 2 = 4.90) :
  luncheonTotal cost 1 1 1 = 1.00 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_cost_proof_l1871_187152


namespace NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l1871_187171

theorem fraction_zero_implies_x_equals_three (x : ℝ) :
  (3 - |x|) / (x + 3) = 0 ∧ x + 3 ≠ 0 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_x_equals_three_l1871_187171


namespace NUMINAMATH_CALUDE_collinear_points_sum_l1871_187186

-- Define a Point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define collinearity for four points in 3D space
def collinear (p q r s : Point3D) : Prop :=
  ∃ (t₁ t₂ t₃ : ℝ), 
    q.x - p.x = t₁ * (r.x - p.x) ∧
    q.y - p.y = t₁ * (r.y - p.y) ∧
    q.z - p.z = t₁ * (r.z - p.z) ∧
    s.x - p.x = t₂ * (r.x - p.x) ∧
    s.y - p.y = t₂ * (r.y - p.y) ∧
    s.z - p.z = t₂ * (r.z - p.z) ∧
    t₃ * (q.x - p.x) = s.x - p.x ∧
    t₃ * (q.y - p.y) = s.y - p.y ∧
    t₃ * (q.z - p.z) = s.z - p.z

theorem collinear_points_sum (a b : ℝ) : 
  collinear 
    (Point3D.mk 2 a b) 
    (Point3D.mk a 3 b) 
    (Point3D.mk a b 4) 
    (Point3D.mk 5 b a) → 
  a + b = 9 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l1871_187186


namespace NUMINAMATH_CALUDE_sum_product_bounds_l1871_187132

theorem sum_product_bounds (a b c d : ℝ) (h : a + b + c + d = 1) :
  0 ≤ a * b + a * c + a * d + b * c + b * d + c * d ∧
  a * b + a * c + a * d + b * c + b * d + c * d ≤ 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_bounds_l1871_187132


namespace NUMINAMATH_CALUDE_container_volume_increase_l1871_187149

/-- Given a cylindrical container with volume V = πr²h that holds 3 gallons,
    prove that a new container with triple the radius and double the height holds 54 gallons. -/
theorem container_volume_increase (r h : ℝ) (h1 : r > 0) (h2 : h > 0) :
  π * r^2 * h = 3 → π * (3*r)^2 * (2*h) = 54 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_increase_l1871_187149


namespace NUMINAMATH_CALUDE_first_traveler_constant_speed_second_traveler_constant_speed_l1871_187108

-- Define the speeds and distances
def speed1 : ℝ := 4
def speed2 : ℝ := 6
def total_distance : ℝ := 24

-- Define the constant speeds to be proven
def constant_speed1 : ℝ := 4.8
def constant_speed2 : ℝ := 5

-- Theorem for the first traveler
theorem first_traveler_constant_speed :
  let time1 := (total_distance / 2) / speed1
  let time2 := (total_distance / 2) / speed2
  let total_time := time1 + time2
  total_distance / total_time = constant_speed1 := by sorry

-- Theorem for the second traveler
theorem second_traveler_constant_speed :
  let total_time : ℝ := 2 -- Arbitrary total time
  let distance1 := speed1 * (total_time / 2)
  let distance2 := speed2 * (total_time / 2)
  let total_distance := distance1 + distance2
  total_distance / total_time = constant_speed2 := by sorry

end NUMINAMATH_CALUDE_first_traveler_constant_speed_second_traveler_constant_speed_l1871_187108


namespace NUMINAMATH_CALUDE_three_digit_number_l1871_187185

/-- Given a three-digit natural number where the hundreds digit is 5,
    the tens digit is 1, and the units digit is 3, prove that the number is 513. -/
theorem three_digit_number (n : ℕ) : 
  n ≥ 100 ∧ n < 1000 ∧ 
  (n / 100 = 5) ∧ 
  ((n / 10) % 10 = 1) ∧ 
  (n % 10 = 3) → 
  n = 513 := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_l1871_187185


namespace NUMINAMATH_CALUDE_dice_game_probability_l1871_187187

/-- Represents a pair of dice rolls -/
structure DiceRoll :=
  (first : Nat) (second : Nat)

/-- The set of all possible dice rolls -/
def allRolls : Finset DiceRoll := sorry

/-- The set of dice rolls that sum to 8 -/
def rollsSum8 : Finset DiceRoll := sorry

/-- Probability of rolling a specific combination -/
def probSpecificRoll : ℚ := 1 / 36

theorem dice_game_probability : 
  (Finset.card rollsSum8 : ℚ) * probSpecificRoll = 5 / 36 := by sorry

end NUMINAMATH_CALUDE_dice_game_probability_l1871_187187


namespace NUMINAMATH_CALUDE_inequality_proof_l1871_187105

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1871_187105


namespace NUMINAMATH_CALUDE_removed_triangles_area_l1871_187168

/-- Given a square from which isosceles right triangles are removed from each corner
    to form a rectangle with a diagonal of 15 units, the combined area of the four
    removed triangles is 28.125 square units. -/
theorem removed_triangles_area (s : ℝ) (x : ℝ) : 
  (s - 2*x)^2 + (s - 2*x)^2 = 15^2 →
  4 * (1/2 * x^2) = 28.125 := by
sorry

end NUMINAMATH_CALUDE_removed_triangles_area_l1871_187168


namespace NUMINAMATH_CALUDE_prob_two_sunny_days_l1871_187147

/-- The probability of exactly 2 sunny days in a 5-day period with 75% chance of rain each day -/
theorem prob_two_sunny_days : 
  let n : ℕ := 5  -- Total number of days
  let p : ℚ := 3/4  -- Probability of rain each day
  let k : ℕ := 2  -- Number of sunny days we want
  Nat.choose n k * (1 - p)^k * p^(n - k) = 135/512 :=
by sorry

end NUMINAMATH_CALUDE_prob_two_sunny_days_l1871_187147


namespace NUMINAMATH_CALUDE_segment_PQ_length_l1871_187155

-- Define the points on a line
variable (P Q R S T : ℝ)

-- Define the order of points
axiom order : P < Q ∧ Q < R ∧ R < S ∧ S < T

-- Define the sum of distances from P and Q to other points
axiom sum_distances_P : |Q - P| + |R - P| + |S - P| + |T - P| = 67
axiom sum_distances_Q : |P - Q| + |R - Q| + |S - Q| + |T - Q| = 34

-- Theorem to prove
theorem segment_PQ_length : |Q - P| = 11 := by
  sorry

end NUMINAMATH_CALUDE_segment_PQ_length_l1871_187155


namespace NUMINAMATH_CALUDE_greatest_x_value_l1871_187167

theorem greatest_x_value : 
  (∃ (x : ℝ), ((4*x - 16)/(3*x - 4))^2 + (4*x - 16)/(3*x - 4) = 20) ∧ 
  (∀ (x : ℝ), ((4*x - 16)/(3*x - 4))^2 + (4*x - 16)/(3*x - 4) = 20 → x ≤ 36/19) ∧
  (((4*(36/19) - 16)/(3*(36/19) - 4))^2 + (4*(36/19) - 16)/(3*(36/19) - 4) = 20) :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_value_l1871_187167


namespace NUMINAMATH_CALUDE_even_decreasing_inequality_l1871_187104

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def decreasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, 0 ≤ x₁ → 0 ≤ x₂ → x₁ ≠ x₂ → (f x₂ - f x₁) / (x₂ - x₁) < 0

theorem even_decreasing_inequality (f : ℝ → ℝ) 
  (h_even : is_even f) (h_dec : decreasing_on_nonneg f) : 
  f 3 < f (-2) ∧ f (-2) < f 1 := by sorry

end NUMINAMATH_CALUDE_even_decreasing_inequality_l1871_187104


namespace NUMINAMATH_CALUDE_justin_reading_problem_l1871_187158

/-- Justin's reading problem -/
theorem justin_reading_problem (pages_first_day : ℕ) (remaining_days : ℕ) :
  pages_first_day = 10 →
  remaining_days = 6 →
  pages_first_day + remaining_days * (2 * pages_first_day) = 130 :=
by
  sorry

end NUMINAMATH_CALUDE_justin_reading_problem_l1871_187158


namespace NUMINAMATH_CALUDE_number_of_combinations_max_probability_sums_l1871_187172

-- Define the structure of a box
structure Box :=
  (ball1 : Nat)
  (ball2 : Nat)

-- Define the set of boxes
def boxes : List Box := [
  { ball1 := 1, ball2 := 2 },
  { ball1 := 1, ball2 := 2 },
  { ball1 := 1, ball2 := 2 }
]

-- Define a combination of drawn balls
def Combination := Nat × Nat × Nat

-- Function to generate all possible combinations
def generateCombinations (boxes : List Box) : List Combination := sorry

-- Function to calculate the sum of a combination
def sumCombination (c : Combination) : Nat := sorry

-- Function to count occurrences of a sum
def countSum (sum : Nat) (combinations : List Combination) : Nat := sorry

-- Theorem: The number of possible combinations is 8
theorem number_of_combinations :
  (generateCombinations boxes).length = 8 := by sorry

-- Theorem: The sums 4 and 5 have the highest probability
theorem max_probability_sums (combinations : List Combination := generateCombinations boxes) :
  ∀ (s : Nat), s ≠ 4 ∧ s ≠ 5 →
    countSum s combinations ≤ countSum 4 combinations ∧
    countSum s combinations ≤ countSum 5 combinations := by sorry

end NUMINAMATH_CALUDE_number_of_combinations_max_probability_sums_l1871_187172


namespace NUMINAMATH_CALUDE_post_office_distance_l1871_187138

/-- The distance from the village to the post office satisfies the given conditions -/
theorem post_office_distance (D : ℝ) : D > 0 →
  (D / 25 + D / 4 = 5.8) → D = 20 := by sorry

end NUMINAMATH_CALUDE_post_office_distance_l1871_187138


namespace NUMINAMATH_CALUDE_infinite_solutions_l1871_187144

theorem infinite_solutions (b : ℝ) : 
  (∀ x, 5 * (4 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := by sorry

end NUMINAMATH_CALUDE_infinite_solutions_l1871_187144


namespace NUMINAMATH_CALUDE_cheryl_material_problem_l1871_187129

theorem cheryl_material_problem (x : ℝ) : 
  x > 0 ∧ 
  x + 1/3 > 0 ∧ 
  8/24 < x + 1/3 ∧ 
  x = 0.5555555555555556 → 
  x = 0.5555555555555556 := by
sorry

end NUMINAMATH_CALUDE_cheryl_material_problem_l1871_187129


namespace NUMINAMATH_CALUDE_complex_function_minimum_on_unit_circle_l1871_187123

theorem complex_function_minimum_on_unit_circle
  (a : ℝ) (ha : 0 < a ∧ a < 1)
  (f : ℂ → ℂ) (hf : ∀ z, f z = z^2 - z + a) :
  ∀ z : ℂ, 1 ≤ Complex.abs z →
  ∃ z₀ : ℂ, Complex.abs z₀ = 1 ∧ Complex.abs (f z₀) ≤ Complex.abs (f z) :=
by sorry

end NUMINAMATH_CALUDE_complex_function_minimum_on_unit_circle_l1871_187123


namespace NUMINAMATH_CALUDE_counterexample_exists_l1871_187153

theorem counterexample_exists : ∃ a : ℝ, (abs a > 2) ∧ (a ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_counterexample_exists_l1871_187153


namespace NUMINAMATH_CALUDE_girls_in_circle_l1871_187161

/-- The number of girls in a circle of children, given specific conditions. -/
def number_of_girls (total : ℕ) (holding_boys_hand : ℕ) (holding_girls_hand : ℕ) : ℕ :=
  (2 * holding_girls_hand + holding_boys_hand - total) / 2

/-- Theorem stating that the number of girls in the circle is 24. -/
theorem girls_in_circle : number_of_girls 40 22 30 = 24 := by
  sorry

#eval number_of_girls 40 22 30

end NUMINAMATH_CALUDE_girls_in_circle_l1871_187161


namespace NUMINAMATH_CALUDE_strengthened_inequality_l1871_187100

theorem strengthened_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hsum : a + b + c = 1) :
  27 * (a^3 + b^3 + c^3) + 1 ≥ 12 * (a^2 + b^2 + c^2) := by
  sorry

end NUMINAMATH_CALUDE_strengthened_inequality_l1871_187100


namespace NUMINAMATH_CALUDE_initial_wings_count_l1871_187177

/-- The number of initially cooked chicken wings -/
def initial_wings : ℕ := sorry

/-- The number of friends -/
def num_friends : ℕ := 3

/-- The number of additional wings cooked -/
def additional_wings : ℕ := 10

/-- The number of wings each person got -/
def wings_per_person : ℕ := 6

/-- Theorem stating that the number of initially cooked wings is 8 -/
theorem initial_wings_count : initial_wings = 8 := by sorry

end NUMINAMATH_CALUDE_initial_wings_count_l1871_187177


namespace NUMINAMATH_CALUDE_symmetric_complex_division_l1871_187134

/-- Two complex numbers are symmetric with respect to y = x if their real and imaginary parts are swapped -/
def symmetric_wrt_y_eq_x (z₁ z₂ : ℂ) : Prop :=
  z₁.re = z₂.im ∧ z₁.im = z₂.re

/-- The main theorem -/
theorem symmetric_complex_division (z₁ z₂ : ℂ) 
  (h_sym : symmetric_wrt_y_eq_x z₁ z₂) (h_z₁ : z₁ = 1 + 2*I) : 
  z₁ / z₂ = 4/5 + 3/5*I :=
sorry

end NUMINAMATH_CALUDE_symmetric_complex_division_l1871_187134


namespace NUMINAMATH_CALUDE_spies_configuration_exists_l1871_187102

/-- Represents a position on the 6x6 board -/
structure Position where
  row : Fin 6
  col : Fin 6

/-- Represents the direction a spy is facing -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a spy on the board -/
structure Spy where
  pos : Position
  dir : Direction

/-- Determines if a spy can see a given position -/
def Spy.canSee (s : Spy) (p : Position) : Bool :=
  match s.dir with
  | Direction.Up => 
      (s.pos.row < p.row && p.row ≤ s.pos.row + 2 && s.pos.col - 1 ≤ p.col && p.col ≤ s.pos.col + 1) 
  | Direction.Down => 
      (s.pos.row > p.row && p.row ≥ s.pos.row - 2 && s.pos.col - 1 ≤ p.col && p.col ≤ s.pos.col + 1)
  | Direction.Left => 
      (s.pos.col > p.col && p.col ≥ s.pos.col - 2 && s.pos.row - 1 ≤ p.row && p.row ≤ s.pos.row + 1)
  | Direction.Right => 
      (s.pos.col < p.col && p.col ≤ s.pos.col + 2 && s.pos.row - 1 ≤ p.row && p.row ≤ s.pos.row + 1)

/-- A valid configuration of spies -/
def ValidConfiguration (spies : List Spy) : Prop :=
  spies.length = 18 ∧ 
  ∀ s1 s2, s1 ∈ spies → s2 ∈ spies → s1 ≠ s2 → ¬(s1.canSee s2.pos) ∧ ¬(s2.canSee s1.pos)

/-- There exists a valid configuration of 18 spies on a 6x6 board -/
theorem spies_configuration_exists : ∃ spies : List Spy, ValidConfiguration spies := by
  sorry

end NUMINAMATH_CALUDE_spies_configuration_exists_l1871_187102


namespace NUMINAMATH_CALUDE_only_expr3_correct_l1871_187182

-- Define the expressions to be evaluated
def expr1 : Int := (-2)^3
def expr2 : Int := (-3)^2
def expr3 : Int := -3^2
def expr4 : Int := (-2)^2

-- Theorem stating that only the third expression is correct
theorem only_expr3_correct :
  expr1 ≠ 8 ∧ 
  expr2 ≠ -9 ∧ 
  expr3 = -9 ∧ 
  expr4 = 4 :=
by sorry

end NUMINAMATH_CALUDE_only_expr3_correct_l1871_187182


namespace NUMINAMATH_CALUDE_unique_root_of_sin_plus_constant_l1871_187107

theorem unique_root_of_sin_plus_constant :
  ∃! x : ℝ, x = Real.sin x + 1993 := by sorry

end NUMINAMATH_CALUDE_unique_root_of_sin_plus_constant_l1871_187107


namespace NUMINAMATH_CALUDE_power_equality_l1871_187114

theorem power_equality (J : ℕ) (h : (32^4) * (4^4) = 2^J) : J = 28 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l1871_187114


namespace NUMINAMATH_CALUDE_mikeys_leaves_l1871_187112

/-- The number of leaves that blew away -/
def leaves_blown_away (initial final : ℕ) : ℕ := initial - final

/-- Proof that 244 leaves blew away -/
theorem mikeys_leaves : leaves_blown_away 356 112 = 244 := by
  sorry

end NUMINAMATH_CALUDE_mikeys_leaves_l1871_187112


namespace NUMINAMATH_CALUDE_min_distance_exp_curve_to_line_l1871_187116

/-- The minimum distance between a point on y = e^x and a point on y = x is √2/2 -/
theorem min_distance_exp_curve_to_line : 
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  ∃ (d : ℝ), d = Real.sqrt 2 / 2 ∧ 
    ∀ (p q : ℝ × ℝ), p.2 = Real.exp p.1 → q.2 = q.1 → 
      dist p q ≥ d :=
sorry

end NUMINAMATH_CALUDE_min_distance_exp_curve_to_line_l1871_187116


namespace NUMINAMATH_CALUDE_sophomore_latin_probability_l1871_187122

/-- Represents the percentage of students in each class -/
structure ClassDistribution :=
  (freshmen : ℚ)
  (sophomores : ℚ)
  (juniors : ℚ)
  (seniors : ℚ)

/-- Represents the percentage of students taking Latin in each class -/
structure LatinRates :=
  (freshmen : ℚ)
  (sophomores : ℚ)
  (juniors : ℚ)
  (seniors : ℚ)

/-- The probability that a randomly chosen Latin student is a sophomore -/
def sophomoreProbability (dist : ClassDistribution) (rates : LatinRates) : ℚ :=
  (dist.sophomores * rates.sophomores) /
  (dist.freshmen * rates.freshmen + dist.sophomores * rates.sophomores +
   dist.juniors * rates.juniors + dist.seniors * rates.seniors)

theorem sophomore_latin_probability :
  let dist : ClassDistribution := {
    freshmen := 2/5, sophomores := 3/10, juniors := 1/5, seniors := 1/10
  }
  let rates : LatinRates := {
    freshmen := 1, sophomores := 4/5, juniors := 1/2, seniors := 1/5
  }
  sophomoreProbability dist rates = 6/19 := by sorry

end NUMINAMATH_CALUDE_sophomore_latin_probability_l1871_187122


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_proof_l1871_187126

theorem arithmetic_sequence_sum_proof : 
  let n : ℕ := 10
  let a : ℕ := 70
  let d : ℕ := 3
  let l : ℕ := 97
  3 * (n / 2 * (a + l)) = 2505 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_proof_l1871_187126


namespace NUMINAMATH_CALUDE_zip_code_sum_l1871_187125

theorem zip_code_sum (a b c d e : ℕ) : 
  a + b + c + d + e = 10 →
  a = b →
  c = 0 →
  d = 2 * a →
  d + e = 8 := by
sorry

end NUMINAMATH_CALUDE_zip_code_sum_l1871_187125


namespace NUMINAMATH_CALUDE_percentage_boys_school_A_l1871_187197

theorem percentage_boys_school_A (total_boys : ℕ) (boys_A_not_science : ℕ) 
  (h1 : total_boys = 550)
  (h2 : boys_A_not_science = 77)
  (h3 : ∀ P : ℝ, P > 0 → P < 100 → 
    (P / 100) * total_boys * (70 / 100) = boys_A_not_science → P = 20) :
  ∃ P : ℝ, P > 0 ∧ P < 100 ∧ (P / 100) * total_boys * (70 / 100) = boys_A_not_science ∧ P = 20 := by
sorry

end NUMINAMATH_CALUDE_percentage_boys_school_A_l1871_187197


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l1871_187140

theorem rectangle_side_ratio (a b c d : ℝ) (h : a / c = b / d ∧ a / c = 4 / 5) :
  a / c = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l1871_187140


namespace NUMINAMATH_CALUDE_root_equation_implies_expression_value_l1871_187190

theorem root_equation_implies_expression_value (a : ℝ) :
  a^2 - 2*a - 2 = 0 →
  (1 - 1/(a + 1)) / (a^3 / (a^2 + 2*a + 1)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_expression_value_l1871_187190


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l1871_187109

/-- The standard equation of an ellipse with given minor axis length and eccentricity -/
theorem ellipse_standard_equation (b : ℝ) (e : ℝ) : 
  b = 4 ∧ e = 3/5 → 
  ∃ (a : ℝ), (a > b) ∧ 
  ((∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ (x^2/25 + y^2/16 = 1 ∨ x^2/16 + y^2/25 = 1)) ∧
   e^2 = 1 - b^2/a^2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l1871_187109


namespace NUMINAMATH_CALUDE_sum_remainder_mod_seven_l1871_187119

theorem sum_remainder_mod_seven :
  (9543 + 9544 + 9545 + 9546 + 9547) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_seven_l1871_187119


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l1871_187170

/-- Given the prices and quantities of blankets, proves that the unknown rate is 285 when the average price is 162. -/
theorem unknown_blanket_rate (price1 price2 avg_price : ℚ) (qty1 qty2 qty_unknown : ℕ) : 
  price1 = 100 →
  price2 = 150 →
  qty1 = 3 →
  qty2 = 5 →
  qty_unknown = 2 →
  avg_price = 162 →
  (qty1 * price1 + qty2 * price2 + qty_unknown * (avg_price * (qty1 + qty2 + qty_unknown) - qty1 * price1 - qty2 * price2) / qty_unknown) / (qty1 + qty2 + qty_unknown) = avg_price →
  (avg_price * (qty1 + qty2 + qty_unknown) - qty1 * price1 - qty2 * price2) / qty_unknown = 285 := by
  sorry

#check unknown_blanket_rate

end NUMINAMATH_CALUDE_unknown_blanket_rate_l1871_187170


namespace NUMINAMATH_CALUDE_david_win_4022_l1871_187142

/-- The game state representing the numbers on the blackboard -/
def GameState := List Nat

/-- Represents a player in the game -/
inductive Player
| David
| Goliath

/-- The result of the game -/
inductive GameResult
| DavidWins
| GoliathWins

/-- Determine if a number is even -/
def isEven (n : Nat) : Bool :=
  n % 2 = 0

/-- Play the game with the given initial state -/
def playGame (initialState : GameState) : GameResult :=
  sorry

/-- Check if David can guarantee a win from the given initial state -/
def davidCanGuaranteeWin (n : Nat) : Bool :=
  sorry

/-- Find the kth smallest positive integer greater than 1 for which David can guarantee victory -/
def findKthDavidWinNumber (k : Nat) : Nat :=
  sorry

theorem david_win_4022 :
  findKthDavidWinNumber 2011 = 4022 :=
sorry

end NUMINAMATH_CALUDE_david_win_4022_l1871_187142


namespace NUMINAMATH_CALUDE_librarians_work_schedule_l1871_187124

theorem librarians_work_schedule : Nat.lcm 5 (Nat.lcm 8 (Nat.lcm 10 14)) = 280 := by
  sorry

end NUMINAMATH_CALUDE_librarians_work_schedule_l1871_187124


namespace NUMINAMATH_CALUDE_product_equals_one_l1871_187121

theorem product_equals_one (a b : ℝ) : a * (b + 1) + b * (a + 1) = (a + 1) * (b + 1) → a * b = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_one_l1871_187121


namespace NUMINAMATH_CALUDE_goods_train_length_l1871_187145

/-- The length of a goods train passing a man in an opposite-moving train --/
theorem goods_train_length (man_speed goods_speed : ℝ) (pass_time : ℝ) : 
  man_speed = 64 →
  goods_speed = 20 →
  pass_time = 18 →
  ∃ (length : ℝ), abs (length - (man_speed + goods_speed) * 1000 / 3600 * pass_time) < 1 :=
by sorry

end NUMINAMATH_CALUDE_goods_train_length_l1871_187145


namespace NUMINAMATH_CALUDE_prime_pairs_sum_58_l1871_187118

/-- A function that checks if a natural number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- The theorem stating that there are exactly 4 pairs of primes summing to 58 -/
theorem prime_pairs_sum_58 :
  ∃! (s : Finset (ℕ × ℕ)), 
    (∀ (p q : ℕ), (p, q) ∈ s ↔ isPrime p ∧ isPrime q ∧ p + q = 58) ∧
    s.card = 4 :=
sorry

end NUMINAMATH_CALUDE_prime_pairs_sum_58_l1871_187118


namespace NUMINAMATH_CALUDE_show_dog_profit_l1871_187130

/-- Calculate the total profit from breeding and selling show dogs -/
theorem show_dog_profit
  (num_dogs : ℕ)
  (cost_per_dog : ℚ)
  (num_puppies : ℕ)
  (price_per_puppy : ℚ)
  (h1 : num_dogs = 2)
  (h2 : cost_per_dog = 250)
  (h3 : num_puppies = 6)
  (h4 : price_per_puppy = 350) :
  (num_puppies : ℚ) * price_per_puppy - (num_dogs : ℚ) * cost_per_dog = 1600 :=
by sorry

end NUMINAMATH_CALUDE_show_dog_profit_l1871_187130


namespace NUMINAMATH_CALUDE_washer_dryer_cost_difference_l1871_187188

theorem washer_dryer_cost_difference (total_cost washer_cost : ℕ) : 
  total_cost = 1200 → washer_cost = 710 → 
  washer_cost - (total_cost - washer_cost) = 220 := by
  sorry

end NUMINAMATH_CALUDE_washer_dryer_cost_difference_l1871_187188
