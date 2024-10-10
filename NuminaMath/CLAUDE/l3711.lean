import Mathlib

namespace binary_101_equals_5_l3711_371171

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : Nat :=
  bits.foldr (fun b acc => 2 * acc + if b then 1 else 0) 0

/-- The binary representation of 101 -/
def binary_101 : List Bool := [true, false, true]

theorem binary_101_equals_5 : binary_to_decimal binary_101 = 5 := by
  sorry

end binary_101_equals_5_l3711_371171


namespace c_value_for_four_distinct_roots_l3711_371104

/-- The polynomial P(x) -/
def P (c : ℂ) (x : ℂ) : ℂ := (x^2 - 3*x + 5) * (x^2 - c*x + 2) * (x^2 - 5*x + 10)

/-- The theorem stating the relationship between c and the number of distinct roots of P(x) -/
theorem c_value_for_four_distinct_roots (c : ℂ) : 
  (∃ (S : Finset ℂ), S.card = 4 ∧ (∀ x ∈ S, P c x = 0) ∧ (∀ x, P c x = 0 → x ∈ S)) →
  Complex.abs c = Real.sqrt (22.5 - Real.sqrt 165) := by
  sorry

end c_value_for_four_distinct_roots_l3711_371104


namespace ram_money_l3711_371143

/-- Given the ratio of money between Ram and Gopal, and between Gopal and Krishan,
    prove that Ram has 637 rupees when Krishan has 3757 rupees. -/
theorem ram_money (ram gopal krishan : ℚ) : 
  ram / gopal = 7 / 17 →
  gopal / krishan = 7 / 17 →
  krishan = 3757 →
  ram = 637 := by
  sorry

end ram_money_l3711_371143


namespace domain_transformation_l3711_371115

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the domain of f(x+1)
def domain_f_shifted : Set ℝ := Set.Icc 0 1

-- Define the domain of f(√(2x-1))
def domain_f_sqrt : Set ℝ := Set.Icc 1 (5/2)

-- State the theorem
theorem domain_transformation (h : ∀ x ∈ domain_f_shifted, f (x + 1) = f (x + 1)) :
  ∀ x ∈ domain_f_sqrt, f (Real.sqrt (2 * x - 1)) = f (Real.sqrt (2 * x - 1)) :=
sorry

end domain_transformation_l3711_371115


namespace millet_majority_on_day_three_l3711_371121

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  millet : Real
  other : Real

/-- Calculates the next day's feeder state based on the current state -/
def nextDay (state : FeederState) : FeederState :=
  let remainingMillet := state.millet * 0.8
  let newMillet := if state.day = 1 then 0.5 else 0.4
  { day := state.day + 1,
    millet := remainingMillet + newMillet,
    other := 0.6 }

/-- Initial state of the feeder on Monday -/
def initialState : FeederState :=
  { day := 1, millet := 0.4, other := 0.6 }

/-- Theorem stating that on Day 3, more than half of the seeds are millet -/
theorem millet_majority_on_day_three :
  let day3State := nextDay (nextDay initialState)
  day3State.millet / (day3State.millet + day3State.other) > 0.5 := by
  sorry


end millet_majority_on_day_three_l3711_371121


namespace max_distinct_dance_counts_29_15_l3711_371154

/-- Represents the maximum number of distinct dance counts that can be reported
    given a number of boys and girls at a ball, where each boy can dance with
    each girl at most once. -/
def max_distinct_dance_counts (num_boys num_girls : ℕ) : ℕ :=
  sorry

/-- Theorem stating that for 29 boys and 15 girls, the maximum number of
    distinct dance counts is 29. -/
theorem max_distinct_dance_counts_29_15 :
  max_distinct_dance_counts 29 15 = 29 := by sorry

end max_distinct_dance_counts_29_15_l3711_371154


namespace investment_problem_l3711_371102

/-- Given two investors P and Q, where P invested 40000 and their profit ratio is 2:3,
    prove that Q's investment is 60000. -/
theorem investment_problem (P Q : ℕ) (h1 : P = 40000) (h2 : 2 * Q = 3 * P) : Q = 60000 := by
  sorry

end investment_problem_l3711_371102


namespace expand_expression_l3711_371130

theorem expand_expression (m n : ℝ) : (2*m + n - 1) * (2*m - n + 1) = 4*m^2 - n^2 + 2*n - 1 := by
  sorry

end expand_expression_l3711_371130


namespace starting_lineup_combinations_l3711_371109

def total_players : ℕ := 15
def guaranteed_players : ℕ := 3
def lineup_size : ℕ := 5

theorem starting_lineup_combinations :
  Nat.choose (total_players - guaranteed_players) (lineup_size - guaranteed_players) = 66 := by
  sorry

end starting_lineup_combinations_l3711_371109


namespace parabola_slope_AF_l3711_371139

-- Define the parabola
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define the theorem
theorem parabola_slope_AF (C : Parabola) (A F : Point) :
  A.x = -2 ∧ A.y = 3 ∧  -- A is (-2, 3)
  A.x = -C.p/2 ∧        -- A is on the directrix
  F.x = C.p/2 ∧ F.y = 0 -- F is the focus
  →
  (F.y - A.y) / (F.x - A.x) = -3/4 := by
  sorry

end parabola_slope_AF_l3711_371139


namespace parking_lot_cars_l3711_371113

/-- Given a parking lot with observed wheels and wheels per car, calculate the number of cars -/
def number_of_cars (total_wheels : ℕ) (wheels_per_car : ℕ) : ℕ :=
  total_wheels / wheels_per_car

/-- Theorem: In a parking lot with 48 observed wheels and 4 wheels per car, there are 12 cars -/
theorem parking_lot_cars : number_of_cars 48 4 = 12 := by
  sorry

end parking_lot_cars_l3711_371113


namespace solutions_to_equation_l3711_371148

theorem solutions_to_equation : 
  {(m, n) : ℕ × ℕ | 7^m - 3 * 2^n = 1} = {(1, 1), (2, 4)} := by sorry

end solutions_to_equation_l3711_371148


namespace original_fraction_l3711_371106

theorem original_fraction (x y : ℚ) : 
  x > 0 ∧ y > 0 →
  (120 / 100 * x) / (75 / 100 * y) = 2 / 15 →
  x / y = 1 / 12 :=
by sorry

end original_fraction_l3711_371106


namespace k_range_l3711_371195

theorem k_range (k : ℝ) : (∀ x : ℝ, k * x^2 - k * x - 1 < 0) → -4 < k ∧ k ≤ 0 := by
  sorry

end k_range_l3711_371195


namespace complex_sum_equality_l3711_371145

theorem complex_sum_equality : 
  let A : ℂ := 2 + I
  let O : ℂ := -4
  let P : ℂ := -I
  let S : ℂ := 2 + 4*I
  A - O + P + S = 8 + 4*I :=
by sorry

end complex_sum_equality_l3711_371145


namespace june_greatest_drop_l3711_371133

/-- Represents the months in the first half of the year -/
inductive Month
  | January
  | February
  | March
  | April
  | May
  | June

/-- The price change for each month -/
def price_change (m : Month) : ℝ :=
  match m with
  | Month.January => 1.50
  | Month.February => -2.25
  | Month.March => 0.75
  | Month.April => -3.00
  | Month.May => 1.00
  | Month.June => -4.00

/-- The month with the greatest price drop -/
def greatest_drop : Month := Month.June

theorem june_greatest_drop :
  ∀ m : Month, price_change m ≥ price_change greatest_drop → m = greatest_drop :=
by sorry

end june_greatest_drop_l3711_371133


namespace problem_statements_l3711_371125

theorem problem_statements :
  -- Statement 1
  (¬ ∀ x : ℝ, x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≤ 0) ∧
  -- Statement 2
  ∃ a b c d : ℝ, a > b ∧ c > d ∧ a * c ≤ b * d ∧
  -- Statement 3
  ∀ a : ℝ, (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 + (a - 3) * x + a = 0 ∧ y^2 + (a - 3) * y + a = 0) → a < 0 :=
by sorry

end problem_statements_l3711_371125


namespace max_area_difference_l3711_371112

-- Define a rectangle with integer dimensions
structure Rectangle where
  length : ℕ
  width : ℕ

-- Define the perimeter of a rectangle
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

-- Define the area of a rectangle
def area (r : Rectangle) : ℕ := r.length * r.width

-- Theorem statement
theorem max_area_difference : 
  ∃ (r1 r2 : Rectangle), 
    perimeter r1 = 180 ∧ 
    perimeter r2 = 180 ∧ 
    (∀ (r : Rectangle), perimeter r = 180 → 
      area r1 - area r2 ≥ area r1 - area r ∧ 
      area r1 - area r2 ≥ area r - area r2) ∧
    area r1 - area r2 = 1936 :=
sorry

end max_area_difference_l3711_371112


namespace isabel_piggy_bank_l3711_371111

theorem isabel_piggy_bank (initial_amount : ℝ) : 
  initial_amount / 2 / 2 = 51 → initial_amount = 204 := by
  sorry

end isabel_piggy_bank_l3711_371111


namespace opposite_of_three_abs_l3711_371174

theorem opposite_of_three_abs (x : ℝ) : x = -3 → |x + 2| = 1 := by
  sorry

end opposite_of_three_abs_l3711_371174


namespace extra_sodas_l3711_371128

/-- Given that Robin bought 11 sodas and drank 3 sodas, prove that the number of extra sodas is 8. -/
theorem extra_sodas (total : ℕ) (drank : ℕ) (h1 : total = 11) (h2 : drank = 3) :
  total - drank = 8 := by
  sorry

end extra_sodas_l3711_371128


namespace triangle_toothpicks_l3711_371186

/-- Calculates the number of toothpicks needed for a large equilateral triangle
    with a given base length and border. -/
def toothpicks_for_triangle (base : ℕ) (border : ℕ) : ℕ :=
  let interior_triangles := base * (base + 1) / 2
  let interior_toothpicks := 3 * interior_triangles / 2
  let boundary_toothpicks := 3 * base
  let border_toothpicks := 2 * border + 2
  interior_toothpicks + boundary_toothpicks + border_toothpicks

/-- Theorem stating that a triangle with base 100 and border 100 requires 8077 toothpicks -/
theorem triangle_toothpicks :
  toothpicks_for_triangle 100 100 = 8077 := by
  sorry

end triangle_toothpicks_l3711_371186


namespace hyperbola_eccentricity_bound_l3711_371187

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if a line passing through the right focus with a slope angle of 60° 
    intersects the right branch of the hyperbola at exactly one point,
    then the eccentricity e of the hyperbola satisfies e ≥ 2. -/
theorem hyperbola_eccentricity_bound (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  let slope := Real.tan (π / 3)
  (b / a ≥ slope) → e ≥ 2 := by
  sorry

end hyperbola_eccentricity_bound_l3711_371187


namespace binomial_60_3_l3711_371149

theorem binomial_60_3 : Nat.choose 60 3 = 57020 := by sorry

end binomial_60_3_l3711_371149


namespace range_of_a_l3711_371191

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, x^2 - a > 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) → 
  a ∈ Set.Iic (-2) :=
by sorry

end range_of_a_l3711_371191


namespace roots_of_equation_l3711_371151

theorem roots_of_equation :
  let f : ℝ → ℝ := λ x => x * (x + 2) + x + 2
  (f (-2) = 0) ∧ (f (-1) = 0) ∧
  (∀ x : ℝ, f x = 0 → x = -2 ∨ x = -1) :=
by sorry

end roots_of_equation_l3711_371151


namespace blanche_eggs_l3711_371114

theorem blanche_eggs (gertrude nancy martha blanche total_eggs : ℕ) : 
  gertrude = 4 →
  nancy = 2 →
  martha = 2 →
  total_eggs = gertrude + nancy + martha + blanche →
  total_eggs - 2 = 9 →
  blanche = 3 := by
sorry

end blanche_eggs_l3711_371114


namespace journey_length_l3711_371100

theorem journey_length :
  ∀ (total : ℚ),
  (1 / 4 : ℚ) * total + 24 + (1 / 6 : ℚ) * total = total →
  total = 288 / 7 := by
sorry

end journey_length_l3711_371100


namespace abs_sum_inequality_l3711_371124

theorem abs_sum_inequality (x : ℝ) : |x + 1| + |x - 2| ≤ 5 ↔ x ∈ Set.Icc (-2) 3 := by
  sorry

end abs_sum_inequality_l3711_371124


namespace tetrahedron_sphere_inequality_l3711_371188

/-- Given a tetrahedron ABCD with inscribed sphere radius r and exinscribed sphere radii r_A, r_B, r_C, r_D,
    the sum of the reciprocals of the square roots of the sums of squares minus products of adjacent radii
    is less than or equal to 2/r. -/
theorem tetrahedron_sphere_inequality (r r_A r_B r_C r_D : ℝ) 
  (hr : r > 0) (hr_A : r_A > 0) (hr_B : r_B > 0) (hr_C : r_C > 0) (hr_D : r_D > 0) :
  1 / Real.sqrt (r_A^2 - r_A*r_B + r_B^2) + 
  1 / Real.sqrt (r_B^2 - r_B*r_C + r_C^2) + 
  1 / Real.sqrt (r_C^2 - r_C*r_D + r_D^2) + 
  1 / Real.sqrt (r_D^2 - r_D*r_A + r_A^2) ≤ 2 / r :=
by sorry

end tetrahedron_sphere_inequality_l3711_371188


namespace pizza_slices_per_person_l3711_371137

theorem pizza_slices_per_person
  (coworkers : ℕ)
  (pizzas : ℕ)
  (slices_per_pizza : ℕ)
  (h1 : coworkers = 18)
  (h2 : pizzas = 4)
  (h3 : slices_per_pizza = 10)
  : (pizzas * slices_per_pizza) / coworkers = 2 :=
by
  sorry

end pizza_slices_per_person_l3711_371137


namespace intersection_nonempty_implies_k_range_l3711_371126

-- Define the sets M and N
def M : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def N (k : ℝ) : Set ℝ := {x | x ≤ k + 3}

-- State the theorem
theorem intersection_nonempty_implies_k_range (k : ℝ) :
  (M ∩ N k).Nonempty → k ≥ -4 := by
  sorry

end intersection_nonempty_implies_k_range_l3711_371126


namespace final_points_count_l3711_371136

/-- The number of points after performing the insertion operation n times -/
def points_after_operations (initial_points : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => initial_points
  | k + 1 => 2 * points_after_operations initial_points k - 1

theorem final_points_count : points_after_operations 2010 3 = 16073 := by
  sorry

end final_points_count_l3711_371136


namespace arc_length_sector_l3711_371146

/-- The arc length of a sector with central angle 2π/3 and radius 3 is 2π. -/
theorem arc_length_sector (α : Real) (r : Real) (l : Real) : 
  α = 2 * Real.pi / 3 → r = 3 → l = α * r → l = 2 * Real.pi := by
  sorry

end arc_length_sector_l3711_371146


namespace police_can_see_bandit_l3711_371101

/-- Represents a point in the city grid -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a policeman -/
structure Policeman where
  position : Point
  canSeeInfinitely : Bool

/-- Represents the bandit -/
structure Bandit where
  position : Point

/-- Represents the city -/
structure City where
  grid : Set Point
  police : Set Policeman
  bandit : Bandit

/-- Represents the initial configuration of the city -/
def initialCity : City :=
  { grid := Set.univ,
    police := { p | ∃ k : ℤ, p.position = ⟨100 * k, 0⟩ ∧ p.canSeeInfinitely = true },
    bandit := ⟨⟨0, 0⟩⟩ }  -- Arbitrary initial position for the bandit

/-- Represents a strategy for the police -/
def PoliceStrategy := City → City

/-- Theorem: There exists a police strategy that guarantees seeing the bandit -/
theorem police_can_see_bandit :
  ∃ (strategy : PoliceStrategy), ∀ (c : City),
    ∃ (t : ℕ), ∃ (p : Policeman),
      p ∈ (strategy^[t] c).police ∧
      (strategy^[t] c).bandit.position.x = p.position.x ∨
      (strategy^[t] c).bandit.position.y = p.position.y :=
sorry

end police_can_see_bandit_l3711_371101


namespace hilt_share_money_l3711_371110

/-- The number of people Mrs. Hilt will share the money with -/
def number_of_people (total_amount : ℚ) (amount_per_person : ℚ) : ℚ :=
  total_amount / amount_per_person

/-- Theorem stating that Mrs. Hilt will share the money with 3 people -/
theorem hilt_share_money : 
  number_of_people (3.75 : ℚ) (1.25 : ℚ) = 3 := by
  sorry

end hilt_share_money_l3711_371110


namespace shelly_has_enough_thread_l3711_371107

/-- Represents the keychain making scenario for Shelly's friends --/
structure KeychainScenario where
  class_friends : Nat
  club_friends : Nat
  sports_friends : Nat
  class_thread : Nat
  club_thread : Nat
  sports_thread : Nat
  available_thread : Nat

/-- Calculates the total thread needed and checks if it's sufficient --/
def thread_calculation (scenario : KeychainScenario) : 
  (Bool × Nat) :=
  let total_needed := 
    scenario.class_friends * scenario.class_thread +
    scenario.club_friends * scenario.club_thread +
    scenario.sports_friends * scenario.sports_thread
  let is_sufficient := total_needed ≤ scenario.available_thread
  let remaining := scenario.available_thread - total_needed
  (is_sufficient, remaining)

/-- Theorem stating that Shelly has enough thread and calculates the remaining amount --/
theorem shelly_has_enough_thread (scenario : KeychainScenario) 
  (h1 : scenario.class_friends = 10)
  (h2 : scenario.club_friends = 20)
  (h3 : scenario.sports_friends = 5)
  (h4 : scenario.class_thread = 18)
  (h5 : scenario.club_thread = 24)
  (h6 : scenario.sports_thread = 30)
  (h7 : scenario.available_thread = 1200) :
  thread_calculation scenario = (true, 390) := by
  sorry

end shelly_has_enough_thread_l3711_371107


namespace rest_stop_location_l3711_371164

/-- The location of the rest stop between two towns -/
theorem rest_stop_location (town_a town_b rest_stop_fraction : ℚ) : 
  town_a = 30 → 
  town_b = 210 → 
  rest_stop_fraction = 4/5 → 
  town_a + rest_stop_fraction * (town_b - town_a) = 174 := by
sorry

end rest_stop_location_l3711_371164


namespace basketball_game_result_l3711_371118

/-- Calculates the final score difference after the last quarter of a basketball game -/
def final_score_difference (initial_deficit : ℤ) (liz_free_throws : ℕ) (liz_three_pointers : ℕ) (liz_jump_shots : ℕ) (opponent_points : ℕ) : ℤ :=
  initial_deficit - (liz_free_throws + 3 * liz_three_pointers + 2 * liz_jump_shots - opponent_points)

theorem basketball_game_result :
  final_score_difference 20 5 3 4 10 = 8 := by sorry

end basketball_game_result_l3711_371118


namespace unique_prime_solution_l3711_371199

/-- The equation p^2 - 6pq + q^2 + 3q - 1 = 0 has only one solution in prime numbers. -/
theorem unique_prime_solution :
  ∃! (p q : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ p^2 - 6*p*q + q^2 + 3*q - 1 = 0 :=
by
  -- The proof goes here
  sorry

end unique_prime_solution_l3711_371199


namespace art_price_increase_theorem_l3711_371140

/-- Calculates the price increase of an art piece given its initial price and a multiplier for its future price. -/
def art_price_increase (initial_price : ℕ) (price_multiplier : ℕ) : ℕ :=
  (price_multiplier * initial_price) - initial_price

/-- Theorem stating that for an art piece with an initial price of $4000 and a future price 3 times the initial price, the price increase is $8000. -/
theorem art_price_increase_theorem :
  art_price_increase 4000 3 = 8000 := by
  sorry

end art_price_increase_theorem_l3711_371140


namespace ab_equals_six_l3711_371142

theorem ab_equals_six (a b : ℝ) (h : a / 2 = 3 / b) : a * b = 6 := by
  sorry

end ab_equals_six_l3711_371142


namespace circle_transformation_l3711_371165

/-- Given a circle and a transformation, prove the equation of the resulting shape -/
theorem circle_transformation (x y x' y' : ℝ) : 
  (x^2 + y^2 = 4) → (x' = 2*x ∧ y' = 3*y) → ((x'^2 / 16) + (y'^2 / 36) = 1) := by
sorry

end circle_transformation_l3711_371165


namespace cafeteria_pies_l3711_371134

theorem cafeteria_pies (total_apples : ℕ) (handout_percentage : ℚ) (apples_per_pie : ℕ) : 
  total_apples = 800 →
  handout_percentage = 65 / 100 →
  apples_per_pie = 15 →
  (total_apples - (total_apples * handout_percentage).floor) / apples_per_pie = 18 :=
by sorry

end cafeteria_pies_l3711_371134


namespace alphabet_letter_count_l3711_371144

theorem alphabet_letter_count (total : ℕ) (both : ℕ) (straight_only : ℕ) 
  (h1 : total = 40)
  (h2 : both = 10)
  (h3 : straight_only = 24)
  (h4 : total = both + straight_only + (total - (both + straight_only))) :
  total - (both + straight_only) = 6 := by
sorry

end alphabet_letter_count_l3711_371144


namespace distance_traveled_l3711_371131

/-- Given a person traveling at 40 km/hr for 6 hours, prove that the distance traveled is 240 km. -/
theorem distance_traveled (speed : ℝ) (time : ℝ) (h1 : speed = 40) (h2 : time = 6) :
  speed * time = 240 := by
  sorry

end distance_traveled_l3711_371131


namespace min_sum_for_product_3006_l3711_371158

theorem min_sum_for_product_3006 (a b c : ℕ+) (h : a * b * c = 3006) :
  (∀ x y z : ℕ+, x * y * z = 3006 → a + b + c ≤ x + y + z) ∧ a + b + c = 105 := by
  sorry

end min_sum_for_product_3006_l3711_371158


namespace average_children_in_families_with_children_l3711_371189

/-- Given a group of families, some with children and some without, 
    calculate the average number of children in families that have children. -/
theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (total_average : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : total_average = 3)
  (h3 : childless_families = 3)
  (h4 : childless_families < total_families) :
  (total_families : ℚ) * total_average / (total_families - childless_families : ℚ) = 3.75 := by
sorry

end average_children_in_families_with_children_l3711_371189


namespace max_rectangle_area_in_right_triangle_max_rectangle_area_40_60_l3711_371108

/-- Given a right-angled triangle with legs a and b, the maximum area of a rectangle
    that can be cut from it, using the right angle of the triangle as one of the
    rectangle's corners, is (a * b) / 4 -/
theorem max_rectangle_area_in_right_triangle (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let triangle_area := a * b / 2
  let max_rectangle_area := triangle_area / 2
  max_rectangle_area = a * b / 4 := by sorry

/-- The maximum area of a rectangle that can be cut from a right-angled triangle
    with legs measuring 40 cm and 60 cm, using the right angle of the triangle as
    one of the rectangle's corners, is 600 cm² -/
theorem max_rectangle_area_40_60 :
  let a : ℝ := 40
  let b : ℝ := 60
  let max_area := a * b / 4
  max_area = 600 := by sorry

end max_rectangle_area_in_right_triangle_max_rectangle_area_40_60_l3711_371108


namespace point_on_bisector_l3711_371184

/-- 
Given a point (a, 2) in the second quadrant and on the angle bisector of the coordinate axes,
prove that a = -2.
-/
theorem point_on_bisector (a : ℝ) :
  (a < 0) →  -- Point is in the second quadrant
  (a = -2) →  -- Point is on the angle bisector
  a = -2 := by
sorry

end point_on_bisector_l3711_371184


namespace max_e_is_one_l3711_371193

/-- The sequence b_n defined as (8^n - 1) / 7 -/
def b (n : ℕ) : ℤ := (8^n - 1) / 7

/-- The greatest common divisor of b_n and b_(n+1) -/
def e (n : ℕ) : ℕ := Nat.gcd (Int.natAbs (b n)) (Int.natAbs (b (n + 1)))

/-- Theorem: The maximum value of e_n is always 1 -/
theorem max_e_is_one : ∀ n : ℕ, e n = 1 := by sorry

end max_e_is_one_l3711_371193


namespace angle_cde_is_85_l3711_371182

-- Define the points
variable (A B C D E : Point)

-- Define the angles
def angle (P Q R : Point) : ℝ := sorry

-- State the conditions
variable (h1 : angle A B C = 90)
variable (h2 : angle B C D = 90)
variable (h3 : angle C D A = 90)
variable (h4 : angle A E B = 50)
variable (h5 : angle B E D = angle B D E)

-- State the theorem
theorem angle_cde_is_85 : angle C D E = 85 := by sorry

end angle_cde_is_85_l3711_371182


namespace three_fourths_to_sixth_power_l3711_371138

theorem three_fourths_to_sixth_power : (3 / 4 : ℚ) ^ 6 = 729 / 4096 := by
  sorry

end three_fourths_to_sixth_power_l3711_371138


namespace chord_length_through_focus_l3711_371190

/-- Given a parabola y^2 = 8x, prove that a chord AB passing through the focus
    with endpoints A(x₁, y₁) and B(x₂, y₂) on the parabola, where x₁ + x₂ = 10,
    has length |AB| = 14. -/
theorem chord_length_through_focus (x₁ x₂ y₁ y₂ : ℝ) : 
  y₁^2 = 8*x₁ →  -- A is on the parabola
  y₂^2 = 8*x₂ →  -- B is on the parabola
  x₁ + x₂ = 10 → -- Given condition
  -- AB passes through the focus (2, 0)
  (y₂ - y₁) * 2 = (x₂ - x₁) * (y₂ + y₁) →
  -- The length of AB is 14
  (x₂ - x₁)^2 + (y₂ - y₁)^2 = 14^2 := by
sorry

end chord_length_through_focus_l3711_371190


namespace hotel_bubble_bath_amount_l3711_371135

/-- Calculates the total amount of bubble bath needed for a hotel --/
def total_bubble_bath_needed (luxury_suites rooms_for_couples single_rooms family_rooms : ℕ)
  (luxury_capacity couple_capacity single_capacity family_capacity : ℕ)
  (adult_bath_ml child_bath_ml : ℕ) : ℕ :=
  let total_guests := 
    luxury_suites * luxury_capacity + 
    rooms_for_couples * couple_capacity + 
    single_rooms * single_capacity + 
    family_rooms * family_capacity
  let adults := (2 * total_guests) / 3
  let children := total_guests - adults
  adults * adult_bath_ml + children * child_bath_ml

/-- The amount of bubble bath needed for the given hotel configuration --/
theorem hotel_bubble_bath_amount : 
  total_bubble_bath_needed 6 12 15 4 5 2 1 7 20 15 = 1760 := by
  sorry

end hotel_bubble_bath_amount_l3711_371135


namespace min_value_theorem_l3711_371160

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 2*x*y) :
  3*x + 4*y ≥ 5 + 2*Real.sqrt 6 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = 2*x₀*y₀ ∧ 3*x₀ + 4*y₀ = 5 + 2*Real.sqrt 6 :=
by sorry

end min_value_theorem_l3711_371160


namespace sum_of_i_powers_l3711_371147

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers :
  i^12 + i^17 + i^22 + i^27 + i^32 + i^37 = 1 + i :=
by
  sorry

-- Define the property i^4 = 1
axiom i_fourth_power : i^4 = 1

-- Define i as the imaginary unit
axiom i_squared : i^2 = -1

end sum_of_i_powers_l3711_371147


namespace gym_treadmills_l3711_371168

def gym_problem (num_gyms : ℕ) (bikes_per_gym : ℕ) (ellipticals_per_gym : ℕ) 
  (bike_cost : ℚ) (total_cost : ℚ) : Prop :=
  let treadmill_cost : ℚ := bike_cost * (3/2)
  let elliptical_cost : ℚ := treadmill_cost * 2
  let total_bike_cost : ℚ := num_gyms * bikes_per_gym * bike_cost
  let total_elliptical_cost : ℚ := num_gyms * ellipticals_per_gym * elliptical_cost
  let treadmill_cost_per_gym : ℚ := (total_cost - total_bike_cost - total_elliptical_cost) / num_gyms
  let treadmills_per_gym : ℚ := treadmill_cost_per_gym / treadmill_cost
  treadmills_per_gym = 5

theorem gym_treadmills : 
  gym_problem 20 10 5 700 455000 := by
  sorry

end gym_treadmills_l3711_371168


namespace intersection_line_of_circles_l3711_371177

/-- Circle type with center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of the line passing through the intersection points of two circles -/
def intersection_line_equation (c1 c2 : Circle) : ℝ × ℝ → Prop :=
  fun p => p.1 + p.2 = -2

/-- Theorem stating that the line passing through the intersection points of the given circles has the equation x + y = -2 -/
theorem intersection_line_of_circles :
  let c1 : Circle := { center := (-4, -10), radius := 15 }
  let c2 : Circle := { center := (8, 6), radius := Real.sqrt 104 }
  ∀ p, p ∈ { p | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 } ∩
           { p | (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2 } →
  intersection_line_equation c1 c2 p :=
by
  sorry

end intersection_line_of_circles_l3711_371177


namespace eggs_leftover_l3711_371152

theorem eggs_leftover (abigail_eggs beatrice_eggs carson_eggs : ℕ) 
  (h1 : abigail_eggs = 37)
  (h2 : beatrice_eggs = 49)
  (h3 : carson_eggs = 14) :
  (abigail_eggs + beatrice_eggs + carson_eggs) % 12 = 4 := by
  sorry

end eggs_leftover_l3711_371152


namespace olivia_wallet_remaining_l3711_371198

/-- The amount of money remaining in Olivia's wallet after shopping -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Theorem stating that Olivia has 29 dollars left in her wallet -/
theorem olivia_wallet_remaining : remaining_money 54 25 = 29 := by
  sorry

end olivia_wallet_remaining_l3711_371198


namespace sum_of_squares_l3711_371169

theorem sum_of_squares (x y z : ℝ) 
  (h_arithmetic : (x + y + z) / 3 = 9)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 405 := by
sorry

end sum_of_squares_l3711_371169


namespace correct_males_in_orchestra_l3711_371116

/-- The number of males in the orchestra -/
def males_in_orchestra : ℕ := 11

/-- The number of females in the orchestra -/
def females_in_orchestra : ℕ := 12

/-- The number of musicians in the orchestra -/
def musicians_in_orchestra : ℕ := males_in_orchestra + females_in_orchestra

/-- The number of musicians in the band -/
def musicians_in_band : ℕ := 2 * musicians_in_orchestra

/-- The number of musicians in the choir -/
def musicians_in_choir : ℕ := 12 + 17

/-- The total number of musicians in all three groups -/
def total_musicians : ℕ := 98

theorem correct_males_in_orchestra :
  musicians_in_orchestra + musicians_in_band + musicians_in_choir = total_musicians :=
sorry

end correct_males_in_orchestra_l3711_371116


namespace k_range_l3711_371117

theorem k_range (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k^2 - 1 ≤ 0) ↔ k ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) :=
by sorry

end k_range_l3711_371117


namespace june_production_l3711_371183

/-- Represents a restaurant's daily pizza and hot dog production. -/
structure RestaurantProduction where
  hotDogs : ℕ
  pizzaDifference : ℕ

/-- Calculates the total number of pizzas and hot dogs made in June. -/
def totalInJune (r : RestaurantProduction) : ℕ :=
  30 * (r.hotDogs + (r.hotDogs + r.pizzaDifference))

/-- Theorem stating the total production in June for a specific restaurant. -/
theorem june_production (r : RestaurantProduction) 
  (h1 : r.hotDogs = 60) 
  (h2 : r.pizzaDifference = 40) : 
  totalInJune r = 4800 := by
  sorry

#eval totalInJune ⟨60, 40⟩

end june_production_l3711_371183


namespace apple_bag_theorem_l3711_371176

/-- Represents the number of apples in a bag -/
inductive BagSize
  | small : BagSize  -- 6 apples
  | large : BagSize  -- 12 apples

/-- The total number of apples from all bags -/
def totalApples (bags : List BagSize) : Nat :=
  bags.foldl (fun sum bag => sum + match bag with
    | BagSize.small => 6
    | BagSize.large => 12) 0

/-- Theorem stating the possible total numbers of apples -/
theorem apple_bag_theorem (bags : List BagSize) :
  (totalApples bags ≥ 70 ∧ totalApples bags ≤ 80) →
  (totalApples bags = 72 ∨ totalApples bags = 78) := by
  sorry

end apple_bag_theorem_l3711_371176


namespace circle_area_with_diameter_8_l3711_371157

/-- The area of a circle with diameter 8 meters is 16π square meters. -/
theorem circle_area_with_diameter_8 :
  ∃ (A : ℝ), A = π * 16 ∧ A = (π * (8 / 2)^2) := by sorry

end circle_area_with_diameter_8_l3711_371157


namespace sum_of_squares_l3711_371132

theorem sum_of_squares (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 3)
  (h2 : a / x + b / y + c / z = 0)
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 := by
sorry

end sum_of_squares_l3711_371132


namespace sara_pumpkins_l3711_371170

/-- The number of pumpkins eaten by rabbits -/
def pumpkins_eaten : ℕ := 23

/-- The number of pumpkins Sara has left -/
def pumpkins_left : ℕ := 20

/-- The original number of pumpkins Sara grew -/
def original_pumpkins : ℕ := pumpkins_eaten + pumpkins_left

theorem sara_pumpkins : original_pumpkins = 43 := by
  sorry

end sara_pumpkins_l3711_371170


namespace cosine_sum_identity_l3711_371172

theorem cosine_sum_identity (α : ℝ) : 
  Real.cos (π/4 - α) * Real.cos (α + π/12) - Real.sin (π/4 - α) * Real.sin (α + π/12) = 1/2 := by
  sorry

end cosine_sum_identity_l3711_371172


namespace park_short_trees_l3711_371127

def initial_short_trees : ℕ := 3
def short_trees_to_plant : ℕ := 9
def final_short_trees : ℕ := 12

theorem park_short_trees :
  initial_short_trees + short_trees_to_plant = final_short_trees :=
by sorry

end park_short_trees_l3711_371127


namespace marias_carrots_l3711_371162

theorem marias_carrots (initial thrown_out picked_more final : ℕ) : 
  thrown_out = 11 →
  picked_more = 15 →
  final = 52 →
  initial - thrown_out + picked_more = final →
  initial = 48 := by
sorry

end marias_carrots_l3711_371162


namespace equation_solution_l3711_371159

theorem equation_solution : ∃ t : ℝ, t = 9/4 ∧ Real.sqrt (3 * Real.sqrt (t - 1)) = (t + 9) ^ (1/4) :=
  sorry

end equation_solution_l3711_371159


namespace sally_bread_consumption_l3711_371150

/-- The number of sandwiches Sally eats on Saturday -/
def saturday_sandwiches : ℕ := 2

/-- The number of sandwiches Sally eats on Sunday -/
def sunday_sandwiches : ℕ := 1

/-- The number of pieces of bread used in each sandwich -/
def bread_per_sandwich : ℕ := 2

/-- The total number of pieces of bread Sally eats across Saturday and Sunday -/
def total_bread : ℕ := (saturday_sandwiches + sunday_sandwiches) * bread_per_sandwich

theorem sally_bread_consumption :
  total_bread = 6 :=
by sorry

end sally_bread_consumption_l3711_371150


namespace gina_remaining_money_l3711_371178

def initial_amount : ℚ := 400

def mom_fraction : ℚ := 1/4
def clothes_fraction : ℚ := 1/8
def charity_fraction : ℚ := 1/5

def remaining_amount : ℚ := initial_amount * (1 - mom_fraction - clothes_fraction - charity_fraction)

theorem gina_remaining_money :
  remaining_amount = 170 := by sorry

end gina_remaining_money_l3711_371178


namespace richard_cleaning_time_l3711_371185

/-- Richard's room cleaning time in minutes -/
def richard_time : ℕ := 45

/-- Cory's room cleaning time in minutes -/
def cory_time (r : ℕ) : ℕ := r + 3

/-- Blake's room cleaning time in minutes -/
def blake_time (r : ℕ) : ℕ := cory_time r - 4

/-- Total cleaning time for all three people in minutes -/
def total_time : ℕ := 136

theorem richard_cleaning_time :
  richard_time + cory_time richard_time + blake_time richard_time = total_time :=
sorry

end richard_cleaning_time_l3711_371185


namespace point_on_135_degree_angle_l3711_371175

/-- Given a point (√4, a) on the terminal side of the angle 135°, prove that a = 2 -/
theorem point_on_135_degree_angle (a : ℝ) : 
  (∃ (x y : ℝ), x = Real.sqrt 4 ∧ y = a ∧ 
   x = 2 * Real.cos (135 * π / 180) ∧ 
   y = 2 * Real.sin (135 * π / 180)) → 
  a = 2 := by
sorry

end point_on_135_degree_angle_l3711_371175


namespace machine_production_theorem_l3711_371180

/-- Given that 4 machines produce x units in 6 days at a constant rate,
    prove that 16 machines will produce 2x units in 3 days. -/
theorem machine_production_theorem 
  (x : ℝ) -- x is the number of units produced by 4 machines in 6 days
  (h1 : x > 0) -- x is positive
  : 
  let rate := x / (4 * 6) -- rate of production per machine per day
  16 * rate * 3 = 2 * x := by
sorry

end machine_production_theorem_l3711_371180


namespace complex_power_sum_l3711_371161

theorem complex_power_sum (z : ℂ) (h : z + 1 / z = 2 * Real.cos (5 * π / 180)) :
  z^100 + 1 / z^100 = 2 * Real.cos (140 * π / 180) := by
  sorry

end complex_power_sum_l3711_371161


namespace sixteen_points_divide_square_into_ten_equal_triangles_l3711_371103

/-- Represents a point inside a unit square -/
structure PointInSquare where
  x : Real
  y : Real
  inside : 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1

/-- Represents the areas of the four triangles formed by a point and the square's sides -/
structure TriangleAreas where
  a₁ : ℕ
  a₂ : ℕ
  a₃ : ℕ
  a₄ : ℕ
  sum_is_ten : a₁ + a₂ + a₃ + a₄ = 10
  all_positive : 1 ≤ a₁ ∧ 1 ≤ a₂ ∧ 1 ≤ a₃ ∧ 1 ≤ a₄
  all_at_most_four : a₁ ≤ 4 ∧ a₂ ≤ 4 ∧ a₃ ≤ 4 ∧ a₄ ≤ 4

/-- The main theorem stating that there are exactly 16 points satisfying the condition -/
theorem sixteen_points_divide_square_into_ten_equal_triangles :
  ∃ (points : Finset PointInSquare),
    points.card = 16 ∧
    (∀ p ∈ points, ∃ (areas : TriangleAreas), True) ∧
    (∀ p : PointInSquare, p ∉ points → ¬∃ (areas : TriangleAreas), True) := by
  sorry


end sixteen_points_divide_square_into_ten_equal_triangles_l3711_371103


namespace sales_solution_l3711_371105

def sales_problem (m1 m3 m4 m5 m6 avg : ℕ) : Prop :=
  ∃ m2 : ℕ, 
    (m1 + m2 + m3 + m4 + m5 + m6) / 6 = avg ∧
    m2 = 5744

theorem sales_solution :
  sales_problem 5266 5864 6122 6588 4916 5750 :=
by sorry

end sales_solution_l3711_371105


namespace system_implies_sum_l3711_371194

theorem system_implies_sum (x y m : ℝ) : x + m = 4 → y - 5 = m → x + y = 9 := by sorry

end system_implies_sum_l3711_371194


namespace correct_tax_distribution_l3711_371179

/-- Represents the tax calculation for three individuals based on their yields -/
def tax_calculation (total_tax : ℚ) (yield1 yield2 yield3 : ℕ) : Prop :=
  let total_yield := yield1 + yield2 + yield3
  let tax1 := total_tax * (yield1 : ℚ) / total_yield
  let tax2 := total_tax * (yield2 : ℚ) / total_yield
  let tax3 := total_tax * (yield3 : ℚ) / total_yield
  (tax1 = 1 + 3/32) ∧ (tax2 = 1 + 1/4) ∧ (tax3 = 1 + 13/32)

/-- Theorem stating the correct tax distribution for the given problem -/
theorem correct_tax_distribution :
  tax_calculation (15/4) 7 8 9 := by
  sorry

end correct_tax_distribution_l3711_371179


namespace coefficient_x_squared_in_binomial_expansion_l3711_371163

theorem coefficient_x_squared_in_binomial_expansion :
  let n : ℕ := 8
  let k : ℕ := 3
  let coeff : ℤ := (-1)^k * 2^k * Nat.choose n k
  coeff = -448 := by sorry

end coefficient_x_squared_in_binomial_expansion_l3711_371163


namespace car_speed_second_hour_l3711_371173

/-- Given a car traveling for two hours with a speed of 80 km/h in the first hour
    and an average speed of 60 km/h over the two hours,
    prove that the speed in the second hour must be 40 km/h. -/
theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (average_speed : ℝ)
  (h1 : speed_first_hour = 80)
  (h2 : average_speed = 60) :
  let speed_second_hour := 2 * average_speed - speed_first_hour
  speed_second_hour = 40 := by
sorry


end car_speed_second_hour_l3711_371173


namespace typing_time_together_l3711_371181

-- Define the typing rates for Randy and Candy
def randy_rate : ℚ := 1 / 30
def candy_rate : ℚ := 1 / 45

-- Define the combined typing rate
def combined_rate : ℚ := randy_rate + candy_rate

-- Theorem to prove
theorem typing_time_together : (1 : ℚ) / combined_rate = 18 := by
  sorry

end typing_time_together_l3711_371181


namespace x_squared_minus_y_squared_l3711_371123

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 8/15) (h2 : x - y = 1/45) : x^2 - y^2 = 8/675 := by
  sorry

end x_squared_minus_y_squared_l3711_371123


namespace absolute_value_simplification_l3711_371196

theorem absolute_value_simplification : |-4^2 + 5 - 2| = 13 := by
  sorry

end absolute_value_simplification_l3711_371196


namespace token_passing_game_termination_l3711_371119

/-- Represents the state of the token-passing game -/
structure GameState where
  tokens : Fin 1994 → ℕ
  total_tokens : ℕ

/-- Defines a single move in the game -/
def make_move (state : GameState) (i : Fin 1994) : GameState :=
  sorry

/-- Predicate to check if the game has terminated -/
def is_terminated (state : GameState) : Prop :=
  ∀ i : Fin 1994, state.tokens i ≤ 1

/-- The main theorem about the token-passing game -/
theorem token_passing_game_termination 
  (n : ℕ) (initial_state : GameState) 
  (h_initial : ∃ i : Fin 1994, initial_state.tokens i = n ∧ 
               ∀ j : Fin 1994, j ≠ i → initial_state.tokens j = 0) 
  (h_total : initial_state.total_tokens = n) :
  (n < 1994 → ∃ (final_state : GameState), is_terminated final_state) ∧
  (n = 1994 → ∀ (state : GameState), ¬is_terminated state) :=
sorry

end token_passing_game_termination_l3711_371119


namespace inequality_solution_set_l3711_371166

theorem inequality_solution_set (a b c : ℝ) :
  (∀ x : ℝ, a * x + b > c ↔ x < 4) →
  (∀ x : ℝ, a * (x - 3) + b > c ↔ x < 7) :=
by sorry

end inequality_solution_set_l3711_371166


namespace max_type_A_books_l3711_371122

/-- Represents the unit price of type A books -/
def price_A : ℝ := 20

/-- Represents the unit price of type B books -/
def price_B : ℝ := 15

/-- Represents the total number of books to be purchased -/
def total_books : ℕ := 300

/-- Represents the discount factor for type A books -/
def discount_A : ℝ := 0.9

/-- Represents the maximum total cost -/
def max_cost : ℝ := 5100

/-- Theorem stating the maximum number of type A books that can be purchased -/
theorem max_type_A_books : 
  ∃ (n : ℕ), n ≤ total_books ∧ 
  discount_A * price_A * n + price_B * (total_books - n) ≤ max_cost ∧
  ∀ (m : ℕ), m > n → discount_A * price_A * m + price_B * (total_books - m) > max_cost :=
sorry

end max_type_A_books_l3711_371122


namespace triangle_area_l3711_371141

/-- Given a triangle with perimeter 24 cm and inradius 2.5 cm, its area is 30 cm². -/
theorem triangle_area (P : ℝ) (r : ℝ) (A : ℝ) : 
  P = 24 → r = 2.5 → A = r * (P / 2) → A = 30 := by
  sorry

end triangle_area_l3711_371141


namespace inequality_proof_l3711_371153

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  Real.sqrt (a * b) ≤ (1 / 3) * Real.sqrt ((a^2 + b^2) / 2) + (2 / 3) * (2 / (1 / a + 1 / b)) :=
by sorry

end inequality_proof_l3711_371153


namespace unique_solution_k_squared_minus_2016_equals_3_to_n_l3711_371120

theorem unique_solution_k_squared_minus_2016_equals_3_to_n :
  ∃! (k n : ℕ), k > 0 ∧ n > 0 ∧ k^2 - 2016 = 3^n := by sorry

end unique_solution_k_squared_minus_2016_equals_3_to_n_l3711_371120


namespace expression_1_expression_2_expression_3_expression_4_expression_5_l3711_371197

-- Expression 1
theorem expression_1 : 0.11 * 1.8 + 8.2 * 0.11 = 1.1 := by sorry

-- Expression 2
theorem expression_2 : 0.8 * (3.2 - 2.99 / 2.3) = 1.52 := by sorry

-- Expression 3
theorem expression_3 : 3.5 - 3.5 * 0.98 = 0.07 := by sorry

-- Expression 4
theorem expression_4 : 12.5 * 2.5 * 3.2 = 100 := by sorry

-- Expression 5
theorem expression_5 : (8.1 - 5.4) / 3.6 + 85.7 = 86.45 := by sorry

end expression_1_expression_2_expression_3_expression_4_expression_5_l3711_371197


namespace megan_markers_theorem_l3711_371155

/-- The number of markers Megan had initially -/
def initial_markers : ℕ := sorry

/-- The number of markers Robert gave to Megan -/
def roberts_markers : ℕ := 109

/-- The total number of markers Megan has now -/
def total_markers : ℕ := 326

/-- Theorem stating that the initial number of markers plus the markers given by Robert equals the total number of markers Megan has now -/
theorem megan_markers_theorem : initial_markers + roberts_markers = total_markers := by sorry

end megan_markers_theorem_l3711_371155


namespace f_extrema_l3711_371156

-- Define the function f
def f (x y : ℝ) : ℝ := x^3 + y^3 + 6*x*y

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) :=
  {p | -3 ≤ p.1 ∧ p.1 ≤ 1 ∧ -3 ≤ p.2 ∧ p.2 ≤ 2}

theorem f_extrema :
  ∃ (min_point max_point : ℝ × ℝ),
    min_point ∈ rectangle ∧
    max_point ∈ rectangle ∧
    (∀ p ∈ rectangle, f min_point.1 min_point.2 ≤ f p.1 p.2) ∧
    (∀ p ∈ rectangle, f p.1 p.2 ≤ f max_point.1 max_point.2) ∧
    min_point = (-3, 2) ∧
    max_point = (1, 2) ∧
    f min_point.1 min_point.2 = -55 ∧
    f max_point.1 max_point.2 = 21 :=
  sorry


end f_extrema_l3711_371156


namespace parabola_line_intersection_l3711_371192

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola x^2 = 12y -/
def Parabola := {p : Point | p.x^2 = 12 * p.y}

/-- Represents a line y = kx + m -/
def Line (k m : ℝ) := {p : Point | p.y = k * p.x + m}

/-- The focus of the parabola -/
def focus : Point := ⟨0, 3⟩

theorem parabola_line_intersection (k m : ℝ) (h_k : k > 0) :
  ∃ A B : Point,
    A ∈ Parabola ∧
    B ∈ Parabola ∧
    A ∈ Line k m ∧
    B ∈ Line k m ∧
    focus ∈ Line k m ∧
    (A.x - B.x)^2 + (A.y - B.y)^2 = 36^2 →
    k = Real.sqrt 2 := by
  sorry

end parabola_line_intersection_l3711_371192


namespace nell_card_difference_l3711_371129

/-- Represents Nell's card collection --/
structure CardCollection where
  initial_baseball : Nat
  initial_ace : Nat
  current_baseball : Nat
  current_ace : Nat

/-- Calculates the difference between baseball and Ace cards --/
def card_difference (c : CardCollection) : Int :=
  c.current_baseball - c.current_ace

/-- Theorem stating the difference between Nell's baseball and Ace cards --/
theorem nell_card_difference (nell : CardCollection)
  (h1 : nell.initial_baseball = 438)
  (h2 : nell.initial_ace = 18)
  (h3 : nell.current_baseball = 178)
  (h4 : nell.current_ace = 55) :
  card_difference nell = 123 := by
  sorry

end nell_card_difference_l3711_371129


namespace langsley_commute_time_l3711_371167

theorem langsley_commute_time :
  let first_bus_time : ℕ := 40
  let first_bus_delay : ℕ := 10
  let first_wait_time : ℕ := 10
  let second_bus_time : ℕ := 50
  let second_bus_delay : ℕ := 5
  let second_wait_time : ℕ := 15
  let third_bus_time : ℕ := 95
  let third_bus_delay : ℕ := 15
  first_bus_time + first_bus_delay + first_wait_time +
  second_bus_time + second_bus_delay + second_wait_time +
  third_bus_time + third_bus_delay = 240 := by
sorry

end langsley_commute_time_l3711_371167
