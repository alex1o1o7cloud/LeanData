import Mathlib

namespace quadratic_root_relationship_l2483_248307

/-- Given two quadratic equations and their relationship, prove the ratio of their coefficients -/
theorem quadratic_root_relationship (m n p : ℝ) : 
  m ≠ 0 → n ≠ 0 → p ≠ 0 →
  (∃ r₁ r₂ : ℝ, (r₁ + r₂ = -p ∧ r₁ * r₂ = m) ∧
               (3*r₁ + 3*r₂ = -m ∧ 9*r₁*r₂ = n)) →
  n/p = 27 := by
sorry

end quadratic_root_relationship_l2483_248307


namespace probability_of_shared_character_l2483_248326

/-- Represents an idiom card -/
structure IdiomCard where
  idiom : String

/-- The set of all idiom cards -/
def idiomCards : Finset IdiomCard := sorry

/-- Two cards share a character -/
def shareCharacter (card1 card2 : IdiomCard) : Prop := sorry

/-- The number of ways to choose 2 cards from the set -/
def totalChoices : Nat := Nat.choose idiomCards.card 2

/-- The number of ways to choose 2 cards that share a character -/
def favorableChoices : Nat := sorry

theorem probability_of_shared_character :
  (favorableChoices : ℚ) / totalChoices = 2 / 5 := by sorry

end probability_of_shared_character_l2483_248326


namespace system_one_solution_system_two_solution_l2483_248388

-- System (1)
theorem system_one_solution (x y : ℝ) : 
  x + y = 3 ∧ x - y = 1 → x = 2 ∧ y = 1 := by sorry

-- System (2)
theorem system_two_solution (x y : ℝ) : 
  x/2 - (y+1)/3 = 1 ∧ 3*x + 2*y = 10 → x = 3 ∧ y = 1/2 := by sorry

end system_one_solution_system_two_solution_l2483_248388


namespace divisibility_by_forty_l2483_248332

theorem divisibility_by_forty (p : ℕ) (h_prime : Prime p) (h_ge_seven : p ≥ 7) :
  (∃ q : ℕ, Prime q ∧ q ≥ 7 ∧ 40 ∣ (q^2 - 1)) ∧
  (∃ r : ℕ, Prime r ∧ r ≥ 7 ∧ ¬(40 ∣ (r^2 - 1))) :=
by sorry

end divisibility_by_forty_l2483_248332


namespace only_zhong_symmetric_l2483_248319

-- Define a type for Chinese characters
inductive ChineseCharacter : Type
  | ai   : ChineseCharacter  -- 爱
  | wo   : ChineseCharacter  -- 我
  | zhong : ChineseCharacter  -- 中
  | hua  : ChineseCharacter  -- 华

-- Define a property for vertical symmetry
def hasVerticalSymmetry (c : ChineseCharacter) : Prop :=
  match c with
  | ChineseCharacter.zhong => True
  | _ => False

-- Theorem stating that only 中 (zhong) has vertical symmetry
theorem only_zhong_symmetric :
  ∀ (c : ChineseCharacter),
    hasVerticalSymmetry c ↔ c = ChineseCharacter.zhong :=
by
  sorry


end only_zhong_symmetric_l2483_248319


namespace complex_point_to_number_l2483_248318

theorem complex_point_to_number (z : ℂ) : (z / Complex.I).re = 3 ∧ (z / Complex.I).im = -1 → z = 1 + 3 * Complex.I := by
  sorry

end complex_point_to_number_l2483_248318


namespace map_distance_calculation_l2483_248339

/-- Given a map scale and actual distances, calculate the map distance --/
theorem map_distance_calculation 
  (map_distance_mountains : ℝ) 
  (actual_distance_mountains : ℝ) 
  (actual_distance_ram : ℝ) :
  let scale := actual_distance_mountains / map_distance_mountains
  actual_distance_ram / scale = map_distance_mountains * (actual_distance_ram / actual_distance_mountains) :=
by sorry

end map_distance_calculation_l2483_248339


namespace cubic_extremum_difference_l2483_248368

/-- A cubic function with specific properties -/
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + 3*b*x + c

/-- The derivative of f -/
def f' (a b : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x + 3*b

theorem cubic_extremum_difference (a b c : ℝ) :
  f' a b 2 = 0 → f' a b 1 = -3 →
  ∃ (min_val : ℝ), ∀ (x : ℝ), f a b c x ≥ min_val ∧ 
  ∀ (M : ℝ), ∃ (y : ℝ), f a b c y > M :=
by sorry

end cubic_extremum_difference_l2483_248368


namespace geometric_sequence_increasing_condition_l2483_248311

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

def monotonically_increasing (a : ℕ → ℝ) :=
  ∀ n m, n < m → a n < a m

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  (¬ (q > 1 → monotonically_increasing a) ∧ ¬ (monotonically_increasing a → q > 1)) :=
by sorry

end geometric_sequence_increasing_condition_l2483_248311


namespace original_class_strength_l2483_248391

theorem original_class_strength (original_average : ℝ) (new_students : ℕ) 
  (new_average : ℝ) (average_decrease : ℝ) :
  original_average = 40 →
  new_students = 12 →
  new_average = 32 →
  average_decrease = 4 →
  ∃ x : ℕ, x = 12 ∧ 
    (x + new_students : ℝ) * (original_average - average_decrease) = 
    x * original_average + (new_students : ℝ) * new_average :=
by sorry

end original_class_strength_l2483_248391


namespace B_join_time_l2483_248320

/-- Represents the time (in months) when B joined the business -/
def time_B_joined : ℝ := 7.5

/-- A's initial investment -/
def A_investment : ℝ := 27000

/-- B's investment when joining -/
def B_investment : ℝ := 36000

/-- Total duration of the business in months -/
def total_duration : ℝ := 12

/-- Ratio of A's profit share to B's profit share -/
def profit_ratio : ℝ := 2

theorem B_join_time :
  (A_investment * total_duration) / (B_investment * (total_duration - time_B_joined)) = profit_ratio := by
  sorry

end B_join_time_l2483_248320


namespace probability_10_heads_in_12_flips_l2483_248361

/-- The probability of getting exactly 10 heads in 12 flips of a fair coin -/
theorem probability_10_heads_in_12_flips : 
  (Nat.choose 12 10 : ℚ) / 2^12 = 66 / 4096 := by sorry

end probability_10_heads_in_12_flips_l2483_248361


namespace complex_roots_circle_l2483_248363

theorem complex_roots_circle (z : ℂ) : 
  (z + 2)^6 = 64 * z^6 → Complex.abs (z - (-2/3)) = 2 / Real.sqrt 3 :=
by sorry

end complex_roots_circle_l2483_248363


namespace quadrilateral_angle_sum_l2483_248337

theorem quadrilateral_angle_sum (a b c d : ℕ) : 
  50 ≤ a ∧ a ≤ 200 ∧
  50 ≤ b ∧ b ≤ 200 ∧
  50 ≤ c ∧ c ≤ 200 ∧
  50 ≤ d ∧ d ≤ 200 ∧
  b = 75 ∧ c = 80 ∧ d = 120 ∧
  a + b + c + d = 360 →
  a = 85 := by
sorry

end quadrilateral_angle_sum_l2483_248337


namespace no_solution_implies_positive_b_l2483_248316

theorem no_solution_implies_positive_b (a b : ℝ) :
  (∀ x y : ℝ, y ≠ x^2 + a*x + b ∨ x ≠ y^2 + a*y + b) →
  b > 0 := by
  sorry

end no_solution_implies_positive_b_l2483_248316


namespace system_solution_ratio_l2483_248356

/-- Given a system of linear equations with a specific k value, 
    prove that xz/y^2 equals a specific constant --/
theorem system_solution_ratio (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (eq1 : x + (16/5)*y + 4*z = 0)
  (eq2 : 3*x + (16/5)*y + z = 0)
  (eq3 : 2*x + 4*y + 3*z = 0) :
  ∃ (c : ℝ), x*z/y^2 = c :=
by sorry

end system_solution_ratio_l2483_248356


namespace cubic_unique_solution_iff_l2483_248313

/-- The cubic equation in x with parameter a -/
def cubic_equation (a x : ℝ) : ℝ := x^3 - a*x^2 - 3*a*x + a^2 - 1

/-- The property that the cubic equation has exactly one real solution -/
def has_unique_real_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, cubic_equation a x = 0

theorem cubic_unique_solution_iff (a : ℝ) :
  has_unique_real_solution a ↔ a < -5/4 :=
sorry

end cubic_unique_solution_iff_l2483_248313


namespace alicia_tax_payment_l2483_248390

/-- Calculates the local tax amount in cents given an hourly wage in dollars and a tax rate as a percentage. -/
def local_tax_cents (hourly_wage : ℚ) (tax_rate : ℚ) : ℚ :=
  hourly_wage * 100 * (tax_rate / 100)

/-- Proves that Alicia's local tax payment is 50 cents per hour. -/
theorem alicia_tax_payment :
  local_tax_cents 25 2 = 50 := by
  sorry

#eval local_tax_cents 25 2

end alicia_tax_payment_l2483_248390


namespace regular_hexagon_properties_l2483_248394

/-- A regular hexagon inscribed in a circle -/
structure RegularHexagon where
  /-- The side length of the hexagon -/
  side_length : ℝ
  /-- The radius of the circumscribed circle -/
  radius : ℝ
  /-- The circumference of the circumscribed circle -/
  circumference : ℝ
  /-- The arc length corresponding to one side of the hexagon -/
  arc_length : ℝ
  /-- The area of the hexagon -/
  area : ℝ

/-- Properties of a regular hexagon with side length 6 -/
theorem regular_hexagon_properties :
  ∃ (h : RegularHexagon),
    h.side_length = 6 ∧
    h.radius = 6 ∧
    h.circumference = 12 * Real.pi ∧
    h.arc_length = 2 * Real.pi ∧
    h.area = 54 * Real.sqrt 3 := by
  sorry

end regular_hexagon_properties_l2483_248394


namespace cattle_problem_l2483_248344

/-- Represents the problem of determining the number of cattle that died --/
theorem cattle_problem (initial_cattle : ℕ) (initial_price : ℕ) (price_reduction : ℕ) (total_loss : ℕ) : 
  initial_cattle = 340 →
  initial_price = 204000 →
  price_reduction = 150 →
  total_loss = 25200 →
  ∃ (dead_cattle : ℕ), 
    dead_cattle = 57 ∧ 
    (initial_cattle - dead_cattle) * (initial_price / initial_cattle - price_reduction) = initial_price - total_loss := by
  sorry


end cattle_problem_l2483_248344


namespace polynomial_product_existence_l2483_248304

/-- Polynomial with integer coefficients and bounded absolute values -/
def BoundedPolynomial (n : ℕ) (bound : ℕ) := {p : Polynomial ℤ // ∀ i, i ≤ n → |p.coeff i| ≤ bound}

/-- The main theorem -/
theorem polynomial_product_existence 
  (f : BoundedPolynomial 5 4)
  (g : BoundedPolynomial 3 1)
  (h : BoundedPolynomial 2 1)
  (h1 : (f.val).eval 10 = (g.val).eval 10 * (h.val).eval 10) :
  ∃ (f' : Polynomial ℤ), ∀ x, f'.eval x = (g.val).eval x * (h.val).eval x := by
  sorry

end polynomial_product_existence_l2483_248304


namespace buying_100_tickets_may_not_win_l2483_248382

/-- Represents a lottery with a given number of tickets and winning probability per ticket -/
structure Lottery where
  totalTickets : ℕ
  winningProbability : ℝ
  winningProbability_nonneg : 0 ≤ winningProbability
  winningProbability_le_one : winningProbability ≤ 1

/-- The probability of not winning when buying a certain number of tickets -/
def probNotWinning (lottery : Lottery) (ticketsBought : ℕ) : ℝ :=
  (1 - lottery.winningProbability) ^ ticketsBought

/-- Theorem stating that buying 100 tickets in the given lottery may not result in a win -/
theorem buying_100_tickets_may_not_win (lottery : Lottery)
  (h1 : lottery.totalTickets = 100000)
  (h2 : lottery.winningProbability = 0.01) :
  probNotWinning lottery 100 > 0 := by
  sorry

#check buying_100_tickets_may_not_win

end buying_100_tickets_may_not_win_l2483_248382


namespace money_distribution_l2483_248366

theorem money_distribution (A B C : ℕ) 
  (total : A + B + C = 400)
  (AC_sum : A + C = 300)
  (C_amount : C = 50) : 
  B + C = 150 := by
  sorry

end money_distribution_l2483_248366


namespace journey_speed_calculation_l2483_248367

/-- Proves that given a journey of 448 km completed in 20 hours, where the first half is traveled at 21 km/hr, the speed for the second half must be 24 km/hr. -/
theorem journey_speed_calculation (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) (second_half_speed : ℝ) :
  total_distance = 448 →
  total_time = 20 →
  first_half_speed = 21 →
  (total_distance / 2) / first_half_speed + (total_distance / 2) / second_half_speed = total_time →
  second_half_speed = 24 := by
  sorry

#check journey_speed_calculation

end journey_speed_calculation_l2483_248367


namespace eight_routes_A_to_B_l2483_248376

/-- The number of different routes from A to B, given that all routes must pass through C -/
def routes_A_to_B (roads_A_to_C roads_C_to_B : ℕ) : ℕ :=
  roads_A_to_C * roads_C_to_B

/-- Theorem stating that there are 8 different routes from A to B -/
theorem eight_routes_A_to_B :
  routes_A_to_B 4 2 = 8 := by
  sorry

end eight_routes_A_to_B_l2483_248376


namespace range_of_m_l2483_248389

def p (m : ℝ) : Prop := ∃ x₀ : ℝ, m * x₀^2 + 1 < 1

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

theorem range_of_m : ∀ m : ℝ, (¬(p m ∨ ¬(q m))) ↔ m ∈ Set.Icc (-2) 2 := by
  sorry

end range_of_m_l2483_248389


namespace solution_value_l2483_248378

theorem solution_value (x a : ℝ) (h : x = 3 ∧ 2 * x - 10 = 4 * a) : a = -1 := by
  sorry

end solution_value_l2483_248378


namespace power_sum_and_division_l2483_248383

theorem power_sum_and_division (a b c : ℕ) :
  2^345 + 9^5 / 9^3 = 2^345 + 81 := by
  sorry

end power_sum_and_division_l2483_248383


namespace abs_gt_one_necessary_not_sufficient_product_nonzero_iff_both_nonzero_l2483_248328

-- Theorem for Option A
theorem abs_gt_one_necessary_not_sufficient :
  (∀ x : ℝ, x > 1 → |x| > 1) ∧
  (∃ x : ℝ, |x| > 1 ∧ x ≤ 1) :=
sorry

-- Theorem for Option C
theorem product_nonzero_iff_both_nonzero (a b : ℝ) :
  a * b ≠ 0 ↔ a ≠ 0 ∧ b ≠ 0 :=
sorry

end abs_gt_one_necessary_not_sufficient_product_nonzero_iff_both_nonzero_l2483_248328


namespace inverse_in_S_l2483_248336

-- Define the set S
variable (S : Set ℝ)

-- Define the properties of S
variable (h1 : Set.Subset (Set.range (Int.cast : ℤ → ℝ)) S)
variable (h2 : (Real.sqrt 2 + Real.sqrt 3) ∈ S)
variable (h3 : ∀ x y, x ∈ S → y ∈ S → (x + y) ∈ S)
variable (h4 : ∀ x y, x ∈ S → y ∈ S → (x * y) ∈ S)

-- Theorem statement
theorem inverse_in_S : (Real.sqrt 2 + Real.sqrt 3)⁻¹ ∈ S := by
  sorry

end inverse_in_S_l2483_248336


namespace shirt_cost_l2483_248370

def flat_rate_shipping : ℝ := 5
def shipping_rate : ℝ := 0.2
def shipping_threshold : ℝ := 50
def socks_price : ℝ := 5
def shorts_price : ℝ := 15
def swim_trunks_price : ℝ := 14
def total_bill : ℝ := 102
def shorts_quantity : ℕ := 2

theorem shirt_cost (shirt_price : ℝ) : 
  (shirt_price + socks_price + shorts_quantity * shorts_price + swim_trunks_price > shipping_threshold) →
  (shirt_price + socks_price + shorts_quantity * shorts_price + swim_trunks_price) * 
    (1 + shipping_rate) = total_bill →
  shirt_price = 36 := by
sorry

end shirt_cost_l2483_248370


namespace binomial_26_6_l2483_248360

theorem binomial_26_6 (h1 : Nat.choose 23 5 = 33649) 
                       (h2 : Nat.choose 23 6 = 33649)
                       (h3 : Nat.choose 25 5 = 53130) : 
  Nat.choose 26 6 = 163032 := by
  sorry

end binomial_26_6_l2483_248360


namespace range_of_s_l2483_248372

-- Define the type for composite positive integers
def CompositePositiveInteger := {n : ℕ | n > 1 ∧ ¬ Prime n}

-- Define the function s
def s : CompositePositiveInteger → ℕ :=
  sorry -- Definition of s as sum of distinct prime factors

-- State the theorem about the range of s
theorem range_of_s :
  ∀ m : ℕ, m ≥ 2 ↔ ∃ n : CompositePositiveInteger, s n = m :=
sorry

end range_of_s_l2483_248372


namespace sequence_monotonicity_l2483_248329

theorem sequence_monotonicity (b : ℝ) :
  (∀ n : ℕ, n^2 + b*n < (n+1)^2 + b*(n+1)) ↔ b > -3 :=
sorry

end sequence_monotonicity_l2483_248329


namespace michael_fish_count_l2483_248314

def total_pets : ℕ := 160
def dog_percentage : ℚ := 225 / 1000
def cat_percentage : ℚ := 375 / 1000
def bunny_percentage : ℚ := 15 / 100
def bird_percentage : ℚ := 1 / 10

theorem michael_fish_count :
  let dogs := (dog_percentage * total_pets).floor
  let cats := (cat_percentage * total_pets).floor
  let bunnies := (bunny_percentage * total_pets).floor
  let birds := (bird_percentage * total_pets).floor
  let fish := total_pets - (dogs + cats + bunnies + birds)
  fish = 24 := by
sorry

end michael_fish_count_l2483_248314


namespace unique_solution_absolute_value_equation_l2483_248381

theorem unique_solution_absolute_value_equation :
  ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| :=
by
  -- The unique solution is x = 8/3
  use 8/3
  constructor
  · -- Prove that 8/3 satisfies the equation
    sorry
  · -- Prove that any solution must equal 8/3
    sorry

end unique_solution_absolute_value_equation_l2483_248381


namespace relationship_between_a_b_l2483_248398

theorem relationship_between_a_b (a b : ℝ) (h1 : a + b < 0) (h2 : b > 0) :
  a < -b ∧ -b < b ∧ b < -a := by sorry

end relationship_between_a_b_l2483_248398


namespace equation_D_is_quadratic_l2483_248306

/-- A quadratic equation in x is of the form ax² + bx + c = 0, where a ≠ 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The equation x² - x = 0 -/
def equation_D : QuadraticEquation where
  a := 1
  b := -1
  c := 0
  a_nonzero := by sorry

theorem equation_D_is_quadratic : equation_D.a ≠ 0 ∧ 
  equation_D.a * X^2 + equation_D.b * X + equation_D.c = X^2 - X := by sorry


end equation_D_is_quadratic_l2483_248306


namespace sqrt_neg_three_squared_l2483_248305

theorem sqrt_neg_three_squared : Real.sqrt ((-3)^2) = 3 := by
  sorry

end sqrt_neg_three_squared_l2483_248305


namespace probability_red_or_white_is_five_sixths_l2483_248352

def total_marbles : ℕ := 30
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9

def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

def probability_red_or_white : ℚ :=
  (red_marbles + white_marbles : ℚ) / total_marbles

theorem probability_red_or_white_is_five_sixths :
  probability_red_or_white = 5 / 6 := by
  sorry

end probability_red_or_white_is_five_sixths_l2483_248352


namespace farmland_area_l2483_248392

theorem farmland_area (lizzie_group_area other_group_area remaining_area : ℕ) 
  (h1 : lizzie_group_area = 250)
  (h2 : other_group_area = 265)
  (h3 : remaining_area = 385) :
  lizzie_group_area + other_group_area + remaining_area = 900 := by
  sorry

end farmland_area_l2483_248392


namespace brianna_reread_books_l2483_248322

/-- The number of old books Brianna needs to reread in a year --/
def old_books_to_reread (books_per_month : ℕ) (months_in_year : ℕ) (gifted_books : ℕ) (bought_books : ℕ) (borrowed_books_difference : ℕ) : ℕ :=
  let total_books_needed := books_per_month * months_in_year
  let new_books := gifted_books + bought_books + (bought_books - borrowed_books_difference)
  total_books_needed - new_books

/-- Theorem stating the number of old books Brianna needs to reread --/
theorem brianna_reread_books : 
  old_books_to_reread 2 12 6 8 2 = 4 := by
  sorry

end brianna_reread_books_l2483_248322


namespace smallest_portion_is_ten_l2483_248384

/-- Represents the distribution of bread loaves -/
structure BreadDistribution where
  a : ℕ  -- smallest portion (first term of arithmetic sequence)
  d : ℕ  -- common difference of arithmetic sequence

/-- The problem of distributing bread loaves -/
def breadProblem (bd : BreadDistribution) : Prop :=
  -- Total sum is 100
  (5 * bd.a + 10 * bd.d = 100) ∧
  -- Sum of larger three portions is 1/3 of sum of smaller two portions
  (3 * bd.a + 9 * bd.d = (2 * bd.a + bd.d) / 3)

/-- Theorem stating the smallest portion is 10 -/
theorem smallest_portion_is_ten :
  ∃ (bd : BreadDistribution), breadProblem bd ∧ bd.a = 10 :=
by sorry

end smallest_portion_is_ten_l2483_248384


namespace square_sum_from_difference_and_product_l2483_248321

theorem square_sum_from_difference_and_product (x y : ℝ) 
  (h1 : x - y = 18) 
  (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := by
sorry

end square_sum_from_difference_and_product_l2483_248321


namespace inequality_proof_l2483_248373

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) (hne : a ≠ b) :
  Real.sqrt a + Real.sqrt b < Real.sqrt 2 ∧ Real.sqrt 2 < 1 / (2^a) + 1 / (2^b) := by
  sorry

end inequality_proof_l2483_248373


namespace roxy_garden_plants_l2483_248309

def calculate_remaining_plants (initial_flowering : ℕ) (saturday_flowering : ℕ) (saturday_fruiting : ℕ) (sunday_flowering : ℕ) (sunday_fruiting : ℕ) : ℕ :=
  let initial_fruiting := 2 * initial_flowering
  let saturday_total := initial_flowering + initial_fruiting + saturday_flowering + saturday_fruiting
  let sunday_total := saturday_total - sunday_flowering - sunday_fruiting
  sunday_total

theorem roxy_garden_plants : 
  calculate_remaining_plants 7 3 2 1 4 = 21 := by
  sorry

end roxy_garden_plants_l2483_248309


namespace m_range_l2483_248377

theorem m_range (m : ℝ) 
  (h1 : |m + 1| ≤ 2)
  (h2 : ¬(¬p))
  (h3 : ¬(p ∧ q))
  (p : Prop)
  (q : Prop) :
  -2 < m ∧ m ≤ 1 :=
by sorry

end m_range_l2483_248377


namespace range_of_m_m_value_for_diameter_l2483_248386

-- Define the circle equation
def circle_eq (x y m : ℝ) : Prop :=
  x^2 + y^2 + x - 6*y + m = 0

-- Define the line equation
def line_eq (x y : ℝ) : Prop :=
  x + 2*y - 3 = 0

-- Theorem for the range of m
theorem range_of_m (m : ℝ) :
  (∃ x y, circle_eq x y m) → m < 37/4 :=
sorry

-- Define the condition for PQ being diameter of circle passing through origin
def pq_diameter_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁*x₂ + y₁*y₂ = 0

-- Theorem for the value of m when PQ is diameter of circle passing through origin
theorem m_value_for_diameter (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂, 
    circle_eq x₁ y₁ m ∧ circle_eq x₂ y₂ m ∧
    line_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧
    pq_diameter_through_origin x₁ y₁ x₂ y₂) →
  m = 3 :=
sorry

end range_of_m_m_value_for_diameter_l2483_248386


namespace new_building_windows_l2483_248303

/-- The number of windows needed for a new building -/
def total_windows (installed : ℕ) (install_time : ℕ) (remaining_time : ℕ) : ℕ :=
  installed + remaining_time / install_time

/-- Theorem: The new building needs 14 windows in total -/
theorem new_building_windows :
  total_windows 8 8 48 = 14 := by
  sorry

end new_building_windows_l2483_248303


namespace stock_price_change_l2483_248333

theorem stock_price_change (initial_price : ℝ) (h : initial_price > 0) :
  let price_after_day1 := initial_price * (1 - 0.15)
  let price_after_day2 := price_after_day1 * (1 + 0.25)
  let percent_change := (price_after_day2 - initial_price) / initial_price * 100
  percent_change = 6.25 := by
  sorry

end stock_price_change_l2483_248333


namespace three_intersections_l2483_248393

/-- The number of intersection points between a circle and a parabola -/
def intersection_count (b : ℝ) : ℕ :=
  -- Define the count based on the intersection points
  -- This is a placeholder; the actual implementation would involve solving the system of equations
  sorry

/-- Theorem stating the condition for exactly three intersection points -/
theorem three_intersections (b : ℝ) :
  intersection_count b = 3 ↔ b > (1/4 : ℝ) := by
  sorry

end three_intersections_l2483_248393


namespace swim_club_scenario_l2483_248301

/-- Represents a swim club with members, some of whom have passed a lifesaving test
    and some of whom have taken a preparatory course. -/
structure SwimClub where
  total_members : ℕ
  passed_test : ℕ
  not_taken_course : ℕ

/-- The number of members who have taken the preparatory course but not passed the test -/
def members_taken_course_not_passed (club : SwimClub) : ℕ :=
  club.total_members - club.passed_test - club.not_taken_course

/-- Theorem stating the number of members who have taken the preparatory course
    but not passed the test in the given scenario -/
theorem swim_club_scenario :
  let club : SwimClub := {
    total_members := 60,
    passed_test := 18,  -- 30% of 60
    not_taken_course := 30
  }
  members_taken_course_not_passed club = 12 := by
  sorry

end swim_club_scenario_l2483_248301


namespace henry_final_distance_l2483_248380

-- Define the conversion factor from meters to feet
def metersToFeet : ℝ := 3.28084

-- Define Henry's movements
def northDistance : ℝ := 10 -- in meters
def eastDistance : ℝ := 30 -- in feet
def southDistance : ℝ := 10 * metersToFeet + 40 -- in feet

-- Calculate net southward movement
def netSouthDistance : ℝ := southDistance - (northDistance * metersToFeet)

-- Theorem to prove
theorem henry_final_distance :
  Real.sqrt (eastDistance ^ 2 + netSouthDistance ^ 2) = 50 := by
  sorry

end henry_final_distance_l2483_248380


namespace each_score_is_individual_l2483_248379

/-- Represents a student in the study -/
structure Student where
  id : Nat
  score : ℝ

/-- Represents the statistical study -/
structure CivilizationKnowledgeStudy where
  population : Finset Student
  sample : Finset Student
  pop_size : Nat
  sample_size : Nat

/-- Properties of the study -/
def valid_study (study : CivilizationKnowledgeStudy) : Prop :=
  study.pop_size = 1200 ∧
  study.sample_size = 100 ∧
  study.sample ⊆ study.population ∧
  study.population.card = study.pop_size ∧
  study.sample.card = study.sample_size

/-- Theorem stating that each student's score is an individual observation -/
theorem each_score_is_individual (study : CivilizationKnowledgeStudy) 
  (h : valid_study study) : 
  ∀ s ∈ study.population, ∃! x : ℝ, x = s.score :=
sorry

end each_score_is_individual_l2483_248379


namespace prove_scotts_golf_score_drop_l2483_248334

def scotts_golf_problem (first_four_average : ℝ) (fifth_round_score : ℝ) : Prop :=
  let first_four_total := first_four_average * 4
  let five_round_total := first_four_total + fifth_round_score
  let new_average := five_round_total / 5
  first_four_average - new_average = 2

theorem prove_scotts_golf_score_drop :
  scotts_golf_problem 78 68 :=
sorry

end prove_scotts_golf_score_drop_l2483_248334


namespace integer_solutions_of_equation_l2483_248342

theorem integer_solutions_of_equation :
  {(x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y} =
  {(0, -1), (-1, -1), (0, 0), (-1, 0), (5, 2)} := by
  sorry

end integer_solutions_of_equation_l2483_248342


namespace discarded_fruit_percentages_l2483_248362

/-- Represents the percentages of fruit sold and discarded over two days -/
structure FruitPercentages where
  pear_sold_day1 : ℝ
  pear_discarded_day1 : ℝ
  pear_sold_day2 : ℝ
  pear_discarded_day2 : ℝ
  apple_sold_day1 : ℝ
  apple_discarded_day1 : ℝ
  apple_sold_day2 : ℝ
  apple_discarded_day2 : ℝ
  orange_sold_day1 : ℝ
  orange_discarded_day1 : ℝ
  orange_sold_day2 : ℝ
  orange_discarded_day2 : ℝ

/-- Calculates the total percentage of fruit discarded over two days -/
def totalDiscardedPercentage (fp : FruitPercentages) : ℝ × ℝ × ℝ := sorry

/-- Theorem stating the correct percentages of discarded fruit -/
theorem discarded_fruit_percentages (fp : FruitPercentages) 
  (h1 : fp.pear_sold_day1 = 20)
  (h2 : fp.pear_discarded_day1 = 30)
  (h3 : fp.pear_sold_day2 = 10)
  (h4 : fp.pear_discarded_day2 = 20)
  (h5 : fp.apple_sold_day1 = 25)
  (h6 : fp.apple_discarded_day1 = 15)
  (h7 : fp.apple_sold_day2 = 15)
  (h8 : fp.apple_discarded_day2 = 10)
  (h9 : fp.orange_sold_day1 = 30)
  (h10 : fp.orange_discarded_day1 = 35)
  (h11 : fp.orange_sold_day2 = 20)
  (h12 : fp.orange_discarded_day2 = 30) :
  totalDiscardedPercentage fp = (34.08, 16.66875, 35.42) := by
  sorry

end discarded_fruit_percentages_l2483_248362


namespace rachels_homework_l2483_248354

theorem rachels_homework (math_pages reading_pages total_pages : ℕ) : 
  reading_pages = math_pages + 3 →
  total_pages = math_pages + reading_pages →
  total_pages = 23 →
  math_pages = 10 := by
sorry

end rachels_homework_l2483_248354


namespace max_revenue_price_l2483_248353

/-- The revenue function for the toy shop -/
def revenue (p : ℝ) : ℝ := p * (100 - 4 * p)

/-- The theorem stating the price that maximizes revenue -/
theorem max_revenue_price :
  ∃ (p : ℝ), p ≤ 20 ∧ ∀ (q : ℝ), q ≤ 20 → revenue p ≥ revenue q ∧ p = 12.5 := by
  sorry

end max_revenue_price_l2483_248353


namespace A_power_100_eq_A_l2483_248358

/-- The matrix A -/
def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 0, 1],
    ![1, 0, 0],
    ![0, 1, 0]]

/-- Theorem stating that A^100 = A -/
theorem A_power_100_eq_A : A ^ 100 = A := by sorry

end A_power_100_eq_A_l2483_248358


namespace black_marble_probability_l2483_248355

theorem black_marble_probability :
  let yellow : ℕ := 24
  let blue : ℕ := 18
  let green : ℕ := 12
  let red : ℕ := 8
  let white : ℕ := 7
  let black : ℕ := 3
  let purple : ℕ := 2
  let total : ℕ := yellow + blue + green + red + white + black + purple
  (black : ℚ) / total = 3 / 74 := by sorry

end black_marble_probability_l2483_248355


namespace line_equation_through_points_l2483_248310

/-- The equation of a line passing through two points (5, 0) and (2, -5) -/
theorem line_equation_through_points :
  ∃ (A B C : ℝ),
    (A * 5 + B * 0 + C = 0) ∧
    (A * 2 + B * (-5) + C = 0) ∧
    (A = 5 ∧ B = -3 ∧ C = -25) :=
by sorry

end line_equation_through_points_l2483_248310


namespace snow_probability_l2483_248341

theorem snow_probability (p : ℝ) (h : p = 3/4) :
  1 - (1 - p)^5 = 1023/1024 := by sorry

end snow_probability_l2483_248341


namespace third_term_is_35_l2483_248347

/-- An arithmetic sequence with 6 terms -/
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (n : ℕ)
  (h_arithmetic : ∀ i j, i < n → j < n → a (i + 1) - a i = a (j + 1) - a j)
  (h_length : n = 6)
  (h_first : a 0 = 23)
  (h_last : a 5 = 47)

/-- The third term of the arithmetic sequence is 35 -/
theorem third_term_is_35 (seq : ArithmeticSequence) : seq.a 2 = 35 := by
  sorry

end third_term_is_35_l2483_248347


namespace arithmetic_calculations_l2483_248348

theorem arithmetic_calculations :
  ((-7 + 13 - 6 + 20 = 20) ∧
   (-2^3 + (2 - 3) - 2 * (-1)^2023 = -7)) := by sorry

end arithmetic_calculations_l2483_248348


namespace root_sum_theorem_l2483_248327

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 8*x^2 + 10*x - 3

-- Define the roots
axiom p : ℝ
axiom q : ℝ
axiom r : ℝ

-- Axioms stating that p, q, and r are roots of f
axiom p_root : f p = 0
axiom q_root : f q = 0
axiom r_root : f r = 0

-- The theorem to prove
theorem root_sum_theorem :
  p / (q * r + 2) + q / (p * r + 2) + r / (p * q + 2) = 38 := by
  sorry

end root_sum_theorem_l2483_248327


namespace simplify_trigonometric_expression_l2483_248396

theorem simplify_trigonometric_expression (α : Real) (h : π < α ∧ α < 2*π) : 
  ((1 + Real.sin α + Real.cos α) * (Real.sin (α/2) - Real.cos (α/2))) / 
  Real.sqrt (2 + 2 * Real.cos α) = Real.cos α := by
  sorry

end simplify_trigonometric_expression_l2483_248396


namespace flooring_rate_calculation_l2483_248369

/-- Given a rectangular room with specified dimensions and total flooring cost,
    calculate the rate per square meter. -/
theorem flooring_rate_calculation (length width total_cost : ℝ) 
    (h_length : length = 10)
    (h_width : width = 4.75)
    (h_total_cost : total_cost = 42750) : 
    total_cost / (length * width) = 900 := by
  sorry

#check flooring_rate_calculation

end flooring_rate_calculation_l2483_248369


namespace space_geometry_theorem_l2483_248325

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (parallelLP : Line → Plane → Prop)
variable (parallelPP : Plane → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)

-- Define the theorem
theorem space_geometry_theorem 
  (m n : Line) (α β : Plane) 
  (hm_neq_n : m ≠ n) (hα_neq_β : α ≠ β) :
  (perpendicularLP m α ∧ perpendicularLP n β ∧ perpendicular m n → perpendicularPP α β) ∧
  (perpendicularLP m α ∧ parallelLP n β ∧ parallelPP α β → perpendicular m n) :=
sorry

end space_geometry_theorem_l2483_248325


namespace shoe_pairs_count_l2483_248397

theorem shoe_pairs_count (total_shoes : ℕ) (prob_same_color : ℚ) : 
  total_shoes = 14 →
  prob_same_color = 1 / 13 →
  (∃ (n : ℕ), n * 2 = total_shoes ∧ 
    prob_same_color = 1 / (2 * n - 1)) →
  ∃ (pairs : ℕ), pairs = 7 ∧ pairs * 2 = total_shoes :=
by sorry

end shoe_pairs_count_l2483_248397


namespace diamond_two_three_l2483_248349

def diamond (a b : ℝ) : ℝ := a * b^2 - b + 1

theorem diamond_two_three : diamond 2 3 = 16 := by
  sorry

end diamond_two_three_l2483_248349


namespace central_high_teachers_central_high_teachers_count_l2483_248399

/-- Calculates the number of teachers required at Central High School -/
theorem central_high_teachers (total_students : ℕ) (classes_per_student : ℕ) 
  (classes_per_teacher : ℕ) (students_per_class : ℕ) : ℕ :=
  let total_class_occurrences := total_students * classes_per_student
  let unique_classes := total_class_occurrences / students_per_class
  let required_teachers := unique_classes / classes_per_teacher
  required_teachers

/-- Proves that the number of teachers required at Central High School is 120 -/
theorem central_high_teachers_count : 
  central_high_teachers 1500 6 3 25 = 120 := by
  sorry

end central_high_teachers_central_high_teachers_count_l2483_248399


namespace imaginary_part_of_complex_fraction_l2483_248343

theorem imaginary_part_of_complex_fraction : Complex.im (2 * Complex.I / (1 + Complex.I)) = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l2483_248343


namespace point_above_line_l2483_248374

/-- A point (x, y) is above a line ax + by + c = 0 if by < -ax - c -/
def IsAboveLine (x y a b c : ℝ) : Prop := b * y < -a * x - c

/-- The range of t for which (-2, t) is above the line 2x - 3y + 6 = 0 -/
theorem point_above_line (t : ℝ) : 
  IsAboveLine (-2) t 2 (-3) 6 → t > 2/3 := by
  sorry

end point_above_line_l2483_248374


namespace min_value_expression_l2483_248335

theorem min_value_expression (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (sum_eq_one : a + b + c = 1) :
  a + (a * b) ^ (1/3 : ℝ) + (a * b * c) ^ (1/4 : ℝ) ≥ 1/3 + 1/(3 * 3^(1/3 : ℝ)) + 1/(3 * 3^(1/4 : ℝ)) :=
sorry

end min_value_expression_l2483_248335


namespace books_combination_l2483_248387

/- Given conditions -/
def totalBooks : ℕ := 13
def booksToSelect : ℕ := 3

/- Theorem to prove -/
theorem books_combination : Nat.choose totalBooks booksToSelect = 286 := by
  sorry

end books_combination_l2483_248387


namespace at_least_one_equal_to_a_l2483_248364

theorem at_least_one_equal_to_a (x y z a : ℝ) 
  (sum_eq : x + y + z = a) 
  (inv_sum_eq : 1/x + 1/y + 1/z = 1/a) : 
  x = a ∨ y = a ∨ z = a := by
sorry

end at_least_one_equal_to_a_l2483_248364


namespace initial_water_percentage_l2483_248308

theorem initial_water_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_percentage : ℝ)
  (h1 : initial_volume = 20)
  (h2 : added_water = 2)
  (h3 : final_percentage = 20)
  : ∃ initial_percentage : ℝ,
    initial_percentage * initial_volume / 100 + added_water =
    final_percentage * (initial_volume + added_water) / 100 ∧
    initial_percentage = 12 := by
  sorry

end initial_water_percentage_l2483_248308


namespace right_triangle_max_area_l2483_248338

theorem right_triangle_max_area (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 4) :
  (1/2) * a * b ≤ 2 := by
sorry

end right_triangle_max_area_l2483_248338


namespace mrs_hilt_reading_l2483_248385

/-- The number of books Mrs. Hilt read -/
def num_books : ℕ := 4

/-- The number of chapters in each book -/
def chapters_per_book : ℕ := 17

/-- The total number of chapters Mrs. Hilt read -/
def total_chapters : ℕ := num_books * chapters_per_book

theorem mrs_hilt_reading : total_chapters = 68 := by
  sorry

end mrs_hilt_reading_l2483_248385


namespace combine_like_terms_l2483_248317

theorem combine_like_terms (a : ℝ) : 3 * a^2 + 5 * a^2 - a^2 = 7 * a^2 := by
  sorry

end combine_like_terms_l2483_248317


namespace complex_number_location_l2483_248350

theorem complex_number_location (z : ℂ) :
  (z * (1 + Complex.I) = 3 - Complex.I) →
  (0 < z.re ∧ z.im < 0) :=
by sorry

end complex_number_location_l2483_248350


namespace sqrt_x_plus_inverse_sqrt_x_l2483_248331

theorem sqrt_x_plus_inverse_sqrt_x (x : ℝ) (h_pos : x > 0) (h_eq : x + 1/x = 100) :
  Real.sqrt x + 1 / Real.sqrt x = Real.sqrt 102 := by
  sorry

end sqrt_x_plus_inverse_sqrt_x_l2483_248331


namespace c_nonzero_necessary_not_sufficient_l2483_248315

/-- Represents a conic section defined by the equation ax^2 + y^2 = c -/
structure ConicSection where
  a : ℝ
  c : ℝ

/-- Determines if a conic section is an ellipse or hyperbola -/
def isEllipseOrHyperbola (conic : ConicSection) : Prop :=
  (conic.a > 0 ∧ conic.c > 0) ∨ (conic.a < 0 ∧ conic.c ≠ 0)

/-- The main theorem stating that c ≠ 0 is necessary but not sufficient -/
theorem c_nonzero_necessary_not_sufficient :
  (∀ conic : ConicSection, isEllipseOrHyperbola conic → conic.c ≠ 0) ∧
  (∃ conic : ConicSection, conic.c ≠ 0 ∧ ¬isEllipseOrHyperbola conic) :=
sorry

end c_nonzero_necessary_not_sufficient_l2483_248315


namespace mike_pears_count_l2483_248324

/-- The number of pears picked by Jason -/
def jason_pears : ℕ := 46

/-- The number of pears picked by Keith -/
def keith_pears : ℕ := 47

/-- The total number of pears picked -/
def total_pears : ℕ := 105

/-- The number of pears picked by Mike -/
def mike_pears : ℕ := total_pears - (jason_pears + keith_pears)

theorem mike_pears_count : mike_pears = 12 := by
  sorry

end mike_pears_count_l2483_248324


namespace intersection_complement_equals_half_open_interval_l2483_248375

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}

-- Define set N
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- State the theorem
theorem intersection_complement_equals_half_open_interval :
  M ∩ (Set.compl N) = Set.Icc (-2) 0 := by sorry

end intersection_complement_equals_half_open_interval_l2483_248375


namespace sufficient_condition_range_l2483_248395

theorem sufficient_condition_range (a : ℝ) : 
  (∀ x : ℝ, (x - 1)^2 < 9 → (x + 2) * (x + a) < 0) ∧
  (∃ x : ℝ, (x + 2) * (x + a) < 0 ∧ (x - 1)^2 ≥ 9) →
  a < -4 :=
by sorry

end sufficient_condition_range_l2483_248395


namespace gcd_360_504_l2483_248312

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_l2483_248312


namespace solve_equation_l2483_248371

theorem solve_equation (x : ℝ) (h : 3 * x - 2 = 13) : 6 * x + 4 = 34 := by
  sorry

end solve_equation_l2483_248371


namespace quadratic_intersection_theorem_l2483_248300

/-- Quadratic function f(x) = x^2 + 3x + n -/
def f (n : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + n

/-- Predicate for exactly one positive real root -/
def has_exactly_one_positive_root (n : ℝ) : Prop :=
  ∃! x : ℝ, x > 0 ∧ f n x = 0

theorem quadratic_intersection_theorem :
  has_exactly_one_positive_root (-2) ∧
  ∀ n : ℝ, has_exactly_one_positive_root n → n = -2 :=
sorry

end quadratic_intersection_theorem_l2483_248300


namespace largest_angle_in_triangle_l2483_248359

theorem largest_angle_in_triangle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α + β = 105 →      -- Sum of two angles is 7/6 of a right angle (90° * 7/6 = 105°)
  β = α + 20 →       -- One angle is 20° larger than the other
  max α (max β γ) = 75 := by
sorry

end largest_angle_in_triangle_l2483_248359


namespace distinct_digit_sum_l2483_248345

theorem distinct_digit_sum (A B C D : Nat) : 
  A < 10 → B < 10 → C < 10 → D < 10 →
  A ≠ B → A ≠ C → A ≠ D → B ≠ C → B ≠ D → C ≠ D →
  A + B + 1 = D →
  C + D = D + 1 →
  (∃ (count : Nat), count = 6 ∧ 
    (∀ (x : Nat), x < 10 → 
      (∃ (a b c : Nat), a < 10 ∧ b < 10 ∧ c < 10 ∧
        a ≠ b ∧ a ≠ c ∧ a ≠ x ∧ b ≠ c ∧ b ≠ x ∧ c ≠ x ∧
        a + b + 1 = x ∧ c + x = x + 1) ↔ 
      x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9)) :=
by sorry

end distinct_digit_sum_l2483_248345


namespace parabola_equation_correct_l2483_248330

/-- A parabola with x-axis as its axis of symmetry, vertex at the origin, and latus rectum length of 8 -/
structure Parabola where
  symmetry_axis : ℝ → ℝ
  vertex : ℝ × ℝ
  latus_rectum : ℝ
  h_symmetry : symmetry_axis = λ y => 0
  h_vertex : vertex = (0, 0)
  h_latus_rectum : latus_rectum = 8

/-- The equation of the parabola -/
def parabola_equation (p : Parabola) : Set (ℝ × ℝ) :=
  {(x, y) | y^2 = 8*x ∨ y^2 = -8*x}

theorem parabola_equation_correct (p : Parabola) :
  ∀ (x y : ℝ), (x, y) ∈ parabola_equation p ↔
    (∃ t : ℝ, x = t^2 / 2 ∧ y = t) ∨ (∃ t : ℝ, x = -t^2 / 2 ∧ y = t) :=
by sorry

end parabola_equation_correct_l2483_248330


namespace georgie_prank_ways_l2483_248340

/-- The number of windows in the mansion -/
def num_windows : ℕ := 8

/-- The number of ways Georgie can accomplish the prank -/
def prank_ways : ℕ := num_windows * (num_windows - 1) * (num_windows - 2)

/-- Theorem stating that the number of ways Georgie can accomplish the prank is 336 -/
theorem georgie_prank_ways : prank_ways = 336 := by
  sorry

end georgie_prank_ways_l2483_248340


namespace dianas_age_dianas_age_is_eight_l2483_248346

/-- Diana's age today, given that she is twice as old as Grace and Grace turned 3 a year ago -/
theorem dianas_age : ℕ :=
  let graces_age_last_year : ℕ := 3
  let graces_age_today : ℕ := graces_age_last_year + 1
  let dianas_age : ℕ := 2 * graces_age_today
  8

/-- Proof that Diana's age is 8 years old today -/
theorem dianas_age_is_eight : dianas_age = 8 := by
  sorry

end dianas_age_dianas_age_is_eight_l2483_248346


namespace litter_size_l2483_248323

/-- Represents the number of puppies in the litter -/
def puppies : ℕ := sorry

/-- The profit John makes from selling the puppies -/
def profit : ℕ := 1500

/-- The amount John pays to the stud owner -/
def stud_fee : ℕ := 300

/-- The price for which John sells each puppy -/
def price_per_puppy : ℕ := 600

theorem litter_size : 
  puppies = 8 ∧ 
  (puppies / 2 - 1) * price_per_puppy - stud_fee = profit :=
sorry

end litter_size_l2483_248323


namespace soccer_players_count_l2483_248365

theorem soccer_players_count (total_socks : ℕ) (socks_per_player : ℕ) : 
  total_socks = 16 → socks_per_player = 2 → total_socks / socks_per_player = 8 := by
  sorry

end soccer_players_count_l2483_248365


namespace min_distance_to_line_l2483_248351

/-- The minimum squared distance from the origin to the line 4x + 3y - 10 = 0 is 4 -/
theorem min_distance_to_line : 
  (∀ m n : ℝ, 4*m + 3*n - 10 = 0 → m^2 + n^2 ≥ 4) ∧ 
  (∃ m n : ℝ, 4*m + 3*n - 10 = 0 ∧ m^2 + n^2 = 4) := by
  sorry

end min_distance_to_line_l2483_248351


namespace matrix_power_four_l2483_248357

def A : Matrix (Fin 2) (Fin 2) ℝ := !![1, -1; 1, 1]

theorem matrix_power_four : 
  A ^ 4 = !![(-4 : ℝ), 0; 0, -4] := by sorry

end matrix_power_four_l2483_248357


namespace jane_calculation_l2483_248302

theorem jane_calculation (x y z : ℝ) 
  (h1 : x - y - z = 7)
  (h2 : x - (y + z) = 19) : 
  x - y = 13 := by sorry

end jane_calculation_l2483_248302
