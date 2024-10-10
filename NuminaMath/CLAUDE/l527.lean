import Mathlib

namespace right_triangle_area_l527_52799

theorem right_triangle_area (a b : ℝ) (h1 : a^2 = 64) (h2 : b^2 = 81) :
  (1/2) * a * b = 36 := by
  sorry

end right_triangle_area_l527_52799


namespace min_sum_of_arithmetic_sequence_l527_52734

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  d : ℤ      -- Common difference

/-- Sum of the first n terms of an arithmetic sequence -/
def sumOfTerms (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * seq.a 1 + n * (n - 1) / 2 * seq.d

theorem min_sum_of_arithmetic_sequence (seq : ArithmeticSequence) :
  seq.a 1 = -7 →
  sumOfTerms seq 3 = -15 →
  ∀ n : ℕ, sumOfTerms seq n ≥ -16 ∧ 
  (∃ m : ℕ, sumOfTerms seq m = -16) := by
sorry

end min_sum_of_arithmetic_sequence_l527_52734


namespace circle_center_and_radius_l527_52748

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- State the theorem
theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    center = (-1, 0) ∧
    radius = Real.sqrt 2 := by
  sorry

end circle_center_and_radius_l527_52748


namespace sock_pair_count_l527_52796

def white_socks : ℕ := 5
def brown_socks : ℕ := 3
def blue_socks : ℕ := 2
def black_socks : ℕ := 2

def total_socks : ℕ := white_socks + brown_socks + blue_socks + black_socks

def choose (n k : ℕ) : ℕ := Nat.choose n k

def same_color_pairs : ℕ :=
  choose white_socks 2 + choose brown_socks 2 + choose blue_socks 2 + choose black_socks 2

theorem sock_pair_count : same_color_pairs = 15 := by
  sorry

end sock_pair_count_l527_52796


namespace mikaela_savings_l527_52740

/-- Calculates the total savings for Mikaela over two months of tutoring --/
def total_savings (
  hourly_rate_month1 : ℚ)
  (hours_month1 : ℚ)
  (hourly_rate_month2 : ℚ)
  (additional_hours_month2 : ℚ)
  (spending_ratio_month1 : ℚ)
  (spending_ratio_month2 : ℚ) : ℚ :=
  let earnings_month1 := hourly_rate_month1 * hours_month1
  let savings_month1 := earnings_month1 * (1 - spending_ratio_month1)
  let hours_month2 := hours_month1 + additional_hours_month2
  let earnings_month2 := hourly_rate_month2 * hours_month2
  let savings_month2 := earnings_month2 * (1 - spending_ratio_month2)
  savings_month1 + savings_month2

/-- Proves that Mikaela's total savings from both months is $190 --/
theorem mikaela_savings :
  total_savings 10 35 12 5 (4/5) (3/4) = 190 := by
  sorry

end mikaela_savings_l527_52740


namespace total_sightings_first_quarter_l527_52708

/-- The total number of animal sightings in the first three months of the year. -/
def total_sightings (january_sightings : ℕ) : ℕ :=
  let february_sightings := 3 * january_sightings
  let march_sightings := february_sightings / 2
  january_sightings + february_sightings + march_sightings

/-- Theorem stating that the total number of animal sightings in the first three months is 143,
    given that there were 26 sightings in January. -/
theorem total_sightings_first_quarter (h : total_sightings 26 = 143) : total_sightings 26 = 143 := by
  sorry

end total_sightings_first_quarter_l527_52708


namespace ten_women_circular_reseating_l527_52798

/-- The number of ways n women can be reseated in a circular arrangement,
    where each woman sits in her original seat or a seat adjacent to it. -/
def C : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | n + 2 => C (n + 1) + C n

theorem ten_women_circular_reseating : C 10 = 89 := by
  sorry

end ten_women_circular_reseating_l527_52798


namespace hacker_can_achieve_goal_l527_52773

/-- Represents a user in the social network -/
structure User where
  id : Nat
  followers : Finset Nat
  rating : Nat

/-- Represents the social network -/
structure SocialNetwork where
  users : Finset User
  m : Nat

/-- Represents a hacker's action: increasing a user's rating by 1 or doing nothing -/
inductive HackerAction
  | Increase (userId : Nat)
  | DoNothing

/-- Update ratings based on followers -/
def updateRatings (sn : SocialNetwork) : SocialNetwork :=
  sorry

/-- Apply hacker's action to the social network -/
def applyHackerAction (sn : SocialNetwork) (action : HackerAction) : SocialNetwork :=
  sorry

/-- Check if all ratings are divisible by m -/
def allRatingsDivisible (sn : SocialNetwork) : Prop :=
  sorry

/-- The main theorem -/
theorem hacker_can_achieve_goal (sn : SocialNetwork) :
  ∃ (actions : List HackerAction), allRatingsDivisible (actions.foldl applyHackerAction sn) :=
sorry

end hacker_can_achieve_goal_l527_52773


namespace percentage_of_population_l527_52785

theorem percentage_of_population (total_population : ℕ) (part_population : ℕ) :
  total_population = 28800 →
  part_population = 23040 →
  (part_population : ℚ) / (total_population : ℚ) * 100 = 80 := by
sorry

end percentage_of_population_l527_52785


namespace sum_of_specific_geometric_series_l527_52739

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_specific_geometric_series :
  geometric_series_sum (1/4) (1/2) 7 = 127/256 := by
  sorry

end sum_of_specific_geometric_series_l527_52739


namespace shaded_area_possibilities_l527_52744

/-- Represents a rectangle on a grid --/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the configuration of rectangles in the problem --/
structure Configuration where
  abcd : Rectangle
  pqrs : Rectangle
  qrst : Rectangle
  upper_right : Rectangle

/-- The main theorem statement --/
theorem shaded_area_possibilities (config : Configuration) : 
  (config.abcd.width * config.abcd.height = 33) →
  (config.abcd.width < 7 ∧ config.abcd.height < 7) →
  (config.abcd.width ≥ 1 ∧ config.abcd.height ≥ 1) →
  (config.pqrs.width < 7 ∧ config.pqrs.height < 7) →
  (config.qrst.width < 7 ∧ config.qrst.height < 7) →
  (config.upper_right.width < 7 ∧ config.upper_right.height < 7) →
  (config.qrst.width = config.qrst.height) →
  (config.pqrs.width < config.upper_right.height) →
  (config.pqrs.width ≠ config.pqrs.height) →
  (config.upper_right.width ≠ config.upper_right.height) →
  (∃ (shaded_area : ℕ), 
    shaded_area = config.abcd.width * config.abcd.height - 
      (config.pqrs.width * config.pqrs.height + 
       config.qrst.width * config.qrst.height + 
       config.upper_right.width * config.upper_right.height) ∧
    (shaded_area = 21 ∨ shaded_area = 20 ∨ shaded_area = 17)) :=
by
  sorry


end shaded_area_possibilities_l527_52744


namespace quadratic_roots_condition_l527_52766

theorem quadratic_roots_condition (a : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ x^2 - 2*x + a + 1 = 0 ∧ y^2 - 2*y + a + 1 = 0) ↔ a < -1 :=
sorry

end quadratic_roots_condition_l527_52766


namespace set_problem_l527_52736

def I (x : ℝ) : Set ℝ := {2, 3, x^2 + 2*x - 3}
def A : Set ℝ := {5}

theorem set_problem (x y : ℝ) (C : Set ℝ) : 
  C ⊆ I x → C \ A = {2, y} → 
  ((x = -4 ∧ y = 3) ∨ (x = 2 ∧ y = 3)) :=
by sorry

end set_problem_l527_52736


namespace polynomial_equality_l527_52713

theorem polynomial_equality (a : ℝ) (h : 2 * a^2 - 3 * a - 5 = 0) :
  4 * a^4 - 12 * a^3 + 9 * a^2 - 10 = 15 := by
  sorry

end polynomial_equality_l527_52713


namespace complex_fraction_equality_l527_52731

theorem complex_fraction_equality : (1 + 3 * Complex.I) / (1 - Complex.I) = -1 + 2 * Complex.I := by
  sorry

end complex_fraction_equality_l527_52731


namespace initial_workers_count_l527_52702

theorem initial_workers_count (W : ℕ) : 
  (2 : ℚ) / 3 * W = W - (W / 3) →  -- Initially, 2/3 of workers are men
  (W / 3 + 10 : ℚ) / (W + 10) = 2 / 5 →  -- After hiring 10 women, 40% of workforce is female
  W = 90 := by
sorry

end initial_workers_count_l527_52702


namespace books_from_second_shop_l527_52724

theorem books_from_second_shop 
  (first_shop_books : ℕ)
  (first_shop_cost : ℕ)
  (second_shop_cost : ℕ)
  (average_price : ℕ)
  (h1 : first_shop_books = 55)
  (h2 : first_shop_cost = 1500)
  (h3 : second_shop_cost = 340)
  (h4 : average_price = 16) :
  ∃ (second_shop_books : ℕ),
    (first_shop_cost + second_shop_cost) = 
    average_price * (first_shop_books + second_shop_books) ∧
    second_shop_books = 60 := by
  sorry

end books_from_second_shop_l527_52724


namespace coins_in_first_stack_l527_52705

theorem coins_in_first_stack (total : ℕ) (stack2 : ℕ) (h1 : total = 12) (h2 : stack2 = 8) :
  total - stack2 = 4 := by
  sorry

end coins_in_first_stack_l527_52705


namespace square_plus_self_divisible_by_two_l527_52721

theorem square_plus_self_divisible_by_two (a : ℤ) : 
  ∃ k : ℤ, a^2 + a = 2 * k :=
by
  sorry

end square_plus_self_divisible_by_two_l527_52721


namespace decimal_34_to_binary_binary_to_decimal_34_l527_52749

/-- Converts a natural number to its binary representation as a list of bits -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec go (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: go (m / 2)
  go n

/-- Converts a list of bits to its decimal representation -/
def fromBinary (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

theorem decimal_34_to_binary :
  toBinary 34 = [false, true, false, false, false, true] :=
by sorry

theorem binary_to_decimal_34 :
  fromBinary [false, true, false, false, false, true] = 34 :=
by sorry

end decimal_34_to_binary_binary_to_decimal_34_l527_52749


namespace white_surface_fraction_is_half_l527_52711

/-- Represents a cube composed of smaller cubes -/
structure CompositeCube where
  smallCubeCount : ℕ
  smallCubeSideLength : ℝ
  whiteCubeCount : ℕ
  blackCubeCount : ℕ
  redCubeCount : ℕ

/-- The fraction of the surface area that is white -/
def whiteSurfaceFraction (c : CompositeCube) : ℚ :=
  sorry

/-- Our specific cube configuration -/
def ourCube : CompositeCube :=
  { smallCubeCount := 64
  , smallCubeSideLength := 1
  , whiteCubeCount := 36
  , blackCubeCount := 8
  , redCubeCount := 20 }

theorem white_surface_fraction_is_half :
  whiteSurfaceFraction ourCube = 1/2 := by
  sorry

end white_surface_fraction_is_half_l527_52711


namespace last_two_digits_product_l527_52722

theorem last_two_digits_product (n : ℤ) : 
  (n % 100 % 4 = 0) → 
  ((n % 100) / 10 + n % 10 = 13) → 
  ((n % 100) / 10 * (n % 10) = 42) := by
sorry

end last_two_digits_product_l527_52722


namespace ivanov_net_worth_calculation_l527_52778

/-- The net worth of the Ivanov family -/
def ivanov_net_worth : ℕ := by sorry

/-- The value of the Ivanov family's apartment in rubles -/
def apartment_value : ℕ := 3000000

/-- The value of the Ivanov family's car in rubles -/
def car_value : ℕ := 900000

/-- The amount in the Ivanov family's bank deposit in rubles -/
def bank_deposit : ℕ := 300000

/-- The value of the Ivanov family's securities in rubles -/
def securities_value : ℕ := 200000

/-- The amount of liquid cash the Ivanov family has in rubles -/
def liquid_cash : ℕ := 100000

/-- The Ivanov family's mortgage balance in rubles -/
def mortgage_balance : ℕ := 1500000

/-- The Ivanov family's car loan balance in rubles -/
def car_loan_balance : ℕ := 500000

/-- The Ivanov family's debt to relatives in rubles -/
def debt_to_relatives : ℕ := 200000

/-- Theorem stating that the Ivanov family's net worth is 2,300,000 rubles -/
theorem ivanov_net_worth_calculation :
  ivanov_net_worth = 
    (apartment_value + car_value + bank_deposit + securities_value + liquid_cash) -
    (mortgage_balance + car_loan_balance + debt_to_relatives) :=
by sorry

end ivanov_net_worth_calculation_l527_52778


namespace math_club_team_selection_l527_52789

/-- The number of ways to select a team from a math club --/
def select_team (total_boys : ℕ) (total_girls : ℕ) (team_size : ℕ) (team_boys : ℕ) (team_girls : ℕ) 
  (experienced_boys : ℕ) (experienced_girls : ℕ) : ℕ :=
  (Nat.choose (total_boys - experienced_boys) (team_boys - experienced_boys)) * 
  (Nat.choose (total_girls - experienced_girls) (team_girls - experienced_girls))

/-- Theorem: The number of ways to select the team is 540 --/
theorem math_club_team_selection :
  select_team 7 10 6 3 3 1 1 = 540 := by
  sorry

end math_club_team_selection_l527_52789


namespace inheritance_satisfies_tax_equation_l527_52714

/-- Represents the inheritance amount in dollars -/
def inheritance : ℝ := sorry

/-- The total tax paid is $15000 -/
def total_tax : ℝ := 15000

/-- Federal tax rate is 25% -/
def federal_tax_rate : ℝ := 0.25

/-- State tax rate is 15% -/
def state_tax_rate : ℝ := 0.15

/-- Theorem stating that the inheritance satisfies the tax equation -/
theorem inheritance_satisfies_tax_equation : 
  federal_tax_rate * inheritance + state_tax_rate * (1 - federal_tax_rate) * inheritance = total_tax := by
  sorry

end inheritance_satisfies_tax_equation_l527_52714


namespace marcos_boat_distance_l527_52758

/-- The distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proof that given a speed of 30 mph and a time of 10 minutes, the distance traveled is 5 miles -/
theorem marcos_boat_distance :
  let speed : ℝ := 30  -- Speed in miles per hour
  let time : ℝ := 10 / 60  -- Time in hours (10 minutes converted to hours)
  distance speed time = 5 := by
  sorry

end marcos_boat_distance_l527_52758


namespace arithmetic_sequence_a2_l527_52712

/-- An arithmetic sequence with specified conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a2 (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a1 : a 1 = 3) 
  (h_sum : a 2 + a 3 = 12) : 
  a 2 = 5 := by
sorry

end arithmetic_sequence_a2_l527_52712


namespace square_diff_value_l527_52732

theorem square_diff_value (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) : x^2 - y^2 = 80 := by
  sorry

end square_diff_value_l527_52732


namespace complex_magnitude_l527_52751

theorem complex_magnitude (i z : ℂ) : 
  i * i = -1 → i * z = 1 - i → Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_l527_52751


namespace wall_width_correct_l527_52763

/-- Represents the dimensions and properties of a wall -/
structure Wall where
  width : ℝ
  height : ℝ
  length : ℝ
  volume : ℝ

/-- The width of the wall given the conditions -/
def wall_width (w : Wall) : ℝ :=
  (384 : ℝ) ^ (1/3)

/-- Theorem stating that the calculated width satisfies the given conditions -/
theorem wall_width_correct (w : Wall) 
  (h_height : w.height = 6 * w.width)
  (h_length : w.length = 7 * w.height)
  (h_volume : w.volume = 16128) : 
  w.width = wall_width w := by
  sorry

#eval wall_width { width := 0, height := 0, length := 0, volume := 16128 }

end wall_width_correct_l527_52763


namespace snail_distance_is_29_l527_52790

def snail_path : List ℤ := [3, -5, 8, 0]

def distance (a b : ℤ) : ℤ := Int.natAbs (b - a)

def total_distance (path : List ℤ) : ℤ :=
  (List.zip path path.tail).foldl (fun acc (a, b) => acc + distance a b) 0

theorem snail_distance_is_29 : total_distance snail_path = 29 := by
  sorry

end snail_distance_is_29_l527_52790


namespace smallest_four_digit_congruence_l527_52738

theorem smallest_four_digit_congruence : ∃ (x : ℕ), 
  (x ≥ 1000 ∧ x < 10000) ∧
  (∀ y : ℕ, y ≥ 1000 ∧ y < 10000 →
    (5 * y ≡ 15 [ZMOD 20] ∧
     3 * y + 7 ≡ 10 [ZMOD 8] ∧
     -3 * y + 2 ≡ 2 * y [ZMOD 35]) →
    x ≤ y) ∧
  (5 * x ≡ 15 [ZMOD 20] ∧
   3 * x + 7 ≡ 10 [ZMOD 8] ∧
   -3 * x + 2 ≡ 2 * x [ZMOD 35]) ∧
  x = 1009 :=
by sorry

end smallest_four_digit_congruence_l527_52738


namespace binomial_26_6_l527_52793

theorem binomial_26_6 (h1 : Nat.choose 24 5 = 42504) 
                      (h2 : Nat.choose 24 6 = 134596) 
                      (h3 : Nat.choose 24 7 = 346104) : 
  Nat.choose 26 6 = 657800 := by
  sorry

end binomial_26_6_l527_52793


namespace total_weight_loss_l527_52707

def weight_loss_problem (seth_loss jerome_loss veronica_loss total_loss : ℝ) : Prop :=
  seth_loss = 17.5 ∧
  jerome_loss = 3 * seth_loss ∧
  veronica_loss = seth_loss + 1.5 ∧
  total_loss = seth_loss + jerome_loss + veronica_loss

theorem total_weight_loss :
  ∃ (seth_loss jerome_loss veronica_loss total_loss : ℝ),
    weight_loss_problem seth_loss jerome_loss veronica_loss total_loss ∧
    total_loss = 89 := by
  sorry

end total_weight_loss_l527_52707


namespace lucy_age_theorem_l527_52750

/-- Lucy's age at the end of 2000 -/
def lucy_age_2000 : ℝ := 27.5

/-- Lucy's grandfather's age at the end of 2000 -/
def grandfather_age_2000 : ℝ := 3 * lucy_age_2000

/-- The sum of Lucy's and her grandfather's birth years -/
def birth_years_sum : ℝ := 3890

/-- Lucy's age at the end of 2010 -/
def lucy_age_2010 : ℝ := lucy_age_2000 + 10

theorem lucy_age_theorem :
  lucy_age_2000 = (grandfather_age_2000 / 3) ∧
  (2000 - lucy_age_2000) + (2000 - grandfather_age_2000) = birth_years_sum ∧
  lucy_age_2010 = 37.5 := by
  sorry

end lucy_age_theorem_l527_52750


namespace sum_of_squares_of_roots_l527_52776

theorem sum_of_squares_of_roots (k l m n a b c : ℕ) :
  k ≠ l ∧ k ≠ m ∧ k ≠ n ∧ l ≠ m ∧ l ≠ n ∧ m ≠ n →
  ((a * k^2 - b * k + c = 0 ∨ c * k^2 - 16 * b * k + 256 * a = 0) ∧
   (a * l^2 - b * l + c = 0 ∨ c * l^2 - 16 * b * l + 256 * a = 0) ∧
   (a * m^2 - b * m + c = 0 ∨ c * m^2 - 16 * b * m + 256 * a = 0) ∧
   (a * n^2 - b * n + c = 0 ∨ c * n^2 - 16 * b * n + 256 * a = 0)) →
  k^2 + l^2 + m^2 + n^2 = 325 :=
by sorry

end sum_of_squares_of_roots_l527_52776


namespace polynomial_uniqueness_l527_52768

theorem polynomial_uniqueness (Q : ℝ → ℝ) :
  (∀ x, Q x = Q 0 + Q 1 * x + Q 2 * x^2) →
  Q (-1) = 3 →
  Q 3 = 15 →
  ∀ x, Q x = -2 * x^2 + 6 * x - 1 := by
sorry

end polynomial_uniqueness_l527_52768


namespace problem_statement_l527_52783

theorem problem_statement :
  let M := (Real.sqrt (3 + Real.sqrt 8) + Real.sqrt (3 - Real.sqrt 8)) / Real.sqrt (2 * Real.sqrt 2 + 1) - Real.sqrt (4 - 2 * Real.sqrt 3)
  M = 3 - Real.sqrt 3 := by sorry

end problem_statement_l527_52783


namespace gloria_has_23_maple_trees_l527_52726

/-- Represents the problem of calculating Gloria's maple trees --/
def GloriasMapleTrees (cabin_price cash_on_hand leftover cypress_count pine_count cypress_price pine_price maple_price : ℕ) : Prop :=
  let total_needed := cabin_price - cash_on_hand
  let cypress_income := cypress_count * cypress_price
  let pine_income := pine_count * pine_price
  let maple_income := total_needed - cypress_income - pine_income
  ∃ (maple_count : ℕ), 
    maple_count * maple_price = maple_income ∧
    maple_count * maple_price + cypress_income + pine_income + cash_on_hand = cabin_price + leftover

theorem gloria_has_23_maple_trees : 
  GloriasMapleTrees 129000 150 350 20 600 100 200 300 → 
  ∃ (maple_count : ℕ), maple_count = 23 :=
sorry

end gloria_has_23_maple_trees_l527_52726


namespace bird_feeder_theft_ratio_l527_52746

/-- Given a bird feeder with the following properties:
  - Holds 2 cups of birdseed
  - Each cup of birdseed can feed 14 birds
  - The feeder actually feeds 21 birds weekly
  Prove that the ratio of birdseed stolen to total birdseed is 1:4 -/
theorem bird_feeder_theft_ratio 
  (total_cups : ℚ) 
  (birds_per_cup : ℕ) 
  (birds_fed : ℕ) : 
  total_cups = 2 →
  birds_per_cup = 14 →
  birds_fed = 21 →
  (total_cups - (birds_fed : ℚ) / (birds_per_cup : ℚ)) / total_cups = 1 / 4 := by
  sorry

end bird_feeder_theft_ratio_l527_52746


namespace parabola_directrix_l527_52759

/-- The equation of the directrix of the parabola y^2 = 2x is x = -1/2 -/
theorem parabola_directrix : ∀ x y : ℝ, y^2 = 2*x → (∃ p : ℝ, p > 0 ∧ x = -p/2) := by
  sorry

end parabola_directrix_l527_52759


namespace blue_segments_count_l527_52703

/-- Represents the number of rows and columns in the square array -/
def n : ℕ := 10

/-- Represents the total number of red dots -/
def total_red_dots : ℕ := 52

/-- Represents the number of red dots at corners -/
def corner_red_dots : ℕ := 2

/-- Represents the number of red dots on edges (excluding corners) -/
def edge_red_dots : ℕ := 16

/-- Represents the number of green line segments -/
def green_segments : ℕ := 98

/-- Theorem stating that the number of blue line segments is 37 -/
theorem blue_segments_count :
  let total_segments := 2 * n * (n - 1)
  let interior_red_dots := total_red_dots - corner_red_dots - edge_red_dots
  let red_connections := 2 * corner_red_dots + 3 * edge_red_dots + 4 * interior_red_dots
  let red_segments := (red_connections - green_segments) / 2
  let blue_segments := total_segments - red_segments - green_segments
  blue_segments = 37 := by sorry

end blue_segments_count_l527_52703


namespace f_monotonicity_and_extreme_value_l527_52779

noncomputable section

def f (x : ℝ) := Real.log x - x

theorem f_monotonicity_and_extreme_value :
  (∀ x₁ x₂, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1 → f x₁ < f x₂) ∧
  (∀ x₁ x₂, 1 < x₁ ∧ x₁ < x₂ → f x₁ > f x₂) ∧
  (∀ x, 0 < x → f x ≤ f 1) ∧
  f 1 = -1 :=
by sorry

end f_monotonicity_and_extreme_value_l527_52779


namespace red_jelly_beans_l527_52762

/-- The number of red jelly beans in a bag, given the following conditions:
  1. It takes three bags of jelly beans to fill the fishbowl.
  2. Each bag has a similar distribution of colors.
  3. One bag contains: 13 black, 36 green, 28 purple, 32 yellow, and 18 white jelly beans.
  4. The total number of red and white jelly beans in the fishbowl is 126. -/
theorem red_jelly_beans (black green purple yellow white : ℕ)
  (h1 : black = 13)
  (h2 : green = 36)
  (h3 : purple = 28)
  (h4 : yellow = 32)
  (h5 : white = 18)
  (h6 : (red + white) * 3 = 126) :
  red = 24 :=
sorry

end red_jelly_beans_l527_52762


namespace min_value_reciprocal_product_l527_52743

theorem min_value_reciprocal_product (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_mean : a + 2*b = 6) : 
  (∀ x y, x > 0 → y > 0 → x + 2*y = 6 → (1 / (a * b)) ≤ (1 / (x * y))) → 
  (1 / (a * b)) = 2/9 :=
sorry

end min_value_reciprocal_product_l527_52743


namespace two_identical_squares_exist_l527_52754

-- Define the type for the table entries
inductive Entry
| Zero
| One

-- Define the 5x5 table
def Table := Fin 5 → Fin 5 → Entry

-- Define the property of having ones in top-left and bottom-right corners, and zeros in the other corners
def CornerCondition (t : Table) : Prop :=
  t 0 0 = Entry.One ∧
  t 4 4 = Entry.One ∧
  t 0 4 = Entry.Zero ∧
  t 4 0 = Entry.Zero

-- Define a 2x2 square in the table
def Square (t : Table) (i j : Fin 4) : Fin 2 → Fin 2 → Entry :=
  fun x y => t (i + x) (j + y)

-- Define when two squares are equal
def SquaresEqual (s1 s2 : Fin 2 → Fin 2 → Entry) : Prop :=
  ∀ (x y : Fin 2), s1 x y = s2 x y

-- The main theorem
theorem two_identical_squares_exist (t : Table) (h : CornerCondition t) :
  ∃ (i1 j1 i2 j2 : Fin 4), (i1, j1) ≠ (i2, j2) ∧
    SquaresEqual (Square t i1 j1) (Square t i2 j2) := by
  sorry

end two_identical_squares_exist_l527_52754


namespace hyperbola_equation_proof_l527_52774

/-- Represents a hyperbola with given properties -/
structure Hyperbola where
  /-- The eccentricity of the hyperbola -/
  e : ℝ
  /-- The distance from the focus to the asymptote -/
  d : ℝ
  /-- The hyperbola is centered at the origin -/
  center_origin : True
  /-- The foci are on the x-axis -/
  foci_on_x_axis : True
  /-- The eccentricity is √6/2 -/
  e_value : e = Real.sqrt 6 / 2
  /-- The distance from the focus to the asymptote is 1 -/
  d_value : d = 1

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 = 1

/-- Theorem stating that the given hyperbola has the specified equation -/
theorem hyperbola_equation_proof (h : Hyperbola) :
  ∀ x y : ℝ, hyperbola_equation h x y ↔ x^2 / 2 - y^2 = 1 :=
by sorry

end hyperbola_equation_proof_l527_52774


namespace expression_value_l527_52787

theorem expression_value (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : abs a ≠ abs b) :
  let expr1 := b^2 / a^2 + a^2 / b^2 - 2
  let expr2 := (a + b) / (b - a) + (b - a) / (a + b)
  let expr3 := (1 / a^2 + 1 / b^2) / (1 / b^2 - 1 / a^2) - (1 / b^2 - 1 / a^2) / (1 / a^2 + 1 / b^2)
  expr1 * expr2 * expr3 = -8 := by sorry

end expression_value_l527_52787


namespace meal_cost_proof_l527_52797

/-- Given the cost of two different meal combinations, 
    prove the cost of a single sandwich, coffee, and pie. -/
theorem meal_cost_proof (sandwich_cost coffee_cost pie_cost : ℚ) : 
  5 * sandwich_cost + 8 * coffee_cost + 2 * pie_cost = (5.40 : ℚ) →
  3 * sandwich_cost + 11 * coffee_cost + 2 * pie_cost = (4.95 : ℚ) →
  sandwich_cost + coffee_cost + pie_cost = (1.55 : ℚ) := by
  sorry

end meal_cost_proof_l527_52797


namespace min_value_theorem_l527_52781

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ (min_val : ℝ), min_val = 2 * Real.sqrt 3 ∧
   ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 →
   (2 * x^2 + 1) / (x * y) - 2 ≥ min_val) ∧
  (2 * a^2 + 1) / (a * b) - 2 = 2 * Real.sqrt 3 ↔ a = (Real.sqrt 3 - 1) / 2 :=
sorry

end min_value_theorem_l527_52781


namespace negation_of_universal_proposition_l527_52767

def A : Set ℤ := {x | ∃ k, x = 2*k + 1}
def B : Set ℤ := {x | ∃ k, x = 2*k}

theorem negation_of_universal_proposition :
  (¬ (∀ x ∈ A, (2 * x) ∈ B)) ↔ (∃ x ∈ A, (2 * x) ∉ B) :=
sorry

end negation_of_universal_proposition_l527_52767


namespace prob_both_selected_l527_52745

/-- The probability of brother X being selected -/
def prob_X : ℚ := 1/5

/-- The probability of brother Y being selected -/
def prob_Y : ℚ := 2/3

/-- Theorem: The probability of both brothers X and Y being selected is 2/15 -/
theorem prob_both_selected : prob_X * prob_Y = 2/15 := by
  sorry

end prob_both_selected_l527_52745


namespace arithmetic_sequence_product_l527_52770

/-- An arithmetic sequence of integers -/
def ArithmeticSequence (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

/-- The sequence is increasing -/
def IncreasingSequence (b : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → b n < b m

theorem arithmetic_sequence_product (b : ℕ → ℤ) :
  ArithmeticSequence b →
  IncreasingSequence b →
  b 4 * b 5 = 30 →
  (b 3 * b 6 = -1652 ∨ b 3 * b 6 = -308 ∨ b 3 * b 6 = -68 ∨ b 3 * b 6 = 28) :=
by sorry

end arithmetic_sequence_product_l527_52770


namespace thomas_needs_2000_more_l527_52720

/-- Thomas's savings scenario over two years -/
structure SavingsScenario where
  allowance_per_week : ℕ
  weeks_in_year : ℕ
  hourly_wage : ℕ
  hours_per_week : ℕ
  car_cost : ℕ
  weekly_expenses : ℕ

/-- Calculate the amount Thomas needs to save more -/
def amount_needed_more (s : SavingsScenario) : ℕ :=
  let first_year_savings := s.allowance_per_week * s.weeks_in_year
  let second_year_earnings := s.hourly_wage * s.hours_per_week * s.weeks_in_year
  let total_earnings := first_year_savings + second_year_earnings
  let total_expenses := s.weekly_expenses * (2 * s.weeks_in_year)
  let net_savings := total_earnings - total_expenses
  s.car_cost - net_savings

/-- Thomas's specific savings scenario -/
def thomas_scenario : SavingsScenario :=
  { allowance_per_week := 50
  , weeks_in_year := 52
  , hourly_wage := 9
  , hours_per_week := 30
  , car_cost := 15000
  , weekly_expenses := 35 }

/-- Theorem stating that Thomas needs $2000 more to buy the car -/
theorem thomas_needs_2000_more :
  amount_needed_more thomas_scenario = 2000 := by sorry

end thomas_needs_2000_more_l527_52720


namespace part_one_part_two_l527_52737

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + 3| - m + 1

-- Part I
theorem part_one (m : ℝ) (h1 : m > 0) 
  (h2 : Set.Iic (-2) ∪ Set.Ici 2 = {x | f m (x - 3) ≥ 0}) : 
  m = 3 := by sorry

-- Part II
theorem part_two : 
  {t : ℝ | ∃ x, |x + 3| - 2 ≥ |2*x - 1| - t^2 + 5/2*t} = 
  Set.Iic 1 ∪ Set.Ici (3/2) := by sorry

end part_one_part_two_l527_52737


namespace m_range_for_p_necessary_not_sufficient_for_q_l527_52772

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 - 8*x - 20 > 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 > 0

-- Define the sets A and B
def A : Set ℝ := {x | p x}
def B (m : ℝ) : Set ℝ := {x | q x m}

-- Theorem statement
theorem m_range_for_p_necessary_not_sufficient_for_q :
  ∀ m : ℝ, (m > 0 ∧ (∀ x : ℝ, q x m → p x) ∧ (∃ x : ℝ, p x ∧ ¬q x m)) ↔ m ≥ 9 :=
sorry

end m_range_for_p_necessary_not_sufficient_for_q_l527_52772


namespace points_in_small_square_l527_52755

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square region in a 2D plane -/
structure Square where
  center : Point
  side_length : ℝ

/-- Check if a point is inside a square -/
def is_point_in_square (p : Point) (s : Square) : Prop :=
  abs (p.x - s.center.x) ≤ s.side_length / 2 ∧
  abs (p.y - s.center.y) ≤ s.side_length / 2

/-- The main theorem -/
theorem points_in_small_square (points : Finset Point) 
    (h1 : points.card = 51)
    (h2 : ∀ p ∈ points, is_point_in_square p ⟨⟨0.5, 0.5⟩, 1⟩) :
    ∃ (small_square : Square),
      small_square.side_length = 0.2 ∧
      ∃ (p1 p2 p3 : Point),
        p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧
        p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧
        is_point_in_square p1 small_square ∧
        is_point_in_square p2 small_square ∧
        is_point_in_square p3 small_square :=
  sorry

end points_in_small_square_l527_52755


namespace poly_simplification_poly_evaluation_l527_52725

-- Define the original polynomial
def original_poly (x : ℝ) : ℝ :=
  (2*x^5 - 3*x^4 + 5*x^3 - 9*x^2 + 8*x - 15) + (5*x^4 - 2*x^3 + 3*x^2 - 4*x + 9)

-- Define the simplified polynomial
def simplified_poly (x : ℝ) : ℝ :=
  2*x^5 + 2*x^4 + 3*x^3 - 6*x^2 + 4*x - 6

-- Theorem stating that the original polynomial equals the simplified polynomial
theorem poly_simplification (x : ℝ) : original_poly x = simplified_poly x := by
  sorry

-- Theorem stating that the simplified polynomial evaluated at x = 2 equals 98
theorem poly_evaluation : simplified_poly 2 = 98 := by
  sorry

end poly_simplification_poly_evaluation_l527_52725


namespace quadratic_real_roots_l527_52747

theorem quadratic_real_roots (a b c : ℝ) (ha : a ≠ 0) (hac : a * c < 0) :
  ∃ x : ℝ, a * x^2 + b * x + c = 0 := by
sorry

end quadratic_real_roots_l527_52747


namespace min_value_of_expression_l527_52730

/-- The minimum value of 1/a + 4/b given the conditions -/
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x y : ℝ, a * x - b * y + 2 = 0 → x^2 + y^2 + 2*x - 2*y = 0) → 
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (a * A.1 - b * A.2 + 2 = 0) ∧ (A.1^2 + A.2^2 + 2*A.1 - 2*A.2 = 0) ∧
    (a * B.1 - b * B.2 + 2 = 0) ∧ (B.1^2 + B.2^2 + 2*B.1 - 2*B.2 = 0) ∧
    (∀ C D : ℝ × ℝ, C ≠ D → 
      (a * C.1 - b * C.2 + 2 = 0) → (C.1^2 + C.2^2 + 2*C.1 - 2*C.2 = 0) →
      (a * D.1 - b * D.2 + 2 = 0) → (D.1^2 + D.2^2 + 2*D.1 - 2*D.2 = 0) →
      (A.1 - B.1)^2 + (A.2 - B.2)^2 ≥ (C.1 - D.1)^2 + (C.2 - D.2)^2)) →
  (1/a + 4/b) ≥ 9/2 :=
sorry

end min_value_of_expression_l527_52730


namespace logarithm_calculation_l527_52716

theorem logarithm_calculation : 
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 - (5 : ℝ) ^ (Real.log 3 / Real.log 5) = -1 :=
by sorry

-- Note: We cannot include the second part of the problem due to inconsistencies in the problem statement and solution.

end logarithm_calculation_l527_52716


namespace pizza_coverage_theorem_l527_52728

/-- Represents the properties of a pizza with pepperoni -/
structure PepperoniPizza where
  pizza_diameter : ℝ
  pepperoni_across : ℕ
  total_pepperoni : ℕ

/-- Calculates the fraction of pizza covered by pepperoni -/
def pepperoni_coverage (p : PepperoniPizza) : ℚ :=
  sorry

/-- Theorem stating the fraction of pizza covered by pepperoni for the given conditions -/
theorem pizza_coverage_theorem (p : PepperoniPizza) 
  (h1 : p.pizza_diameter = 18)
  (h2 : p.pepperoni_across = 8)
  (h3 : p.total_pepperoni = 36) :
  pepperoni_coverage p = 9/16 := by
  sorry

end pizza_coverage_theorem_l527_52728


namespace negation_of_universal_proposition_l527_52764

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 2*x + 5 > 0) ↔ (∃ x : ℝ, x^2 + 2*x + 5 ≤ 0) := by sorry

end negation_of_universal_proposition_l527_52764


namespace train_length_train_length_proof_l527_52782

/-- Calculates the length of a train given its speed and the time it takes to cross a bridge of known length. -/
theorem train_length (train_speed : ℝ) (bridge_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let speed_ms := train_speed * 1000 / 3600
  let total_distance := speed_ms * crossing_time
  total_distance - bridge_length

/-- Proves that a train traveling at 40 km/h that crosses a 300-meter bridge in 45 seconds has a length of approximately 199.95 meters. -/
theorem train_length_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |train_length 40 300 45 - 199.95| < ε :=
sorry

end train_length_train_length_proof_l527_52782


namespace det_scale_by_three_l527_52719

theorem det_scale_by_three (x y z w : ℝ) :
  Matrix.det !![x, y; z, w] = 7 →
  Matrix.det !![3*x, 3*y; 3*z, 3*w] = 63 := by
  sorry

end det_scale_by_three_l527_52719


namespace arithmetic_sequence_sum_l527_52742

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum 
  (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) 
  (h2 : a 4 = 12) : 
  a 1 + a 7 = 24 := by
  sorry

end arithmetic_sequence_sum_l527_52742


namespace toy_price_calculation_l527_52723

theorem toy_price_calculation (toy_price : ℝ) : 
  (3 * toy_price + 2 * 5 + 5 * 6 = 70) → toy_price = 10 := by
  sorry

end toy_price_calculation_l527_52723


namespace min_abs_phi_l527_52727

/-- Given a function y = 2sin(2x - φ) whose graph is symmetric about the point (4π/3, 0),
    the minimum value of |φ| is π/3 -/
theorem min_abs_phi (φ : ℝ) : 
  (∀ x : ℝ, 2 * Real.sin (2 * x - φ) = 2 * Real.sin (2 * (8 * π / 3 - x) - φ)) →
  ∃ k : ℤ, φ = 8 * π / 3 - k * π →
  |φ| ≥ π / 3 ∧ ∃ φ₀ : ℝ, |φ₀| = π / 3 ∧ 
    (∀ x : ℝ, 2 * Real.sin (2 * x - φ₀) = 2 * Real.sin (2 * (8 * π / 3 - x) - φ₀)) :=
by sorry

end min_abs_phi_l527_52727


namespace inequality_solution_range_l527_52757

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| + |x - 3| < a) → a > 4 := by
sorry

end inequality_solution_range_l527_52757


namespace max_value_of_z_l527_52752

/-- Given real numbers x and y satisfying the conditions,
    prove that the maximum value of z = 2x - y is 5 -/
theorem max_value_of_z (x y : ℝ) 
  (h1 : x - 2*y + 2 ≥ 0) 
  (h2 : x + y ≤ 1) 
  (h3 : y + 1 ≥ 0) : 
  ∃ (z : ℝ), z = 2*x - y ∧ z ≤ 5 ∧ ∀ (w : ℝ), w = 2*x - y → w ≤ 5 :=
sorry

end max_value_of_z_l527_52752


namespace library_tables_l527_52701

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : Nat) : Nat :=
  let units := n % 10
  let sixes := (n / 10) % 10
  let thirty_sixes := n / 100
  thirty_sixes * 36 + sixes * 6 + units

/-- Calculates the number of tables needed given the total number of people and people per table -/
def tablesNeeded (totalPeople : Nat) (peoplePerTable : Nat) : Nat :=
  (totalPeople + peoplePerTable - 1) / peoplePerTable

theorem library_tables (seatingCapacity : Nat) (peoplePerTable : Nat) :
  seatingCapacity = 231 ∧ peoplePerTable = 3 →
  tablesNeeded (base6ToBase10 seatingCapacity) peoplePerTable = 31 := by
  sorry

end library_tables_l527_52701


namespace circle_radius_from_diameter_l527_52733

theorem circle_radius_from_diameter (diameter : ℝ) (radius : ℝ) :
  diameter = 14 → radius = diameter / 2 → radius = 7 := by sorry

end circle_radius_from_diameter_l527_52733


namespace inequality_proof_l527_52729

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + b + c ≥ a * b * c) : 
  (2/a + 3/b + 6/c ≥ 6 ∧ 2/b + 3/c + 6/a ≥ 6) ∨
  (2/b + 3/c + 6/a ≥ 6 ∧ 2/c + 3/a + 6/b ≥ 6) ∨
  (2/c + 3/a + 6/b ≥ 6 ∧ 2/a + 3/b + 6/c ≥ 6) := by
  sorry

end inequality_proof_l527_52729


namespace largest_x_satisfying_equation_l527_52756

theorem largest_x_satisfying_equation : 
  ∃ (x : ℝ), ∀ (y : ℝ), 
    (|y^2 - 11*y + 24| + |2*y^2 + 6*y - 56| = |y^2 + 17*y - 80|) → 
    y ≤ x ∧ 
    |x^2 - 11*x + 24| + |2*x^2 + 6*x - 56| = |x^2 + 17*x - 80| ∧
    x = 8 := by
  sorry

end largest_x_satisfying_equation_l527_52756


namespace g_behavior_at_infinity_l527_52784

def g (x : ℝ) : ℝ := -3 * x^3 + 50 * x^2 - 4 * x + 10

theorem g_behavior_at_infinity :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → g x < M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < -N → g x > M) := by
sorry

end g_behavior_at_infinity_l527_52784


namespace distance_to_big_rock_l527_52715

/-- The distance to Big Rock given the rower's speed, river current, and round trip time -/
theorem distance_to_big_rock (v : ℝ) (c : ℝ) (t : ℝ) (h1 : v = 6) (h2 : c = 1) (h3 : t = 1) :
  ∃ d : ℝ, d = 35 / 12 ∧ d / (v - c) + d / (v + c) = t :=
by sorry

end distance_to_big_rock_l527_52715


namespace parabola_directrix_l527_52717

/-- The equation of a parabola -/
def parabola_equation (x y : ℝ) : Prop := x^2 + 12*y = 0

/-- The equation of the directrix -/
def directrix_equation (y : ℝ) : Prop := y = 3

/-- Theorem: The directrix of the parabola x^2 + 12y = 0 is y = 3 -/
theorem parabola_directrix :
  ∀ x y : ℝ, parabola_equation x y → directrix_equation y :=
by sorry

end parabola_directrix_l527_52717


namespace tangent_sum_difference_l527_52791

theorem tangent_sum_difference (α β : Real) 
  (h1 : Real.tan (α + β) = 2/5)
  (h2 : Real.tan (β - π/4) = 1/4) :
  Real.tan (α + π/4) = 3/22 := by
  sorry

end tangent_sum_difference_l527_52791


namespace beads_per_necklace_l527_52786

theorem beads_per_necklace (total_beads : ℕ) (num_necklaces : ℕ) 
  (h1 : total_beads = 28) (h2 : num_necklaces = 4) :
  total_beads / num_necklaces = 7 := by
  sorry

end beads_per_necklace_l527_52786


namespace parallelogram_area_l527_52710

/-- The area of a parallelogram with base 26 cm and height 16 cm is 416 square centimeters. -/
theorem parallelogram_area : 
  let base : ℝ := 26
  let height : ℝ := 16
  let area : ℝ := base * height
  area = 416 := by
  sorry

end parallelogram_area_l527_52710


namespace tin_can_equation_l527_52760

/-- Represents the number of can bodies that can be made from one sheet of tinplate -/
def bodies_per_sheet : ℕ := 15

/-- Represents the number of can bottoms that can be made from one sheet of tinplate -/
def bottoms_per_sheet : ℕ := 42

/-- Represents the total number of available sheets of tinplate -/
def total_sheets : ℕ := 108

/-- Represents the number of can bottoms needed for one complete tin can -/
def bottoms_per_can : ℕ := 2

theorem tin_can_equation (x : ℕ) :
  x ≤ total_sheets →
  (bottoms_per_can * bodies_per_sheet * x = bottoms_per_sheet * (total_sheets - x)) ↔
  (2 * 15 * x = 42 * (108 - x)) :=
by sorry

end tin_can_equation_l527_52760


namespace sector_area_l527_52700

theorem sector_area (r : ℝ) (θ : ℝ) (h1 : r = 6) (h2 : θ = π / 6) :
  (1 / 2) * r^2 * θ = 3 * π := by
  sorry

#check sector_area

end sector_area_l527_52700


namespace baker_sales_difference_l527_52777

theorem baker_sales_difference (cakes_made pastries_made cakes_sold pastries_sold : ℕ) : 
  cakes_made = 14 →
  pastries_made = 153 →
  cakes_sold = 97 →
  pastries_sold = 8 →
  cakes_sold - pastries_sold = 89 := by
sorry

end baker_sales_difference_l527_52777


namespace alices_journey_time_l527_52775

/-- Represents the problem of Alice's journey to the library -/
theorem alices_journey_time :
  ∀ (d : ℝ) (r_w : ℝ),
    r_w > 0 →
    (3/4 * d) / r_w = 9 →
    (1/4 * d) / (4 * r_w) + 9 = 9.75 :=
by sorry

end alices_journey_time_l527_52775


namespace equation_solution_and_condition_l527_52765

theorem equation_solution_and_condition :
  ∃ x : ℝ, (3 * x + 7 = 22) ∧ (2 * x + 1 ≠ 9) := by
  sorry

end equation_solution_and_condition_l527_52765


namespace bus_students_count_l527_52788

/-- The number of students on the left side of the bus -/
def left_students : ℕ := 36

/-- The number of students on the right side of the bus -/
def right_students : ℕ := 27

/-- The total number of students on the bus -/
def total_students : ℕ := left_students + right_students

/-- Theorem: The total number of students on the bus is 63 -/
theorem bus_students_count : total_students = 63 := by
  sorry

end bus_students_count_l527_52788


namespace largest_divisible_n_l527_52741

theorem largest_divisible_n : ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > n → ¬((m + 12) ∣ (m^3 + 144))) ∧ 
  ((n + 12) ∣ (n^3 + 144)) ∧ 
  n = 84 := by
  sorry

end largest_divisible_n_l527_52741


namespace complement_union_M_N_l527_52704

def I : Set ℕ := {x | x ≤ 10}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 4, 6, 8, 10}

theorem complement_union_M_N :
  (M ∪ N)ᶜ = {0, 5, 7, 9} := by sorry

end complement_union_M_N_l527_52704


namespace particle_probability_l527_52706

def move_probability (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1/3) * move_probability (x-1) y +
       (1/3) * move_probability x (y-1) +
       (1/3) * move_probability (x-1) (y-1)

theorem particle_probability :
  move_probability 4 4 = 245 / 3^7 :=
sorry

end particle_probability_l527_52706


namespace inequality_solution_set_l527_52753

theorem inequality_solution_set : 
  {x : ℝ | (x - 1) * (x + 2) < 0} = Set.Ioo (-2 : ℝ) (1 : ℝ) := by
  sorry

end inequality_solution_set_l527_52753


namespace solution_set_quadratic_inequality_l527_52769

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, -x^2 + 2*x + 3 ≥ 0 ↔ x ∈ Set.Icc (-1) 3 :=
by sorry

end solution_set_quadratic_inequality_l527_52769


namespace apple_percentage_after_removal_l527_52771

def initial_apples : ℕ := 10
def initial_oranges : ℕ := 23
def oranges_removed : ℕ := 13

def remaining_oranges : ℕ := initial_oranges - oranges_removed
def total_fruit_after : ℕ := initial_apples + remaining_oranges

theorem apple_percentage_after_removal : 
  (initial_apples : ℚ) / total_fruit_after * 100 = 50 := by
  sorry

end apple_percentage_after_removal_l527_52771


namespace savings_calculation_l527_52761

/-- Calculates a person's savings given their income and income-to-expenditure ratio -/
def calculate_savings (income : ℚ) (income_ratio : ℚ) (expenditure_ratio : ℚ) : ℚ :=
  income - (income * expenditure_ratio / income_ratio)

/-- Proves that for a given income and income-to-expenditure ratio, the savings are as calculated -/
theorem savings_calculation (income : ℚ) (income_ratio : ℚ) (expenditure_ratio : ℚ) :
  income = 20000 ∧ income_ratio = 4 ∧ expenditure_ratio = 3 →
  calculate_savings income income_ratio expenditure_ratio = 5000 :=
by sorry

end savings_calculation_l527_52761


namespace possible_sums_B_l527_52735

theorem possible_sums_B (a b c d : ℕ+) 
  (h1 : a * b = 2 * (c + d))
  (h2 : c * d = 2 * (a + b))
  (h3 : a + b ≥ c + d) :
  c + d = 13 ∨ c + d = 10 ∨ c + d = 9 ∨ c + d = 8 := by
sorry

end possible_sums_B_l527_52735


namespace ratio_odd_even_divisors_l527_52792

def N : ℕ := 38 * 38 * 91 * 210

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_odd_even_divisors :
  (sum_odd_divisors N) * 14 = sum_even_divisors N := by sorry

end ratio_odd_even_divisors_l527_52792


namespace opposite_of_one_l527_52709

theorem opposite_of_one : ∃ x : ℤ, x + 1 = 0 ∧ x = -1 := by
  sorry

end opposite_of_one_l527_52709


namespace difference_of_squares_l527_52794

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end difference_of_squares_l527_52794


namespace intersection_of_A_and_B_l527_52795

-- Define sets A and B
def A : Set ℝ := {y | ∃ x, y = 2^x}
def B : Set ℝ := {y | ∃ x, y = -x^2 + 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {y | 0 < y ∧ y ≤ 2} := by sorry

end intersection_of_A_and_B_l527_52795


namespace ring_payment_possible_l527_52718

/-- Represents a chain of rings -/
structure RingChain :=
  (size : ℕ)

/-- Represents a cut ring chain -/
structure CutRingChain :=
  (segments : List RingChain)
  (total_size : ℕ)

/-- Represents a daily payment -/
structure DailyPayment :=
  (rings_given : ℕ)
  (rings_taken : ℕ)

def is_valid_payment_sequence (payments : List DailyPayment) : Prop :=
  payments.length = 7 ∧
  ∀ p ∈ payments, p.rings_given - p.rings_taken = 1

def can_make_payments (chain : RingChain) : Prop :=
  ∃ (cut_chain : CutRingChain) (payments : List DailyPayment),
    chain.size = 7 ∧
    cut_chain.total_size = 7 ∧
    cut_chain.segments.length ≤ 3 ∧
    is_valid_payment_sequence payments

theorem ring_payment_possible :
  ∃ (chain : RingChain), can_make_payments chain :=
sorry

end ring_payment_possible_l527_52718


namespace stratified_sampling_size_l527_52780

/-- Represents the number of units produced by each workshop -/
structure WorkshopProduction where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents the sampling information -/
structure SamplingInfo where
  total_sample : ℕ
  workshop_b_sample : ℕ

/-- Theorem stating the correct sample size for the given scenario -/
theorem stratified_sampling_size 
  (prod : WorkshopProduction)
  (sample : SamplingInfo)
  (h1 : prod.a = 96)
  (h2 : prod.b = 84)
  (h3 : prod.c = 60)
  (h4 : sample.workshop_b_sample = 7)
  (h5 : sample.workshop_b_sample / sample.total_sample = prod.b / (prod.a + prod.b + prod.c)) :
  sample.total_sample = 70 := by
  sorry


end stratified_sampling_size_l527_52780
