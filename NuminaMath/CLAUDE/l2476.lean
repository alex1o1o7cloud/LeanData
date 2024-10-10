import Mathlib

namespace kaleb_cherries_left_l2476_247611

/-- Calculates the number of cherries Kaleb has left after eating some. -/
def cherries_left (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem: Given Kaleb had 67 cherries initially and ate 25 cherries,
    the number of cherries he had left is equal to 42. -/
theorem kaleb_cherries_left : cherries_left 67 25 = 42 := by
  sorry

end kaleb_cherries_left_l2476_247611


namespace box_of_balls_l2476_247688

theorem box_of_balls (N : ℕ) : N - 44 = 70 - N → N = 57 := by
  sorry

end box_of_balls_l2476_247688


namespace total_adoption_cost_l2476_247674

def cat_cost : ℕ := 50
def adult_dog_cost : ℕ := 100
def puppy_cost : ℕ := 150
def num_cats : ℕ := 2
def num_adult_dogs : ℕ := 3
def num_puppies : ℕ := 2

theorem total_adoption_cost :
  cat_cost * num_cats + adult_dog_cost * num_adult_dogs + puppy_cost * num_puppies = 700 := by
  sorry

end total_adoption_cost_l2476_247674


namespace binomial_coefficient_equation_unique_solution_l2476_247668

theorem binomial_coefficient_equation_unique_solution : 
  ∃! n : ℕ, Nat.choose 25 n + Nat.choose 25 12 = Nat.choose 26 13 ∧ n = 13 := by
  sorry

end binomial_coefficient_equation_unique_solution_l2476_247668


namespace expression_evaluation_l2476_247600

theorem expression_evaluation : 
  (2020^3 - 3 * 2020^2 * 2021 + 5 * 2020 * 2021^2 - 2021^3 + 4) / (2020 * 2021) = 
  4042 + 3 / (4080420 : ℚ) := by
sorry

end expression_evaluation_l2476_247600


namespace gold_coin_percentage_l2476_247691

/-- Represents the composition of objects in an urn --/
structure UrnComposition where
  beadPercentage : ℝ
  silverCoinPercentage : ℝ
  goldCoinPercentage : ℝ

/-- The urn composition satisfies the given conditions --/
def validUrnComposition (u : UrnComposition) : Prop :=
  u.beadPercentage = 30 ∧
  u.silverCoinPercentage + u.goldCoinPercentage = 70 ∧
  u.silverCoinPercentage = 35

theorem gold_coin_percentage (u : UrnComposition) 
  (h : validUrnComposition u) : u.goldCoinPercentage = 35 := by
  sorry

#check gold_coin_percentage

end gold_coin_percentage_l2476_247691


namespace coalition_percentage_is_79_percent_l2476_247610

/-- Represents the election results and voter information -/
structure ElectionData where
  total_votes : ℕ
  invalid_vote_percentage : ℚ
  registered_voters : ℕ
  candidate_x_valid_percentage : ℚ
  candidate_y_valid_percentage : ℚ
  candidate_z_valid_percentage : ℚ

/-- Calculates the percentage of valid votes received by a coalition of two candidates -/
def coalition_percentage (data : ElectionData) : ℚ :=
  data.candidate_x_valid_percentage + data.candidate_y_valid_percentage

/-- Theorem stating that the coalition of candidates X and Y received 79% of the valid votes -/
theorem coalition_percentage_is_79_percent (data : ElectionData)
  (h1 : data.total_votes = 750000)
  (h2 : data.invalid_vote_percentage = 18 / 100)
  (h3 : data.registered_voters = 900000)
  (h4 : data.candidate_x_valid_percentage = 47 / 100)
  (h5 : data.candidate_y_valid_percentage = 32 / 100)
  (h6 : data.candidate_z_valid_percentage = 21 / 100) :
  coalition_percentage data = 79 / 100 := by
  sorry


end coalition_percentage_is_79_percent_l2476_247610


namespace solution_satisfies_relationship_l2476_247640

theorem solution_satisfies_relationship (x y : ℝ) : 
  (2 * x + y = 7) → (x - y = 5) → (x + 2 * y = 2) := by
  sorry

end solution_satisfies_relationship_l2476_247640


namespace find_A_l2476_247638

theorem find_A (A B : ℕ) (h : 15 = 3 * A ∧ 15 = 5 * B) : A = 5 := by
  sorry

end find_A_l2476_247638


namespace apartment_count_l2476_247622

theorem apartment_count : 
  ∀ (total : ℕ) 
    (at_least_one : ℕ) 
    (at_least_two : ℕ) 
    (only_one : ℕ),
  at_least_one = (85 * total) / 100 →
  at_least_two = (60 * total) / 100 →
  only_one = 30 →
  only_one = at_least_one - at_least_two →
  total = 75 := by
sorry

end apartment_count_l2476_247622


namespace intersection_of_lines_l2476_247627

theorem intersection_of_lines (k : ℝ) : 
  (∃ x y : ℝ, y = -2 * x + 3 ∧ y = k * x + 4 ∧ x = 1 ∧ y = 1) → k = -3 := by
  sorry

end intersection_of_lines_l2476_247627


namespace commute_distance_is_21_l2476_247661

/-- Represents the carpool scenario with given parameters -/
structure Carpool where
  friends : ℕ := 5
  gas_price : ℚ := 5/2
  car_efficiency : ℚ := 30
  commute_days_per_week : ℕ := 5
  commute_weeks_per_month : ℕ := 4
  individual_payment : ℚ := 14

/-- Calculates the one-way commute distance given a Carpool scenario -/
def calculate_commute_distance (c : Carpool) : ℚ :=
  (c.individual_payment * c.friends * c.car_efficiency) / 
  (2 * c.gas_price * c.commute_days_per_week * c.commute_weeks_per_month)

/-- Theorem stating that the one-way commute distance is 21 miles -/
theorem commute_distance_is_21 (c : Carpool) : 
  calculate_commute_distance c = 21 := by
  sorry

end commute_distance_is_21_l2476_247661


namespace point_B_coordinates_l2476_247671

def point_A : ℝ × ℝ := (-1, 5)
def vector_a : ℝ × ℝ := (2, 3)

theorem point_B_coordinates :
  ∀ (B : ℝ × ℝ),
  (B.1 - point_A.1, B.2 - point_A.2) = (3 * vector_a.1, 3 * vector_a.2) →
  B = (5, 14) := by
sorry

end point_B_coordinates_l2476_247671


namespace masha_sasha_numbers_l2476_247621

theorem masha_sasha_numbers : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  a > 11 ∧ 
  b > 11 ∧ 
  (∀ (x y : ℕ), x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y < a + b → 
    ∃! (p q : ℕ), p ≠ q ∧ p > 11 ∧ q > 11 ∧ p + q = x + y) ∧
  (Even a ∨ Even b) ∧
  (∀ (x y : ℕ), x ≠ y ∧ x > 11 ∧ y > 11 ∧ x + y = a + b → (x = 12 ∧ y = 16) ∨ (x = 16 ∧ y = 12)) :=
by
  sorry

end masha_sasha_numbers_l2476_247621


namespace shelter_cat_dog_difference_l2476_247645

/-- Given an animal shelter with a total of 60 animals and 40 cats,
    prove that the number of cats exceeds the number of dogs by 20. -/
theorem shelter_cat_dog_difference :
  let total_animals : ℕ := 60
  let num_cats : ℕ := 40
  let num_dogs : ℕ := total_animals - num_cats
  num_cats - num_dogs = 20 := by
  sorry

end shelter_cat_dog_difference_l2476_247645


namespace square_of_binomial_l2476_247647

theorem square_of_binomial (d : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, x^2 - 60*x + d = (a*x + b)^2) → d = 900 := by
  sorry

end square_of_binomial_l2476_247647


namespace intersection_implies_x_value_l2476_247693

def A (x : ℝ) : Set ℝ := {9, 2 - x, x^2 + 1}
def B (x : ℝ) : Set ℝ := {1, 2 * x^2}

theorem intersection_implies_x_value :
  ∀ x : ℝ, A x ∩ B x = {2} → x = -1 := by
  sorry

end intersection_implies_x_value_l2476_247693


namespace audiobook_completion_time_l2476_247662

/-- Calculates the time to finish audiobooks given the number of books, length per book, and daily listening time. -/
def timeToFinishAudiobooks (numBooks : ℕ) (hoursPerBook : ℕ) (hoursPerDay : ℕ) : ℕ :=
  numBooks * (hoursPerBook / hoursPerDay)

/-- Proves that under the given conditions, it takes 90 days to finish the audiobooks. -/
theorem audiobook_completion_time :
  timeToFinishAudiobooks 6 30 2 = 90 :=
by
  sorry

#eval timeToFinishAudiobooks 6 30 2

end audiobook_completion_time_l2476_247662


namespace go_game_draw_probability_l2476_247633

theorem go_game_draw_probability 
  (p_not_lose : ℝ) 
  (p_win : ℝ) 
  (h1 : p_not_lose = 0.6) 
  (h2 : p_win = 0.5) : 
  p_not_lose - p_win = 0.1 := by
sorry

end go_game_draw_probability_l2476_247633


namespace sine_cosine_identity_l2476_247695

theorem sine_cosine_identity :
  Real.sin (65 * π / 180) * Real.cos (35 * π / 180) -
  Real.sin (25 * π / 180) * Real.sin (35 * π / 180) = 1 / 2 := by
  sorry

end sine_cosine_identity_l2476_247695


namespace cruise_liner_travelers_l2476_247632

theorem cruise_liner_travelers :
  ∃ a : ℕ,
    250 ≤ a ∧ a ≤ 400 ∧
    a % 15 = 8 ∧
    a % 25 = 17 ∧
    (a = 292 ∨ a = 367) :=
by sorry

end cruise_liner_travelers_l2476_247632


namespace triangle_perimeter_bound_l2476_247692

theorem triangle_perimeter_bound : 
  ∀ (a b c : ℝ), 
    a = 7 → 
    b = 21 → 
    a + b > c → 
    a + c > b → 
    b + c > a → 
    a + b + c < 56 :=
by
  sorry

end triangle_perimeter_bound_l2476_247692


namespace problem_solution_l2476_247636

theorem problem_solution (p q r s : ℝ) 
  (h : p^2 + q^2 + r^2 + 4 = s + Real.sqrt (p + q + r - s)) : 
  s = 5/4 := by
sorry

end problem_solution_l2476_247636


namespace waxing_time_is_36_minutes_l2476_247602

/-- Represents the time spent on different parts of car washing -/
structure CarWashTime where
  windows : ℕ
  body : ℕ
  tires : ℕ

/-- Calculates the total waxing time for all cars -/
def calculate_waxing_time (normal_car_time : CarWashTime) (normal_car_count : ℕ) (suv_count : ℕ) (total_time : ℕ) : ℕ :=
  let normal_car_wash_time := normal_car_time.windows + normal_car_time.body + normal_car_time.tires
  let total_wash_time_without_waxing := normal_car_wash_time * normal_car_count + (normal_car_wash_time * 2 * suv_count)
  total_time - total_wash_time_without_waxing

/-- Theorem stating that the waxing time is 36 minutes given the problem conditions -/
theorem waxing_time_is_36_minutes :
  let normal_car_time : CarWashTime := ⟨4, 7, 4⟩
  let normal_car_count : ℕ := 2
  let suv_count : ℕ := 1
  let total_time : ℕ := 96
  calculate_waxing_time normal_car_time normal_car_count suv_count total_time = 36 := by
  sorry

end waxing_time_is_36_minutes_l2476_247602


namespace square_inequality_equivalence_l2476_247672

theorem square_inequality_equivalence (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a > b ↔ a^2 > b^2 := by sorry

end square_inequality_equivalence_l2476_247672


namespace solution_set_characterization_l2476_247613

theorem solution_set_characterization (k : ℝ) :
  (∀ x : ℝ, (|x - 2007| + |x + 2007| = k) ↔ (x < -2007 ∨ x > 2007)) ↔ k > 4014 := by
  sorry

end solution_set_characterization_l2476_247613


namespace cubic_factor_implies_c_zero_l2476_247694

/-- Represents a cubic polynomial of the form ax^3 + bx^2 + cx + d -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents a quadratic polynomial of the form ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a linear polynomial of the form ax + b -/
structure LinearPolynomial where
  a : ℝ
  b : ℝ

def has_factor (p : CubicPolynomial) (q : QuadraticPolynomial) : Prop :=
  ∃ l : LinearPolynomial, 
    p.a * (q.a * l.a) = p.a ∧
    p.b * (q.a * l.b + q.b * l.a) = p.b ∧
    p.c * (q.b * l.b + q.c * l.a) = p.c ∧
    p.d * (q.c * l.b) = p.d

theorem cubic_factor_implies_c_zero 
  (p : CubicPolynomial) 
  (h : p.a = 3 ∧ p.b = 0 ∧ p.d = 12) 
  (q : QuadraticPolynomial) 
  (hq : q.a = 1 ∧ q.c = 2) 
  (h_factor : has_factor p q) : 
  p.c = 0 := by
  sorry

end cubic_factor_implies_c_zero_l2476_247694


namespace exist_three_distinct_digits_forming_squares_l2476_247650

/-- A function that constructs a three-digit number from three digits -/
def threeDigitNumber (a b c : Nat) : Nat :=
  100 * a + 10 * b + c

/-- Theorem stating the existence of three distinct digits forming squares -/
theorem exist_three_distinct_digits_forming_squares :
  ∃ (A B C : Nat),
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    ∃ (x y z : Nat),
      threeDigitNumber A B C = x^2 ∧
      threeDigitNumber C B A = y^2 ∧
      threeDigitNumber C A B = z^2 :=
by
  sorry

#eval threeDigitNumber 9 6 1
#eval threeDigitNumber 1 6 9
#eval threeDigitNumber 1 9 6

end exist_three_distinct_digits_forming_squares_l2476_247650


namespace trim_100_edge_polyhedron_l2476_247651

/-- Represents a polyhedron before and after vertex trimming --/
structure TrimmedPolyhedron where
  initial_edges : ℕ
  is_convex : Bool
  trimmed_vertices : ℕ
  trimmed_edges : ℕ

/-- Represents the process of trimming vertices of a polyhedron --/
def trim_vertices (p : TrimmedPolyhedron) : TrimmedPolyhedron :=
  { p with
    trimmed_vertices := 2 * p.initial_edges,
    trimmed_edges := 3 * p.initial_edges
  }

/-- Theorem stating the result of trimming vertices of a specific polyhedron --/
theorem trim_100_edge_polyhedron :
  ∀ p : TrimmedPolyhedron,
    p.initial_edges = 100 →
    p.is_convex = true →
    (trim_vertices p).trimmed_vertices = 200 ∧
    (trim_vertices p).trimmed_edges = 300 := by
  sorry


end trim_100_edge_polyhedron_l2476_247651


namespace power_relationship_l2476_247648

theorem power_relationship (y : ℝ) (h : (10 : ℝ) ^ (4 * y) = 49) : (10 : ℝ) ^ (-2 * y) = 1 / 7 := by
  sorry

end power_relationship_l2476_247648


namespace min_difference_is_one_l2476_247606

/-- Triangle with integer side lengths and specific properties -/
structure IntegerTriangle where
  DE : ℕ
  EF : ℕ
  FD : ℕ
  perimeter_eq : DE + EF + FD = 398
  side_order : DE < EF ∧ EF ≤ FD

/-- The minimum difference between EF and DE in an IntegerTriangle is 1 -/
theorem min_difference_is_one :
  ∀ t : IntegerTriangle, (∀ s : IntegerTriangle, t.EF - t.DE ≤ s.EF - s.DE) → t.EF - t.DE = 1 := by
  sorry

#check min_difference_is_one

end min_difference_is_one_l2476_247606


namespace sqrt_two_equals_two_to_one_sixth_l2476_247669

theorem sqrt_two_equals_two_to_one_sixth : ∃ (x : ℝ), x > 0 ∧ x^2 = 2 ∧ x = 2^(1/6) := by sorry

end sqrt_two_equals_two_to_one_sixth_l2476_247669


namespace sum_first_last_number_l2476_247683

theorem sum_first_last_number (a b c d : ℝ) : 
  (a + b + c) / 3 = 6 →
  (b + c + d) / 3 = 5 →
  d = 4 →
  a + d = 11 := by
sorry

end sum_first_last_number_l2476_247683


namespace reading_difference_l2476_247631

/-- Calculates the total pages read given a list of (rate, days) pairs -/
def totalPagesRead (readingPlan : List (Nat × Nat)) : Nat :=
  readingPlan.map (fun (rate, days) => rate * days) |>.sum

theorem reading_difference : 
  let gregPages := totalPagesRead [(18, 7), (22, 14)]
  let bradPages := totalPagesRead [(26, 5), (20, 12)]
  let emilyPages := totalPagesRead [(15, 3), (24, 7), (18, 7)]
  gregPages + bradPages - emilyPages = 465 := by
  sorry

#eval totalPagesRead [(18, 7), (22, 14)] -- Greg's pages
#eval totalPagesRead [(26, 5), (20, 12)] -- Brad's pages
#eval totalPagesRead [(15, 3), (24, 7), (18, 7)] -- Emily's pages

end reading_difference_l2476_247631


namespace quadratic_inequality_solution_set_l2476_247601

theorem quadratic_inequality_solution_set (a b : ℝ) : 
  (∀ x, ax^2 + bx + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → 
  a = -12 ∧ b = -2 := by
sorry

end quadratic_inequality_solution_set_l2476_247601


namespace relay_team_permutations_l2476_247658

theorem relay_team_permutations (n : ℕ) (k : ℕ) :
  n = 5 → k = 3 → Nat.factorial k = 6 := by
  sorry

end relay_team_permutations_l2476_247658


namespace tom_crab_price_l2476_247624

/-- A crab seller's weekly income and catch details -/
structure CrabSeller where
  buckets : ℕ
  crabs_per_bucket : ℕ
  days_per_week : ℕ
  weekly_income : ℕ

/-- Calculate the price per crab for a crab seller -/
def price_per_crab (seller : CrabSeller) : ℚ :=
  seller.weekly_income / (seller.buckets * seller.crabs_per_bucket * seller.days_per_week)

/-- Tom's crab selling business -/
def tom : CrabSeller :=
  { buckets := 8
    crabs_per_bucket := 12
    days_per_week := 7
    weekly_income := 3360 }

/-- Theorem stating that Tom sells each crab for $5 -/
theorem tom_crab_price : price_per_crab tom = 5 := by
  sorry


end tom_crab_price_l2476_247624


namespace gcd_n_cube_minus_27_and_n_minus_3_l2476_247697

theorem gcd_n_cube_minus_27_and_n_minus_3 (n : ℕ) (h : n > 3^2) :
  Nat.gcd (n^3 - 27) (n - 3) = n - 3 := by
  sorry

end gcd_n_cube_minus_27_and_n_minus_3_l2476_247697


namespace abc_inequalities_l2476_247676

theorem abc_inequalities (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  (a * b + b * c + a * c ≤ 1/3) ∧ (1/a + 1/b + 1/c ≥ 9) := by
  sorry

end abc_inequalities_l2476_247676


namespace unique_pell_solution_l2476_247653

def isPellSolution (x y : ℕ+) : Prop :=
  (x : ℤ)^2 - 2003 * (y : ℤ)^2 = 1

def isFundamentalSolution (x₀ y₀ : ℕ+) : Prop :=
  isPellSolution x₀ y₀ ∧ ∀ x y : ℕ+, isPellSolution x y → x₀ ≤ x ∧ y₀ ≤ y

def allPrimeFactorsDivide (x x₀ : ℕ+) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ x → p ∣ x₀

theorem unique_pell_solution (x₀ y₀ x y : ℕ+) :
  isFundamentalSolution x₀ y₀ →
  isPellSolution x y →
  allPrimeFactorsDivide x x₀ →
  x = x₀ ∧ y = y₀ := by
  sorry

end unique_pell_solution_l2476_247653


namespace best_approximation_l2476_247641

def f (x : ℝ) := x^2 + 2*x

def table_values : List ℝ := [1.63, 1.64, 1.65, 1.66]

def target_value : ℝ := 6

theorem best_approximation :
  ∀ x ∈ table_values, 
    abs (f 1.65 - target_value) ≤ abs (f x - target_value) ∧
    (∀ y ∈ table_values, abs (f y - target_value) < abs (f 1.65 - target_value) → y = 1.65) :=
by sorry

end best_approximation_l2476_247641


namespace sunflower_majority_on_tuesday_l2476_247634

/-- Represents the state of the bird feeder on a given day -/
structure FeederState where
  day : Nat
  sunflower_seeds : Real
  other_seeds : Real

/-- Calculates the next day's feeder state -/
def next_day_state (state : FeederState) : FeederState :=
  { day := state.day + 1,
    sunflower_seeds := state.sunflower_seeds * 0.7 + 0.2,
    other_seeds := state.other_seeds * 0.4 + 0.3 }

/-- Initial state of the feeder on Sunday -/
def initial_state : FeederState :=
  { day := 1,
    sunflower_seeds := 0.4,
    other_seeds := 0.6 }

/-- Theorem stating that on Day 3 (Tuesday), sunflower seeds make up more than half of the total seeds -/
theorem sunflower_majority_on_tuesday :
  let state₃ := next_day_state (next_day_state initial_state)
  state₃.sunflower_seeds > (state₃.sunflower_seeds + state₃.other_seeds) / 2 := by
  sorry


end sunflower_majority_on_tuesday_l2476_247634


namespace total_book_pairs_l2476_247620

/-- Represents the number of books in each genre -/
def books_per_genre : ℕ := 4

/-- Represents the number of genres -/
def num_genres : ℕ := 3

/-- Calculates the number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The main theorem stating the total number of possible book pairs -/
theorem total_book_pairs : 
  (choose_two num_genres * books_per_genre * books_per_genre) + 
  (choose_two books_per_genre) = 54 := by sorry

end total_book_pairs_l2476_247620


namespace roots_of_varying_signs_l2476_247663

theorem roots_of_varying_signs :
  (∃ x y : ℝ, x * y < 0 ∧ 4 * x^2 - 8 = 40 ∧ 4 * y^2 - 8 = 40) ∧
  (∃ x y : ℝ, x * y < 0 ∧ (3*x-2)^2 = (x+2)^2 ∧ (3*y-2)^2 = (y+2)^2) ∧
  (∃ x y : ℝ, x * y < 0 ∧ x^3 - 8*x^2 + 13*x + 10 = 0 ∧ y^3 - 8*y^2 + 13*y + 10 = 0) :=
by
  sorry


end roots_of_varying_signs_l2476_247663


namespace state_fraction_l2476_247699

theorem state_fraction (total_states : ℕ) (period_states : ℕ) 
  (h1 : total_states = 22) (h2 : period_states = 12) : 
  (period_states : ℚ) / total_states = 6 / 11 := by
  sorry

end state_fraction_l2476_247699


namespace train_length_train_length_proof_l2476_247615

/-- The length of a train given its speed, a man's speed in the opposite direction, and the time it takes to pass the man. -/
theorem train_length (train_speed : ℝ) (man_speed : ℝ) (passing_time : ℝ) : ℝ :=
  let relative_speed := train_speed + man_speed
  let relative_speed_ms := relative_speed * (5 / 18)
  relative_speed_ms * passing_time

/-- Proof that a train with speed 60 km/hr passing a man running at 6 km/hr in the opposite direction
    in approximately 29.997600191984645 seconds has a length of approximately 550 meters. -/
theorem train_length_proof : 
  ∃ ε > 0, |train_length 60 6 29.997600191984645 - 550| < ε :=
sorry

end train_length_train_length_proof_l2476_247615


namespace expand_polynomial_l2476_247670

theorem expand_polynomial (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end expand_polynomial_l2476_247670


namespace total_dress_designs_l2476_247665

/-- The number of available fabric colors -/
def num_colors : ℕ := 5

/-- The number of available patterns -/
def num_patterns : ℕ := 6

/-- The number of available sizes -/
def num_sizes : ℕ := 3

/-- Theorem stating the total number of possible dress designs -/
theorem total_dress_designs : num_colors * num_patterns * num_sizes = 90 := by
  sorry

end total_dress_designs_l2476_247665


namespace composite_sum_of_prime_powers_l2476_247686

theorem composite_sum_of_prime_powers (p q t : Nat) : 
  Prime p → Prime q → Prime t → p ≠ q → p ≠ t → q ≠ t →
  ∃ n : Nat, n > 1 ∧ n ∣ (2016^p + 2017^q + 2018^t) :=
by sorry

end composite_sum_of_prime_powers_l2476_247686


namespace least_value_quadratic_l2476_247679

theorem least_value_quadratic (y : ℝ) : 
  (2 * y^2 + 7 * y + 3 = 5) → y ≥ -2 :=
by sorry

end least_value_quadratic_l2476_247679


namespace smallest_t_for_70_degrees_l2476_247614

-- Define the temperature function
def T (t : ℝ) : ℝ := -t^2 + 10*t + 60

-- Define the atmospheric pressure function (not used in the proof, but included for completeness)
def P (t : ℝ) : ℝ := 800 - 2*t

-- Theorem statement
theorem smallest_t_for_70_degrees :
  ∃ (t : ℝ), t > 0 ∧ T t = 70 ∧ ∀ (s : ℝ), s > 0 ∧ T s = 70 → t ≤ s :=
by
  -- The proof would go here
  sorry

end smallest_t_for_70_degrees_l2476_247614


namespace range_of_f_l2476_247680

-- Define the function
def f (x : ℝ) : ℝ := |x - 3| - |x + 5|

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Icc (-8) 8 :=
sorry

end range_of_f_l2476_247680


namespace direct_proportionality_from_equation_l2476_247617

/-- Two real numbers are directly proportional if their ratio is constant -/
def DirectlyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ y = k * x

/-- Given A and B are non-zero real numbers satisfying 3A = 4B, 
    prove that A and B are directly proportional -/
theorem direct_proportionality_from_equation (A B : ℝ) 
    (h1 : 3 * A = 4 * B) (h2 : A ≠ 0) (h3 : B ≠ 0) : 
    DirectlyProportional A B := by
  sorry

end direct_proportionality_from_equation_l2476_247617


namespace complex_magnitude_example_l2476_247655

theorem complex_magnitude_example : Complex.abs (-3 - (5/4)*Complex.I) = 13/4 := by
  sorry

end complex_magnitude_example_l2476_247655


namespace inscribed_circle_radius_l2476_247635

/-- The radius of the inscribed circle of a triangle with sides 6, 8, and 10 is 2 -/
theorem inscribed_circle_radius (a b c : ℝ) (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = 2 := by sorry

end inscribed_circle_radius_l2476_247635


namespace cookie_sheet_perimeter_is_24_l2476_247605

/-- The perimeter of a rectangular cookie sheet -/
def cookie_sheet_perimeter (width length : ℝ) : ℝ :=
  2 * width + 2 * length

/-- Theorem: The perimeter of a rectangular cookie sheet with width 10 inches and length 2 inches is 24 inches -/
theorem cookie_sheet_perimeter_is_24 :
  cookie_sheet_perimeter 10 2 = 24 := by
  sorry

end cookie_sheet_perimeter_is_24_l2476_247605


namespace unique_coprime_pair_l2476_247628

def divides (a b : ℕ) : Prop := ∃ k, b = a * k

theorem unique_coprime_pair :
  ∀ a b : ℕ,
    a > 0 ∧ b > 0 →
    a < b →
    (∀ n : ℕ, n > 0 → divides b ((n+2)*a^(n+1002) - (n+1)*a^(n+1001) - n*a^(n+1000))) →
    (∀ d : ℕ, d > 1 → (divides d a ∧ divides d b) → d = 1) →
    a = 3 ∧ b = 5 :=
by sorry

end unique_coprime_pair_l2476_247628


namespace difference_2050th_2060th_term_l2476_247678

def arithmetic_sequence (a₁ a₂ : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1) * (a₂ - a₁)

theorem difference_2050th_2060th_term : 
  let a₁ := 2
  let a₂ := 9
  |arithmetic_sequence a₁ a₂ 2060 - arithmetic_sequence a₁ a₂ 2050| = 70 := by
  sorry

end difference_2050th_2060th_term_l2476_247678


namespace quadratic_properties_l2476_247608

-- Define the quadratic function
def y (m x : ℝ) : ℝ := 2*m*x^2 + (1-m)*x - 1 - m

-- Theorem statement
theorem quadratic_properties :
  -- 1. When m = -1, the vertex of the graph is at (1/2, 1/2)
  (y (-1) (1/2) = 1/2) ∧
  -- 2. When m > 0, the length of the segment intercepted by the graph on the x-axis is greater than 3/2
  (∀ m > 0, ∃ x₁ x₂, y m x₁ = 0 ∧ y m x₂ = 0 ∧ |x₁ - x₂| > 3/2) ∧
  -- 3. When m ≠ 0, the graph always passes through the fixed points (1, 0) and (-1/2, -3/2)
  (∀ m ≠ 0, y m 1 = 0 ∧ y m (-1/2) = -3/2) :=
by sorry

end quadratic_properties_l2476_247608


namespace binomial_coefficient_problem_l2476_247616

theorem binomial_coefficient_problem (h1 : Nat.choose 20 12 = 125970)
                                     (h2 : Nat.choose 18 12 = 18564)
                                     (h3 : Nat.choose 19 12 = 50388) :
  Nat.choose 20 13 = 125970 := by
  sorry

end binomial_coefficient_problem_l2476_247616


namespace number_of_tests_l2476_247646

theorem number_of_tests (n : ℕ) (S : ℝ) : 
  (S + 97) / n = 90 → 
  (S + 73) / n = 87 → 
  n = 8 := by
sorry

end number_of_tests_l2476_247646


namespace second_to_first_ratio_l2476_247689

/-- Represents the amount of food eaten by each guinea pig -/
structure GuineaPigFood where
  first : ℚ
  second : ℚ
  third : ℚ

/-- Calculates the total food eaten by all guinea pigs -/
def totalFood (gpf : GuineaPigFood) : ℚ :=
  gpf.first + gpf.second + gpf.third

/-- Theorem: The ratio of food eaten by the second guinea pig to the first guinea pig is 2:1 -/
theorem second_to_first_ratio (gpf : GuineaPigFood) : 
  gpf.first = 2 → 
  gpf.third = gpf.second + 3 → 
  totalFood gpf = 13 → 
  gpf.second / gpf.first = 2 := by
sorry

end second_to_first_ratio_l2476_247689


namespace crackers_per_box_l2476_247673

theorem crackers_per_box (darren_boxes calvin_boxes total_crackers : ℕ) : 
  darren_boxes = 4 →
  calvin_boxes = 2 * darren_boxes - 1 →
  total_crackers = 264 →
  (darren_boxes + calvin_boxes) * (total_crackers / (darren_boxes + calvin_boxes)) = total_crackers →
  total_crackers / (darren_boxes + calvin_boxes) = 24 := by
sorry

end crackers_per_box_l2476_247673


namespace polyhedron_volume_l2476_247619

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the polyhedron formed by cutting a regular quadrangular prism -/
structure Polyhedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A1 : Point3D
  C1 : Point3D
  D1 : Point3D
  O : Point3D  -- Center of the base

/-- The volume of the polyhedron -/
def volume (p : Polyhedron) : ℝ := sorry

/-- The dihedral angle between two planes -/
def dihedralAngle (plane1 plane2 : Set Point3D) : ℝ := sorry

/-- The distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Main theorem stating the volume of the polyhedron -/
theorem polyhedron_volume (p : Polyhedron) :
  (distance p.A p.B = 1) →  -- AB = 1
  (distance p.A p.A1 = distance p.O p.C1) →  -- AA₁ = OC₁
  (dihedralAngle {p.A, p.B, p.C, p.D} {p.A1, p.B, p.C1, p.D1} = π/4) →  -- 45° dihedral angle
  (volume p = Real.sqrt 2 / 2) := by
  sorry

end polyhedron_volume_l2476_247619


namespace decimal_difference_value_l2476_247664

/-- The value of the repeating decimal 0.727272... -/
def repeating_decimal : ℚ := 8 / 11

/-- The value of the terminating decimal 0.72 -/
def terminating_decimal : ℚ := 72 / 100

/-- The difference between the repeating decimal 0.727272... and the terminating decimal 0.72 -/
def decimal_difference : ℚ := repeating_decimal - terminating_decimal

theorem decimal_difference_value : decimal_difference = 8 / 1100 := by
  sorry

end decimal_difference_value_l2476_247664


namespace quadratic_equation_solution_l2476_247666

theorem quadratic_equation_solution (w : ℝ) :
  (w + 15)^2 = (4*w + 9) * (3*w + 6) →
  w^2 = (((-21 + Real.sqrt 7965) / 22)^2) ∨ w^2 = (((-21 - Real.sqrt 7965) / 22)^2) :=
by sorry

end quadratic_equation_solution_l2476_247666


namespace curve_is_rhombus_not_square_l2476_247649

-- Define the curve equation
def curve_equation (x y : ℝ) : Prop :=
  (|x + y| / 2) + |x - y| = 1

-- Define a rhombus
def is_rhombus (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧
  S = {(x, y) | |x| / a + |y| / b = 1}

-- Define the set of points satisfying the curve equation
def curve_set : Set (ℝ × ℝ) :=
  {(x, y) | curve_equation x y}

-- Theorem statement
theorem curve_is_rhombus_not_square :
  is_rhombus curve_set ∧ ¬(∃ (a : ℝ), curve_set = {(x, y) | |x| / a + |y| / a = 1}) :=
sorry

end curve_is_rhombus_not_square_l2476_247649


namespace shifted_quadratic_coefficient_sum_l2476_247618

/-- 
Given a quadratic function f(x) = 3x^2 + 2x + 5, when shifted 7 units to the right,
it results in a new quadratic function g(x) = ax^2 + bx + c.
This theorem proves that the sum of the coefficients a + b + c equals 101.
-/
theorem shifted_quadratic_coefficient_sum :
  ∀ (a b c : ℝ),
  (∀ x, (3 * (x - 7)^2 + 2 * (x - 7) + 5) = (a * x^2 + b * x + c)) →
  a + b + c = 101 := by
sorry

end shifted_quadratic_coefficient_sum_l2476_247618


namespace probability_at_least_one_girl_l2476_247637

def committee_size : ℕ := 7
def num_boys : ℕ := 4
def num_girls : ℕ := 3
def num_selected : ℕ := 2

theorem probability_at_least_one_girl :
  let total_combinations := Nat.choose committee_size num_selected
  let combinations_with_no_girls := Nat.choose num_boys num_selected
  let favorable_combinations := total_combinations - combinations_with_no_girls
  (favorable_combinations : ℚ) / total_combinations = 5 / 7 := by
  sorry

end probability_at_least_one_girl_l2476_247637


namespace iris_mall_spending_l2476_247685

/-- The total amount spent by Iris at the mall --/
def total_spent (jacket_price shorts_price pants_price : ℕ) 
                (jacket_count shorts_count pants_count : ℕ) : ℕ :=
  jacket_price * jacket_count + shorts_price * shorts_count + pants_price * pants_count

/-- Theorem stating that Iris spent $90 at the mall --/
theorem iris_mall_spending : 
  total_spent 10 6 12 3 2 4 = 90 := by
  sorry

end iris_mall_spending_l2476_247685


namespace first_day_student_tickets_l2476_247657

/-- The number of student tickets sold on the first day -/
def student_tickets_day1 : ℕ := 3

/-- The price of a student ticket -/
def student_ticket_price : ℕ := 9

/-- The price of a senior citizen ticket -/
def senior_ticket_price : ℕ := 13

theorem first_day_student_tickets :
  student_tickets_day1 = 3 ∧
  4 * senior_ticket_price + student_tickets_day1 * student_ticket_price = 79 ∧
  12 * senior_ticket_price + 10 * student_ticket_price = 246 ∧
  student_ticket_price = 9 := by
sorry

end first_day_student_tickets_l2476_247657


namespace complex_magnitude_problem_l2476_247660

theorem complex_magnitude_problem (z : ℂ) (h : Complex.I * (2 - z) = 3 + Complex.I) : 
  Complex.abs z = Real.sqrt 10 := by
  sorry

end complex_magnitude_problem_l2476_247660


namespace sandwich_combinations_l2476_247675

theorem sandwich_combinations (meat : ℕ) (cheese : ℕ) (bread : ℕ) :
  meat = 12 → cheese = 11 → bread = 5 →
  (meat * (cheese.choose 3) * bread) = 9900 :=
by
  sorry

end sandwich_combinations_l2476_247675


namespace periodic_odd_function_value_l2476_247630

def periodic_odd_function (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x, f (x + 4) = f x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 1 → f x = x)

theorem periodic_odd_function_value (f : ℝ → ℝ) 
  (h : periodic_odd_function f) : f 7.5 = -0.5 := by
  sorry

end periodic_odd_function_value_l2476_247630


namespace saras_savings_jar_l2476_247690

theorem saras_savings_jar (total_amount : ℕ) (total_bills : ℕ) 
  (h1 : total_amount = 84)
  (h2 : total_bills = 58) : 
  ∃ (ones twos : ℕ), 
    ones + twos = total_bills ∧ 
    ones + 2 * twos = total_amount ∧
    ones = 32 := by
  sorry

end saras_savings_jar_l2476_247690


namespace arithmetic_sequence_middle_term_l2476_247698

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_middle_term
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 1 + a 5 = 16) :
  a 3 = 8 := by
  sorry

end arithmetic_sequence_middle_term_l2476_247698


namespace problem_1_problem_2_l2476_247625

-- Problem 1
theorem problem_1 (a : ℝ) : (-a^2)^3 + (-2*a^3)^2 - a^3 * a^2 = 3*a^6 - a^5 := by
  sorry

-- Problem 2
theorem problem_2 (x y : ℝ) : ((x + 2*y) * (x - 2*y) + 4*(x - y)^2) + 6*x = 5*x^2 - 8*x*y + 6*x := by
  sorry

end problem_1_problem_2_l2476_247625


namespace fifty_third_number_is_53_l2476_247696

/-- Represents the sequence of numbers spoken in the modified counting game -/
def modifiedCountingSequence : ℕ → ℕ
| 0 => 1  -- Jo starts with 1
| n + 1 => 
  let prevNum := modifiedCountingSequence n
  if prevNum % 3 = 0 then prevNum + 2  -- Skip a number after multiples of 3
  else prevNum + 1  -- Otherwise, increment by 1

/-- The 53rd number in the modified counting sequence is 53 -/
theorem fifty_third_number_is_53 : modifiedCountingSequence 52 = 53 := by
  sorry

#eval modifiedCountingSequence 52  -- Evaluates to 53

end fifty_third_number_is_53_l2476_247696


namespace divides_two_pow_minus_one_l2476_247603

theorem divides_two_pow_minus_one (n : ℕ) : n > 0 → (n ∣ 2^n - 1) ↔ n = 1 := by
  sorry

end divides_two_pow_minus_one_l2476_247603


namespace custom_mul_two_neg_three_l2476_247656

-- Define the custom multiplication operation
def custom_mul (a b : ℤ) : ℤ := a^2 + a*b - b^2

-- Theorem statement
theorem custom_mul_two_neg_three :
  custom_mul 2 (-3) = -11 := by
  sorry

end custom_mul_two_neg_three_l2476_247656


namespace train_speed_conversion_l2476_247639

/-- Conversion factor from kilometers to meters -/
def km_to_m : ℝ := 1000

/-- Conversion factor from hours to seconds -/
def h_to_s : ℝ := 3600

/-- Speed of the train in km/h -/
def train_speed_kmh : ℝ := 162

/-- Theorem stating that 162 km/h is equal to 45 m/s -/
theorem train_speed_conversion :
  (train_speed_kmh * km_to_m) / h_to_s = 45 := by
  sorry

end train_speed_conversion_l2476_247639


namespace jen_birds_count_l2476_247682

/-- The number of birds Jen has given the conditions -/
def total_birds (chickens ducks geese : ℕ) : ℕ :=
  chickens + ducks + geese

/-- Theorem stating the total number of birds Jen has -/
theorem jen_birds_count :
  ∀ (chickens ducks geese : ℕ),
    ducks = 150 →
    ducks = 4 * chickens + 10 →
    geese = (ducks + chickens) / 2 →
    total_birds chickens ducks geese = 277 := by
  sorry

#check jen_birds_count

end jen_birds_count_l2476_247682


namespace income_expenditure_ratio_5_4_l2476_247607

/-- Represents the financial state of a person --/
structure FinancialState where
  income : ℕ
  savings : ℕ

/-- Calculates the expenditure given income and savings --/
def expenditure (fs : FinancialState) : ℕ :=
  fs.income - fs.savings

/-- Represents a ratio as a pair of natural numbers --/
structure Ratio where
  numerator : ℕ
  denominator : ℕ

/-- Simplifies a ratio by dividing both parts by their GCD --/
def simplifyRatio (r : Ratio) : Ratio :=
  let gcd := Nat.gcd r.numerator r.denominator
  { numerator := r.numerator / gcd,
    denominator := r.denominator / gcd }

/-- Calculates the ratio of income to expenditure --/
def incomeToExpenditureRatio (fs : FinancialState) : Ratio :=
  simplifyRatio { numerator := fs.income, denominator := expenditure fs }

theorem income_expenditure_ratio_5_4 (fs : FinancialState) 
  (h1 : fs.income = 15000) (h2 : fs.savings = 3000) : 
  incomeToExpenditureRatio fs = { numerator := 5, denominator := 4 } := by
  sorry

end income_expenditure_ratio_5_4_l2476_247607


namespace bug_meeting_point_l2476_247642

/-- Triangle PQR with side lengths PQ = 7, QR = 8, PR = 9 -/
structure Triangle :=
  (PQ : ℝ) (QR : ℝ) (PR : ℝ)
  (PQ_eq : PQ = 7)
  (QR_eq : QR = 8)
  (PR_eq : PR = 9)

/-- Point S where bugs meet -/
def S (t : Triangle) : ℝ := sorry

/-- QS is the distance from Q to S -/
def QS (t : Triangle) : ℝ := sorry

/-- Theorem stating that QS = 5 -/
theorem bug_meeting_point (t : Triangle) : QS t = 5 := by sorry

end bug_meeting_point_l2476_247642


namespace largest_prime_factor_of_4704_l2476_247609

theorem largest_prime_factor_of_4704 : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 4704 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 4704 → q ≤ p :=
by sorry

end largest_prime_factor_of_4704_l2476_247609


namespace words_with_consonants_count_l2476_247652

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'I'}
def vowels : Finset Char := {'A', 'E', 'I'}
def consonants : Finset Char := {'B', 'C', 'D'}

def word_length : Nat := 5

theorem words_with_consonants_count :
  (alphabet.card ^ word_length) - (vowels.card ^ word_length) = 7533 := by
  sorry

end words_with_consonants_count_l2476_247652


namespace cell_population_growth_l2476_247626

/-- Represents the number of cells in the population after n hours -/
def cell_count (n : ℕ) : ℕ :=
  2^(n-1) + 4

/-- The rule for cell population growth -/
def cell_growth_rule (prev : ℕ) : ℕ :=
  2 * (prev - 2)

theorem cell_population_growth (n : ℕ) :
  n > 0 →
  cell_count 1 = 5 →
  (∀ k, k ≥ 1 → cell_count (k + 1) = cell_growth_rule (cell_count k)) →
  cell_count n = 2^(n-1) + 4 :=
by
  sorry

#check cell_population_growth

end cell_population_growth_l2476_247626


namespace max_d_value_l2476_247644

def is_multiple_of_66 (n : ℕ) : Prop := n % 66 = 0

def has_form_4d645e (n : ℕ) (d e : ℕ) : Prop :=
  n = 400000 + 10000 * d + 6000 + 400 + 50 + e ∧ d < 10 ∧ e < 10

theorem max_d_value (n : ℕ) (d e : ℕ) :
  is_multiple_of_66 n → has_form_4d645e n d e → d ≤ 9 :=
by sorry

end max_d_value_l2476_247644


namespace prob_A_more_points_theorem_l2476_247612

/-- Represents a soccer tournament with given conditions -/
structure SoccerTournament where
  num_teams : Nat
  num_games_per_team : Nat
  prob_A_wins_B : ℝ
  prob_win_other_games : ℝ

/-- Calculates the probability that Team A ends up with more points than Team B -/
def prob_A_more_points_than_B (tournament : SoccerTournament) : ℝ :=
  sorry

/-- The main theorem stating the probability for Team A to end up with more points -/
theorem prob_A_more_points_theorem (tournament : SoccerTournament) :
  tournament.num_teams = 7 ∧
  tournament.num_games_per_team = 6 ∧
  tournament.prob_A_wins_B = 0.6 ∧
  tournament.prob_win_other_games = 0.5 →
  prob_A_more_points_than_B tournament = 779 / 1024 :=
  sorry

end prob_A_more_points_theorem_l2476_247612


namespace sector_arc_length_l2476_247667

/-- Given a sector with radius 2 and area 4, the length of the arc
    corresponding to the central angle is 4. -/
theorem sector_arc_length (r : ℝ) (S : ℝ) (l : ℝ) : 
  r = 2 → S = 4 → S = (1/2) * r * l → l = 4 := by sorry

end sector_arc_length_l2476_247667


namespace remainder_divisibility_l2476_247623

theorem remainder_divisibility (N : ℕ) (h : N > 0) (h1 : N % 60 = 49) : N % 15 = 4 := by
  sorry

end remainder_divisibility_l2476_247623


namespace max_students_is_25_l2476_247684

/-- A graph representing friendships in a class of students. -/
structure FriendshipGraph (n : ℕ) where
  friends : Fin n → Fin n → Prop

/-- The property that among any six students, there are two that are not friends. -/
def hasTwoNonFriends (G : FriendshipGraph n) : Prop :=
  ∀ (s : Finset (Fin n)), s.card = 6 → ∃ (i j : Fin n), i ∈ s ∧ j ∈ s ∧ i ≠ j ∧ ¬G.friends i j

/-- The property that for any pair of non-friends, there is a student among the remaining four who is friends with both. -/
def hasCommonFriend (G : FriendshipGraph n) : Prop :=
  ∀ (i j : Fin n), i ≠ j → ¬G.friends i j →
    ∃ (k : Fin n), k ≠ i ∧ k ≠ j ∧ G.friends i k ∧ G.friends j k

/-- The main theorem: The maximum number of students satisfying the given conditions is 25. -/
theorem max_students_is_25 :
  (∃ (n : ℕ) (G : FriendshipGraph n), hasTwoNonFriends G ∧ hasCommonFriend G) ∧
  (∀ (n : ℕ) (G : FriendshipGraph n), hasTwoNonFriends G → hasCommonFriend G → n ≤ 25) :=
sorry

end max_students_is_25_l2476_247684


namespace wide_tall_difference_l2476_247643

/-- Represents a cupboard for storing glasses --/
structure Cupboard where
  capacity : ℕ

/-- Represents the collection of cupboards --/
structure CupboardCollection where
  tall : Cupboard
  wide : Cupboard
  narrow : Cupboard

/-- The problem setup --/
def setup : CupboardCollection where
  tall := { capacity := 20 }
  wide := { capacity := 0 }  -- We don't know the capacity, so we set it to 0
  narrow := { capacity := 10 }  -- After breaking one shelf

/-- The theorem to prove --/
theorem wide_tall_difference (w : ℕ) : 
  w = setup.wide.capacity → w - setup.tall.capacity = w - 20 := by
  sorry

/-- The main result --/
def result : ℕ → ℕ
  | w => w - 20

#check result

end wide_tall_difference_l2476_247643


namespace point_not_in_third_quadrant_l2476_247654

-- Define the linear function
def f (x : ℝ) : ℝ := -x + 8

-- Define what it means for a point to be in the third quadrant
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

-- Theorem statement
theorem point_not_in_third_quadrant (x y : ℝ) (h : y = f x) : ¬ in_third_quadrant x y := by
  sorry

end point_not_in_third_quadrant_l2476_247654


namespace min_value_of_expression_equality_condition_l2476_247677

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*a*c)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 :=
by sorry

theorem equality_condition (a : ℝ) (ha : a > 0) :
  (a / Real.sqrt (a^2 + 8*a*a)) + (a / Real.sqrt (a^2 + 8*a*a)) + (a / Real.sqrt (a^2 + 8*a*a)) = 1 :=
by sorry

end min_value_of_expression_equality_condition_l2476_247677


namespace max_triangle_area_l2476_247659

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 6*x

-- Define points A and B on the parabola
def pointA (x₁ y₁ : ℝ) : Prop := parabola x₁ y₁
def pointB (x₂ y₂ : ℝ) : Prop := parabola x₂ y₂

-- Define the conditions
def conditions (x₁ x₂ : ℝ) : Prop := x₁ ≠ x₂ ∧ x₁ + x₂ = 4

-- Define the perpendicular bisector intersection with x-axis
def pointC (x₁ y₁ x₂ y₂ : ℝ) : ℝ × ℝ := sorry

-- Define the area of triangle ABC
def triangleArea (x₁ y₁ x₂ y₂ : ℝ) : ℝ := sorry

-- Theorem statement
theorem max_triangle_area 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (hA : pointA x₁ y₁) 
  (hB : pointB x₂ y₂) 
  (hC : conditions x₁ x₂) :
  ∃ (max_area : ℝ), 
    (∀ (x₁' y₁' x₂' y₂' : ℝ), 
      pointA x₁' y₁' → pointB x₂' y₂' → conditions x₁' x₂' →
      triangleArea x₁' y₁' x₂' y₂' ≤ max_area) ∧
    max_area = (14/3) * Real.sqrt 7 :=
sorry

end max_triangle_area_l2476_247659


namespace circle_ratio_after_increase_l2476_247629

theorem circle_ratio_after_increase (r : ℝ) (h : r > 0) : 
  (2 * π * (r + 2)) / (2 * (r + 2)) = π := by
  sorry

end circle_ratio_after_increase_l2476_247629


namespace tetrahedron_distance_sum_l2476_247687

/-- Tetrahedron with face areas, distances, and volume -/
structure Tetrahedron where
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  S₄ : ℝ
  H₁ : ℝ
  H₂ : ℝ
  H₃ : ℝ
  H₄ : ℝ
  V : ℝ

/-- The theorem about the sum of weighted distances in a tetrahedron -/
theorem tetrahedron_distance_sum (t : Tetrahedron) (k : ℝ) 
    (h₁ : t.S₁ / 1 = k)
    (h₂ : t.S₂ / 2 = k)
    (h₃ : t.S₃ / 3 = k)
    (h₄ : t.S₄ / 4 = k) :
  1 * t.H₁ + 2 * t.H₂ + 3 * t.H₃ + 4 * t.H₄ = 3 * t.V / k := by
  sorry

end tetrahedron_distance_sum_l2476_247687


namespace distance_to_x_axis_l2476_247604

def point_P : ℝ × ℝ := (5, -12)

theorem distance_to_x_axis :
  ‖point_P.2‖ = 12 := by sorry

end distance_to_x_axis_l2476_247604


namespace flour_remaining_l2476_247681

theorem flour_remaining (initial_amount : ℝ) : 
  let first_removal_percent : ℝ := 60
  let second_removal_percent : ℝ := 25
  let remaining_after_first : ℝ := initial_amount * (100 - first_removal_percent) / 100
  let final_remaining : ℝ := remaining_after_first * (100 - second_removal_percent) / 100
  final_remaining = initial_amount * 30 / 100 := by sorry

end flour_remaining_l2476_247681
