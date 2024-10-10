import Mathlib

namespace tire_circumference_l3482_348228

/-- Given a tire rotating at 400 revolutions per minute on a car traveling at 144 km/h, 
    the circumference of the tire is 6 meters. -/
theorem tire_circumference (revolutions_per_minute : ℝ) (speed_km_per_hour : ℝ) 
  (h1 : revolutions_per_minute = 400) 
  (h2 : speed_km_per_hour = 144) : 
  let speed_m_per_minute : ℝ := speed_km_per_hour * 1000 / 60
  let circumference : ℝ := speed_m_per_minute / revolutions_per_minute
  circumference = 6 := by
sorry

end tire_circumference_l3482_348228


namespace hyperbola_asymptote_m_l3482_348271

/-- Given a hyperbola with equation y² + x²/m = 1 and asymptote y = ±(√3/3)x, prove that m = -3 -/
theorem hyperbola_asymptote_m (m : ℝ) : 
  (∀ x y : ℝ, y^2 + x^2/m = 1 → (y = (Real.sqrt 3)/3 * x ∨ y = -(Real.sqrt 3)/3 * x)) → 
  m = -3 := by sorry

end hyperbola_asymptote_m_l3482_348271


namespace two_std_dev_below_mean_l3482_348223

/-- A normal distribution with given mean and standard deviation -/
structure NormalDistribution where
  mean : ℝ
  std_dev : ℝ
  std_dev_pos : std_dev > 0

/-- The value that is exactly n standard deviations less than the mean -/
def value_n_std_dev_below (d : NormalDistribution) (n : ℝ) : ℝ :=
  d.mean - n * d.std_dev

/-- Theorem: For a normal distribution with mean 15 and standard deviation 1.5,
    the value that is exactly 2 standard deviations less than the mean is 12 -/
theorem two_std_dev_below_mean (d : NormalDistribution) 
    (h1 : d.mean = 15) (h2 : d.std_dev = 1.5) : 
    value_n_std_dev_below d 2 = 12 := by
  sorry

end two_std_dev_below_mean_l3482_348223


namespace real_roots_quadratic_l3482_348283

theorem real_roots_quadratic (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x - 1 = 0) ↔ k ≥ -1 := by
  sorry

end real_roots_quadratic_l3482_348283


namespace coloring_book_shelves_l3482_348239

/-- Given a store with coloring books, prove the number of shelves used -/
theorem coloring_book_shelves 
  (initial_stock : ℕ) 
  (books_sold : ℕ) 
  (books_per_shelf : ℕ) 
  (h1 : initial_stock = 27)
  (h2 : books_sold = 6)
  (h3 : books_per_shelf = 7)
  : (initial_stock - books_sold) / books_per_shelf = 3 := by
  sorry

end coloring_book_shelves_l3482_348239


namespace prime_value_of_cubic_polynomial_l3482_348226

theorem prime_value_of_cubic_polynomial (n : ℕ) (a : ℚ) (b : ℕ) :
  b = n^3 - 4*a*n^2 - 12*n + 144 →
  Nat.Prime b →
  b = 11 := by
  sorry

end prime_value_of_cubic_polynomial_l3482_348226


namespace sum_greater_than_product_iff_one_l3482_348253

theorem sum_greater_than_product_iff_one (m n : ℕ+) :
  m + n > m * n ↔ m = 1 ∨ n = 1 := by sorry

end sum_greater_than_product_iff_one_l3482_348253


namespace yi_rong_ferry_distance_l3482_348294

/-- The Yi Rong ferry problem -/
theorem yi_rong_ferry_distance :
  let ferry_speed : ℝ := 40
  let water_speed : ℝ := 24
  let downstream_speed : ℝ := ferry_speed + water_speed
  let upstream_speed : ℝ := ferry_speed - water_speed
  let distance : ℝ := 192  -- The distance we want to prove

  -- Odd day condition
  (distance / downstream_speed * (43 / 18) = 
   distance / 2 / downstream_speed + distance / 2 / water_speed) ∧ 
  
  -- Even day condition
  (distance / upstream_speed = 
   distance / 2 / water_speed + 1 + distance / 2 / (2 * upstream_speed)) →
  
  distance = 192 := by sorry

end yi_rong_ferry_distance_l3482_348294


namespace count_valid_pairs_l3482_348229

def harmonic_mean (x y : ℕ+) : ℚ := 2 * (x * y) / (x + y)

def valid_pair (x y : ℕ+) : Prop :=
  x < y ∧ harmonic_mean x y = 1024

theorem count_valid_pairs : 
  ∃ (S : Finset (ℕ+ × ℕ+)), (∀ p ∈ S, valid_pair p.1 p.2) ∧ S.card = 9 ∧ 
  (∀ x y : ℕ+, valid_pair x y → (x, y) ∈ S) :=
sorry

end count_valid_pairs_l3482_348229


namespace rectangle_height_calculation_l3482_348246

/-- Represents a rectangle with a base and height in centimeters -/
structure Rectangle where
  base : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.base * r.height

theorem rectangle_height_calculation (r : Rectangle) 
  (h_base : r.base = 9)
  (h_area : area r = 33.3) :
  r.height = 3.7 := by
sorry


end rectangle_height_calculation_l3482_348246


namespace wine_drinkers_l3482_348254

theorem wine_drinkers (soda : Nat) (both : Nat) (total : Nat) (h1 : soda = 22) (h2 : both = 17) (h3 : total = 31) :
  ∃ (wine : Nat), wine + soda - both = total ∧ wine = 26 := by
  sorry

end wine_drinkers_l3482_348254


namespace trig_identity_l3482_348268

theorem trig_identity (α β : ℝ) : 
  Real.sin (2 * α) ^ 2 + Real.sin β ^ 2 + Real.cos (2 * α + β) * Real.cos (2 * α - β) = 1 := by
  sorry

end trig_identity_l3482_348268


namespace johns_weight_bench_safety_percentage_l3482_348216

/-- Proves that the percentage under the maximum weight that John wants to stay is 20% -/
theorem johns_weight_bench_safety_percentage 
  (bench_max_capacity : ℝ) 
  (johns_weight : ℝ) 
  (bar_weight : ℝ) 
  (h1 : bench_max_capacity = 1000) 
  (h2 : johns_weight = 250) 
  (h3 : bar_weight = 550) : 
  100 - (johns_weight + bar_weight) / bench_max_capacity * 100 = 20 := by
sorry

end johns_weight_bench_safety_percentage_l3482_348216


namespace sum_of_roots_quadratic_l3482_348264

theorem sum_of_roots_quadratic (x : ℝ) : x^2 = 16*x - 5 → ∃ y : ℝ, x^2 = 16*x - 5 ∧ x + y = 16 := by
  sorry

end sum_of_roots_quadratic_l3482_348264


namespace cube_root_of_four_sixth_powers_l3482_348291

theorem cube_root_of_four_sixth_powers (x : ℝ) :
  x = (4^6 + 4^6 + 4^6 + 4^6)^(1/3) → x = 16 * (4^(1/3)) :=
by sorry

end cube_root_of_four_sixth_powers_l3482_348291


namespace reciprocal_equation_solution_l3482_348231

theorem reciprocal_equation_solution (x : ℝ) : 
  3 - 1 / (4 * (1 - x)) = 2 * (1 / (4 * (1 - x))) → x = 3 / 4 := by
  sorry

end reciprocal_equation_solution_l3482_348231


namespace max_ratio_theorem_l3482_348211

theorem max_ratio_theorem :
  ∃ (A B : ℝ), 
    (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x^3 + y^4 = x^2*y → x ≤ A ∧ y ≤ B) ∧
    (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x^3 + y^4 = x^2*y ∧ x = A) ∧
    (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x^3 + y^4 = x^2*y ∧ y = B) ∧
    A/B = 729/1024 :=
by sorry

end max_ratio_theorem_l3482_348211


namespace log_equation_solution_l3482_348212

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  4 * Real.log x / Real.log 3 = Real.log (5 * x) / Real.log 3 → x = (5 : ℝ) ^ (1/3) := by
  sorry

end log_equation_solution_l3482_348212


namespace alex_movie_count_l3482_348256

theorem alex_movie_count (total_different_movies : ℕ) 
  (movies_watched_together : ℕ) 
  (dalton_movies : ℕ) 
  (hunter_movies : ℕ) 
  (h1 : total_different_movies = 30)
  (h2 : movies_watched_together = 2)
  (h3 : dalton_movies = 7)
  (h4 : hunter_movies = 12) :
  total_different_movies - movies_watched_together - dalton_movies - hunter_movies = 9 := by
  sorry

end alex_movie_count_l3482_348256


namespace intersection_M_N_l3482_348292

def M : Set ℝ := {x : ℝ | x^2 - 3*x = 0}
def N : Set ℝ := {-1, 1, 3}

theorem intersection_M_N : M ∩ N = {3} := by sorry

end intersection_M_N_l3482_348292


namespace three_positions_from_eight_people_l3482_348255

theorem three_positions_from_eight_people :
  (8 : ℕ).descFactorial 3 = 336 := by
  sorry

end three_positions_from_eight_people_l3482_348255


namespace bowling_ball_weight_l3482_348242

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 10 * b = 5 * c) 
  (h2 : 3 * c = 120) : 
  b = 20 := by sorry

end bowling_ball_weight_l3482_348242


namespace statue_cost_proof_l3482_348247

theorem statue_cost_proof (selling_price : ℝ) (profit_percentage : ℝ) (original_cost : ℝ) : 
  selling_price = 670 ∧ 
  profit_percentage = 0.25 ∧ 
  selling_price = original_cost * (1 + profit_percentage) →
  original_cost = 536 := by
sorry

end statue_cost_proof_l3482_348247


namespace sugar_to_cream_cheese_ratio_l3482_348221

/-- Represents the ingredients and ratios in Betty's cheesecake recipe -/
structure CheesecakeRecipe where
  sugar : ℕ
  cream_cheese : ℕ
  vanilla : ℕ
  eggs : ℕ
  vanilla_to_cream_cheese_ratio : vanilla * 2 = cream_cheese
  eggs_to_vanilla_ratio : eggs = vanilla * 2
  sugar_used : sugar = 2
  eggs_used : eggs = 8

/-- The ratio of sugar to cream cheese in Betty's cheesecake is 1:4 -/
theorem sugar_to_cream_cheese_ratio (recipe : CheesecakeRecipe) : 
  recipe.sugar * 4 = recipe.cream_cheese := by
  sorry

#check sugar_to_cream_cheese_ratio

end sugar_to_cream_cheese_ratio_l3482_348221


namespace systematic_sampling_interval_l3482_348299

/-- Represents a population for systematic sampling -/
structure Population where
  total : Nat
  omitted : Nat

/-- Checks if a given interval is valid for systematic sampling -/
def is_valid_interval (pop : Population) (interval : Nat) : Prop :=
  (pop.total - pop.omitted) % interval = 0

/-- The theorem to prove -/
theorem systematic_sampling_interval (pop : Population) 
  (h1 : pop.total = 102) 
  (h2 : pop.omitted = 2) : 
  is_valid_interval pop 10 := by
  sorry

end systematic_sampling_interval_l3482_348299


namespace sandy_correct_sums_l3482_348209

theorem sandy_correct_sums 
  (total_sums : ℕ) 
  (total_marks : ℤ) 
  (marks_per_correct : ℕ) 
  (marks_per_incorrect : ℕ) 
  (h1 : total_sums = 30)
  (h2 : total_marks = 55)
  (h3 : marks_per_correct = 3)
  (h4 : marks_per_incorrect = 2) :
  ∃ (correct_sums : ℕ), 
    correct_sums * marks_per_correct - (total_sums - correct_sums) * marks_per_incorrect = total_marks ∧ 
    correct_sums = 23 := by
sorry

end sandy_correct_sums_l3482_348209


namespace smallest_k_for_sum_of_squares_divisible_by_360_l3482_348261

theorem smallest_k_for_sum_of_squares_divisible_by_360 :
  ∀ k : ℕ, k > 0 → (k * (k + 1) * (2 * k + 1)) % 2160 = 0 → k ≥ 72 :=
by sorry

end smallest_k_for_sum_of_squares_divisible_by_360_l3482_348261


namespace min_concerts_required_l3482_348210

/-- Represents a concert where some musicians play and others listen -/
structure Concert where
  players : Finset (Fin 6)

/-- Checks if a set of concerts satisfies the condition that 
    for every pair of musicians, each plays for the other in some concert -/
def satisfies_condition (concerts : List Concert) : Prop :=
  ∀ i j, i ≠ j → 
    (∃ c ∈ concerts, i ∈ c.players ∧ j ∉ c.players) ∧
    (∃ c ∈ concerts, j ∈ c.players ∧ i ∉ c.players)

/-- The main theorem: the minimum number of concerts required is 4 -/
theorem min_concerts_required : 
  (∃ concerts : List Concert, concerts.length = 4 ∧ satisfies_condition concerts) ∧
  (∀ concerts : List Concert, concerts.length < 4 → ¬satisfies_condition concerts) :=
sorry

end min_concerts_required_l3482_348210


namespace sandwich_menu_count_l3482_348235

theorem sandwich_menu_count (initial_count sold_out remaining : ℕ) : 
  sold_out = 5 → remaining = 4 → initial_count = sold_out + remaining :=
by
  sorry

end sandwich_menu_count_l3482_348235


namespace tim_surprise_combinations_l3482_348278

/-- Represents the number of choices for each day of the week --/
structure WeekChoices where
  monday : Nat
  tuesday : Nat
  wednesday : Nat
  thursday : Nat
  friday : Nat

/-- Calculates the total number of combinations for Tim's surprise arrangements --/
def totalCombinations (choices : WeekChoices) : Nat :=
  choices.monday * choices.tuesday * choices.wednesday * choices.thursday * choices.friday

/-- Tim's specific choices for each day of the week --/
def timChoices : WeekChoices :=
  { monday := 1
  , tuesday := 2
  , wednesday := 6
  , thursday := 5
  , friday := 2 }

theorem tim_surprise_combinations :
  totalCombinations timChoices = 120 := by
  sorry

end tim_surprise_combinations_l3482_348278


namespace union_of_sets_l3482_348215

theorem union_of_sets (A B : Set ℕ) (m : ℕ) : 
  A = {1, 2, 4} → 
  B = {m, 4, 7} → 
  A ∩ B = {1, 4} → 
  A ∪ B = {1, 2, 4, 7} := by
sorry

end union_of_sets_l3482_348215


namespace total_ways_is_eight_l3482_348237

/-- The number of ways an individual can sign up -/
def sign_up_ways : ℕ := 2

/-- The number of individuals signing up -/
def num_individuals : ℕ := 3

/-- The total number of different ways all individuals can sign up -/
def total_ways : ℕ := sign_up_ways ^ num_individuals

/-- Theorem: The total number of different ways all individuals can sign up is 8 -/
theorem total_ways_is_eight : total_ways = 8 := by
  sorry

end total_ways_is_eight_l3482_348237


namespace joan_balloon_count_l3482_348286

/-- The number of orange balloons Joan has after receiving more from a friend -/
def total_balloons (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem: Given Joan has 8 orange balloons initially and receives 2 more from a friend,
    she now has 10 orange balloons in total. -/
theorem joan_balloon_count : total_balloons 8 2 = 10 := by
  sorry

end joan_balloon_count_l3482_348286


namespace optimal_seating_arrangement_l3482_348213

/-- Represents the seating arrangement for children based on their heights. -/
structure SeatingArrangement where
  x : ℕ  -- Number of seats with three children
  y : ℕ  -- Number of seats with two children
  total_seats : ℕ
  group_a : ℕ  -- Children below 4 feet
  group_b : ℕ  -- Children between 4 and 4.5 feet
  group_c : ℕ  -- Children above 4.5 feet

/-- The seating arrangement satisfies all constraints. -/
def valid_arrangement (s : SeatingArrangement) : Prop :=
  s.x + s.y = s.total_seats ∧
  s.x ≤ s.group_a ∧
  2 * s.x + s.y ≤ s.group_b ∧
  s.y ≤ s.group_c

/-- The optimal seating arrangement exists and is unique. -/
theorem optimal_seating_arrangement :
  ∃! s : SeatingArrangement,
    s.total_seats = 7 ∧
    s.group_a = 5 ∧
    s.group_b = 8 ∧
    s.group_c = 6 ∧
    valid_arrangement s ∧
    s.x = 1 ∧
    s.y = 6 := by
  sorry


end optimal_seating_arrangement_l3482_348213


namespace log_condition_l3482_348298

theorem log_condition (x : ℝ) : 
  (∀ x, Real.log (x + 1) < 0 → x < 0) ∧ 
  (∃ x, x < 0 ∧ Real.log (x + 1) ≥ 0) :=
sorry

end log_condition_l3482_348298


namespace ellipse_major_axis_length_l3482_348262

/-- The length of the major axis of the ellipse x^2/49 + y^2/81 = 1 is 18 -/
theorem ellipse_major_axis_length : 
  let a := Real.sqrt (max 49 81)
  2 * a = 18 := by
  sorry

end ellipse_major_axis_length_l3482_348262


namespace product_difference_sum_l3482_348227

theorem product_difference_sum (A B C D : ℕ) : 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D →
  A > 0 ∧ B > 0 ∧ C > 0 ∧ D > 0 →
  A * B = 72 →
  C * D = 72 →
  A - B = C + D →
  A = 18 := by
sorry

end product_difference_sum_l3482_348227


namespace evaluate_expression_l3482_348269

theorem evaluate_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 2) :
  4 * x^y + 5 * y^x = 76 := by
  sorry

end evaluate_expression_l3482_348269


namespace vector_projection_and_collinearity_l3482_348295

def a : Fin 3 → ℚ := ![2, 2, -1]
def b : Fin 3 → ℚ := ![-1, 4, 3]
def p : Fin 3 → ℚ := ![40/29, 64/29, 17/29]

theorem vector_projection_and_collinearity :
  (∀ i : Fin 3, (a i - p i) • (b - a) = 0) ∧
  (∀ i : Fin 3, (b i - p i) • (b - a) = 0) ∧
  ∃ t : ℚ, ∀ i : Fin 3, p i = a i + t * (b i - a i) := by
  sorry

end vector_projection_and_collinearity_l3482_348295


namespace zenith_school_reading_fraction_l3482_348238

/-- Represents the student body at Zenith Middle School -/
structure StudentBody where
  total : ℕ
  enjoy_reading : ℕ
  dislike_reading : ℕ
  enjoy_and_express : ℕ
  enjoy_but_pretend_dislike : ℕ
  dislike_and_express : ℕ
  dislike_but_pretend_enjoy : ℕ

/-- The conditions of the problem -/
def zenith_school (s : StudentBody) : Prop :=
  s.total > 0 ∧
  s.enjoy_reading = (70 * s.total) / 100 ∧
  s.dislike_reading = s.total - s.enjoy_reading ∧
  s.enjoy_and_express = (70 * s.enjoy_reading) / 100 ∧
  s.enjoy_but_pretend_dislike = s.enjoy_reading - s.enjoy_and_express ∧
  s.dislike_and_express = (75 * s.dislike_reading) / 100 ∧
  s.dislike_but_pretend_enjoy = s.dislike_reading - s.dislike_and_express

/-- The theorem to be proved -/
theorem zenith_school_reading_fraction (s : StudentBody) :
  zenith_school s →
  (s.enjoy_but_pretend_dislike : ℚ) / (s.enjoy_but_pretend_dislike + s.dislike_and_express) = 21 / 43 := by
  sorry


end zenith_school_reading_fraction_l3482_348238


namespace age_difference_is_two_l3482_348240

/-- The age difference between Jayson's dad and mom -/
def age_difference (jayson_age : ℕ) (mom_age_at_birth : ℕ) : ℕ :=
  (4 * jayson_age) - (mom_age_at_birth + jayson_age)

/-- Theorem stating the age difference between Jayson's dad and mom is 2 years -/
theorem age_difference_is_two :
  age_difference 10 28 = 2 := by
  sorry

end age_difference_is_two_l3482_348240


namespace compound_interest_calculation_l3482_348260

/-- Calculate compound interest for a fixed deposit -/
theorem compound_interest_calculation 
  (principal : ℝ) 
  (rate : ℝ) 
  (time : ℕ) 
  (h1 : principal = 50000) 
  (h2 : rate = 0.04) 
  (h3 : time = 3) : 
  (principal * (1 + rate)^time - principal) = (5 * (1 + 0.04)^3 - 5) * 10000 := by
  sorry

end compound_interest_calculation_l3482_348260


namespace dog_water_consumption_l3482_348224

/-- Calculates the water needed for a dog during a hike given the total water capacity,
    human water consumption rate, and duration of the hike. -/
theorem dog_water_consumption
  (total_water : ℝ)
  (human_rate : ℝ)
  (duration : ℝ)
  (h1 : total_water = 4.8 * 1000) -- 4.8 L converted to ml
  (h2 : human_rate = 800)
  (h3 : duration = 4) :
  (total_water - human_rate * duration) / duration = 400 :=
by sorry

end dog_water_consumption_l3482_348224


namespace maxwell_walking_speed_l3482_348208

/-- Proves that Maxwell's walking speed is 4 km/h given the problem conditions --/
theorem maxwell_walking_speed :
  ∀ (maxwell_speed : ℝ),
    maxwell_speed > 0 →
    (4 * maxwell_speed + 18 = 34) →
    maxwell_speed = 4 := by
  sorry

end maxwell_walking_speed_l3482_348208


namespace tv_conditional_probability_l3482_348263

theorem tv_conditional_probability 
  (p_10000 : ℝ) 
  (p_15000 : ℝ) 
  (h1 : p_10000 = 0.80) 
  (h2 : p_15000 = 0.60) : 
  p_15000 / p_10000 = 0.75 := by
sorry

end tv_conditional_probability_l3482_348263


namespace parabola_vertex_above_x_axis_l3482_348245

/-- A parabola with equation y = x^2 - 3x + k has its vertex above the x-axis if and only if k > 9/4 -/
theorem parabola_vertex_above_x_axis (k : ℝ) : 
  (∃ (x y : ℝ), y = x^2 - 3*x + k ∧ y > 0 ∧ ∀ (x' : ℝ), x'^2 - 3*x' + k ≤ y) ↔ k > 9/4 :=
sorry

end parabola_vertex_above_x_axis_l3482_348245


namespace number_problem_l3482_348258

theorem number_problem (x y : ℝ) (h1 : x - y = 12) (h2 : x^2 + y^2 = 320) : 
  x * y = 64 ∧ x^3 + y^3 = 4160 := by
  sorry

end number_problem_l3482_348258


namespace intersection_nonempty_iff_n_between_3_and_5_l3482_348204

/-- A simple polygon in 2D space -/
structure SimplePolygon where
  vertices : List (ℝ × ℝ)
  is_simple : Bool  -- Assume this is true for a simple polygon
  is_counterclockwise : Bool -- Assume this is true for counterclockwise orientation

/-- Represents a half-plane in 2D space -/
structure HalfPlane where
  normal : ℝ × ℝ
  offset : ℝ

/-- Function to get the positive half-planes of a simple polygon -/
def getPositiveHalfPlanes (p : SimplePolygon) : List HalfPlane :=
  sorry  -- Implementation details omitted

/-- Function to check if the intersection of half-planes is non-empty -/
def isIntersectionNonEmpty (planes : List HalfPlane) : Bool :=
  sorry  -- Implementation details omitted

/-- The main theorem -/
theorem intersection_nonempty_iff_n_between_3_and_5 (n : ℕ) :
  (∀ p : SimplePolygon, p.vertices.length = n →
    isIntersectionNonEmpty (getPositiveHalfPlanes p)) ↔ (3 ≤ n ∧ n ≤ 5) :=
  sorry

end intersection_nonempty_iff_n_between_3_and_5_l3482_348204


namespace function_inequality_implies_parameter_bound_l3482_348206

open Real

theorem function_inequality_implies_parameter_bound 
  (f g h : ℝ → ℝ)
  (hf : ∀ x, f x = 1/2 * x^2 - 2*x)
  (hg : ∀ x, g x = a * log x)
  (hh : ∀ x, h x = f x - g x)
  (h_pos : ∀ x, x > 0)
  (h_ineq : ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (h x₁ - h x₂) / (x₁ - x₂) > 2)
  : a ≤ -4 :=
sorry

end function_inequality_implies_parameter_bound_l3482_348206


namespace daniel_initial_noodles_l3482_348244

/-- The number of noodles Daniel gave away -/
def noodles_given : ℕ := 12

/-- The number of noodles Daniel has now -/
def noodles_left : ℕ := 54

/-- The initial number of noodles Daniel had -/
def initial_noodles : ℕ := noodles_given + noodles_left

theorem daniel_initial_noodles :
  initial_noodles = 66 := by sorry

end daniel_initial_noodles_l3482_348244


namespace max_value_of_f_l3482_348233

noncomputable def f (x : ℝ) : ℝ := x * Real.exp (-x)

theorem max_value_of_f :
  ∃ (c : ℝ), c ∈ Set.Icc 0 4 ∧ 
  (∀ x, x ∈ Set.Icc 0 4 → f x ≤ f c) ∧
  f c = 1 / Real.exp 1 := by
  sorry

end max_value_of_f_l3482_348233


namespace junk_mail_distribution_l3482_348243

theorem junk_mail_distribution (total_mail : ℕ) (houses : ℕ) (mail_per_house : ℕ) : 
  total_mail = 14 → houses = 7 → mail_per_house = total_mail / houses → mail_per_house = 2 := by
  sorry

end junk_mail_distribution_l3482_348243


namespace black_white_difference_l3482_348272

/-- Represents a chessboard square color -/
inductive Color
| Black
| White

/-- Represents a chessboard -/
structure Chessboard :=
  (rows : Nat)
  (cols : Nat)
  (startColor : Color)

/-- Counts the number of squares of a given color on the chessboard -/
def countSquares (board : Chessboard) (color : Color) : Nat :=
  sorry

theorem black_white_difference (board : Chessboard) :
  board.rows = 7 ∧ board.cols = 9 ∧ board.startColor = Color.Black →
  countSquares board Color.Black = countSquares board Color.White + 1 := by
  sorry

end black_white_difference_l3482_348272


namespace bus_purchase_problem_l3482_348293

/-- Represents the cost and capacity of a bus type -/
structure BusType where
  cost : ℕ
  capacity : ℕ

/-- Represents a purchasing plan -/
structure PurchasePlan where
  typeA : ℕ
  typeB : ℕ

def totalBuses : ℕ := 10

def scenario1Cost : ℕ := 380
def scenario2Cost : ℕ := 360

def maxTotalCost : ℕ := 880
def minTotalPassengers : ℕ := 5200000

theorem bus_purchase_problem 
  (typeA typeB : BusType)
  (plans : List PurchasePlan)
  (bestPlan : PurchasePlan)
  (minCost : ℕ) :
  (typeA.cost + 3 * typeB.cost = scenario1Cost) →
  (2 * typeA.cost + 2 * typeB.cost = scenario2Cost) →
  (typeA.capacity = 500000) →
  (typeB.capacity = 600000) →
  (∀ plan ∈ plans, 
    plan.typeA + plan.typeB = totalBuses ∧
    plan.typeA * typeA.cost + plan.typeB * typeB.cost ≤ maxTotalCost ∧
    plan.typeA * typeA.capacity + plan.typeB * typeB.capacity ≥ minTotalPassengers) →
  (bestPlan ∈ plans) →
  (∀ plan ∈ plans, 
    plan.typeA * typeA.cost + plan.typeB * typeB.cost ≥ 
    bestPlan.typeA * typeA.cost + bestPlan.typeB * typeB.cost) →
  (minCost = bestPlan.typeA * typeA.cost + bestPlan.typeB * typeB.cost) →
  typeA.cost = 80 ∧ 
  typeB.cost = 100 ∧
  plans = [⟨6, 4⟩, ⟨7, 3⟩, ⟨8, 2⟩] ∧
  bestPlan = ⟨8, 2⟩ ∧
  minCost = 840 := by
  sorry

end bus_purchase_problem_l3482_348293


namespace total_birds_caught_l3482_348236

def birds_caught_day : ℕ := 8

def birds_caught_night (day : ℕ) : ℕ := 2 * day

theorem total_birds_caught :
  birds_caught_day + birds_caught_night birds_caught_day = 24 :=
by sorry

end total_birds_caught_l3482_348236


namespace chinese_remainder_theorem_example_l3482_348276

theorem chinese_remainder_theorem_example :
  ∃! x : ℕ, x < 504 ∧ 
    x % 7 = 1 ∧
    x % 8 = 1 ∧
    x % 9 = 3 :=
by
  -- The proof goes here
  sorry

end chinese_remainder_theorem_example_l3482_348276


namespace box_weight_sum_l3482_348219

theorem box_weight_sum (a b c : ℝ) 
  (hab : a + b = 132)
  (hbc : b + c = 135)
  (hca : c + a = 137)
  (ha : a > 40)
  (hb : b > 40)
  (hc : c > 40) :
  a + b + c = 202 := by
  sorry

end box_weight_sum_l3482_348219


namespace triangle_side_lengths_l3482_348252

/-- A triangle with perimeter 60, two equal sides, and a difference of 21 between two sides has side lengths 27, 27, and 6. -/
theorem triangle_side_lengths :
  ∀ a b : ℝ,
  a > 0 ∧ b > 0 ∧
  2 * a + b = 60 ∧
  a - b = 21 →
  a = 27 ∧ b = 6 :=
by sorry

end triangle_side_lengths_l3482_348252


namespace doughnut_boxes_l3482_348257

theorem doughnut_boxes (total_doughnuts : ℕ) (doughnuts_per_box : ℕ) (h1 : total_doughnuts = 48) (h2 : doughnuts_per_box = 12) :
  total_doughnuts / doughnuts_per_box = 4 := by
  sorry

end doughnut_boxes_l3482_348257


namespace count_cows_l3482_348225

def group_of_animals (ducks cows : ℕ) : Prop :=
  2 * ducks + 4 * cows = 22 + 2 * (ducks + cows)

theorem count_cows : ∃ ducks : ℕ, group_of_animals ducks 11 :=
sorry

end count_cows_l3482_348225


namespace initial_solution_strength_l3482_348251

/-- Proves that the initial solution strength is 60% given the problem conditions --/
theorem initial_solution_strength 
  (initial_volume : ℝ)
  (drained_volume : ℝ)
  (replacement_strength : ℝ)
  (final_strength : ℝ)
  (h1 : initial_volume = 50)
  (h2 : drained_volume = 35)
  (h3 : replacement_strength = 40)
  (h4 : final_strength = 46)
  (h5 : initial_volume - drained_volume + drained_volume = initial_volume)
  (h6 : (initial_volume - drained_volume) * (initial_strength / 100) + 
        drained_volume * (replacement_strength / 100) = 
        initial_volume * (final_strength / 100)) :
  initial_strength = 60 := by
  sorry

#check initial_solution_strength

end initial_solution_strength_l3482_348251


namespace no_zeros_of_g_l3482_348277

open Set
open Function
open Topology

theorem no_zeros_of_g (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x ≠ 0, deriv f x + f x / x > 0) : 
  ∀ x ≠ 0, f x + 1 / x ≠ 0 := by
  sorry

end no_zeros_of_g_l3482_348277


namespace square_area_ratio_l3482_348279

theorem square_area_ratio (side_c side_d : ℝ) (h1 : side_c = 45) (h2 : side_d = 60) :
  (side_c^2) / (side_d^2) = 9 / 16 := by
  sorry

end square_area_ratio_l3482_348279


namespace min_n_for_60n_divisible_by_4_and_8_l3482_348289

theorem min_n_for_60n_divisible_by_4_and_8 : 
  ∃ (n : ℕ), n > 0 ∧ 
    (∀ (m : ℕ), m > 0 → (4 ∣ 60 * m) ∧ (8 ∣ 60 * m) → n ≤ m) ∧
    (4 ∣ 60 * n) ∧ (8 ∣ 60 * n) :=
by
  -- The proof goes here
  sorry

end min_n_for_60n_divisible_by_4_and_8_l3482_348289


namespace f_min_value_l3482_348287

/-- The function f(x) = |x + 3| + |x + 6| + |x + 8| + |x + 10| -/
def f (x : ℝ) : ℝ := |x + 3| + |x + 6| + |x + 8| + |x + 10|

/-- Theorem stating that f(x) has a minimum value of 9 at x = -8 -/
theorem f_min_value :
  (∀ x : ℝ, f x ≥ 9) ∧ f (-8) = 9 := by sorry

end f_min_value_l3482_348287


namespace both_a_and_b_must_join_at_least_one_of_a_or_b_must_join_l3482_348232

-- Define the total number of doctors
def total_doctors : ℕ := 20

-- Define the number of doctors to be chosen
def team_size : ℕ := 5

-- Define the function to calculate combinations
def combination (n k : ℕ) : ℕ := 
  if k > n then 0
  else (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem for part (1)
theorem both_a_and_b_must_join : 
  combination (total_doctors - 2) (team_size - 2) = 816 := by sorry

-- Theorem for part (2)
theorem at_least_one_of_a_or_b_must_join : 
  2 * combination (total_doctors - 2) (team_size - 1) + 
  combination (total_doctors - 2) (team_size - 2) = 5661 := by sorry

end both_a_and_b_must_join_at_least_one_of_a_or_b_must_join_l3482_348232


namespace expression_equality_l3482_348234

theorem expression_equality : 
  |Real.sqrt 3 - 2| - (1 / 2)⁻¹ - 2 * Real.sin (π / 3) = -2 * Real.sqrt 3 := by
  sorry

end expression_equality_l3482_348234


namespace max_quartets_correct_max_quartets_5x5_l3482_348200

/-- Represents a rectangle on a grid --/
structure Rectangle where
  m : ℕ
  n : ℕ

/-- Calculates the maximum number of quartets in a rectangle --/
def max_quartets (rect : Rectangle) : ℕ :=
  if rect.m % 2 = 0 ∧ rect.n % 2 = 1 then
    (rect.m * (rect.n - 1)) / 4
  else if rect.m % 2 = 1 ∧ rect.n % 2 = 0 then
    (rect.n * (rect.m - 1)) / 4
  else if rect.m % 2 = 1 ∧ rect.n % 2 = 1 then
    if (rect.n - 1) % 4 = 0 then
      (rect.m * (rect.n - 1)) / 4
    else
      (rect.m * (rect.n - 1) - 2) / 4
  else
    (rect.m * rect.n) / 4

theorem max_quartets_correct (rect : Rectangle) :
  max_quartets rect =
    if rect.m % 2 = 0 ∧ rect.n % 2 = 1 then
      (rect.m * (rect.n - 1)) / 4
    else if rect.m % 2 = 1 ∧ rect.n % 2 = 0 then
      (rect.n * (rect.m - 1)) / 4
    else if rect.m % 2 = 1 ∧ rect.n % 2 = 1 then
      if (rect.n - 1) % 4 = 0 then
        (rect.m * (rect.n - 1)) / 4
      else
        (rect.m * (rect.n - 1) - 2) / 4
    else
      (rect.m * rect.n) / 4 :=
by sorry

/-- Specific case for 5x5 square --/
def square_5x5 : Rectangle := { m := 5, n := 5 }

theorem max_quartets_5x5 :
  max_quartets square_5x5 = 5 :=
by sorry

end max_quartets_correct_max_quartets_5x5_l3482_348200


namespace cos_2alpha_plus_pi_3_l3482_348288

theorem cos_2alpha_plus_pi_3 (α : Real) 
  (h : Real.sin (π / 6 - α) - Real.cos α = 1 / 3) : 
  Real.cos (2 * α + π / 3) = 7 / 9 := by
  sorry

end cos_2alpha_plus_pi_3_l3482_348288


namespace photo_arrangements_l3482_348296

def number_of_students : ℕ := 5

def arrangements (n : ℕ) : ℕ := sorry

theorem photo_arrangements :
  arrangements number_of_students = 36 := by sorry

end photo_arrangements_l3482_348296


namespace odd_prime_product_probability_l3482_348285

/-- A standard die with six faces numbered from 1 to 6. -/
def StandardDie : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The set of odd prime numbers on a standard die. -/
def OddPrimeOnDie : Finset ℕ := {3, 5}

/-- The number of times the die is rolled. -/
def NumRolls : ℕ := 8

/-- The probability of rolling an odd prime on a single roll of a standard die. -/
def SingleRollProbability : ℚ := (OddPrimeOnDie.card : ℚ) / (StandardDie.card : ℚ)

theorem odd_prime_product_probability :
  (SingleRollProbability ^ NumRolls : ℚ) = 1 / 6561 :=
sorry

end odd_prime_product_probability_l3482_348285


namespace min_value_interval_l3482_348250

def f (x : ℝ) := 3 * x - x^3

theorem min_value_interval (a : ℝ) :
  (∃ x ∈ Set.Ioo (a^2 - 12) a, ∀ y ∈ Set.Ioo (a^2 - 12) a, f y ≥ f x) →
  a ∈ Set.Ioo (-1) 2 := by
  sorry

end min_value_interval_l3482_348250


namespace value_range_of_f_l3482_348214

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- State the theorem
theorem value_range_of_f :
  ∀ y ∈ Set.Icc (-3) 5, ∃ x ∈ Set.Icc 0 2, f x = y ∧
  ∀ x ∈ Set.Icc 0 2, f x ∈ Set.Icc (-3) 5 :=
by sorry

end value_range_of_f_l3482_348214


namespace frances_towel_weight_l3482_348220

theorem frances_towel_weight (mary_towels frances_towels : ℕ) (total_weight : ℝ) :
  mary_towels = 24 →
  mary_towels = 4 * frances_towels →
  total_weight = 60 →
  (frances_towels * (total_weight / (mary_towels + frances_towels))) * 16 = 192 :=
by sorry

end frances_towel_weight_l3482_348220


namespace yanni_paintings_l3482_348201

def painting_count : ℕ := 5

def square_feet_per_painting : List ℕ := [25, 25, 25, 80, 45]

theorem yanni_paintings :
  (painting_count = 5) ∧
  (square_feet_per_painting.length = painting_count) ∧
  (square_feet_per_painting.sum = 200) := by
  sorry

end yanni_paintings_l3482_348201


namespace jewelry_restock_cost_l3482_348217

/-- Represents the inventory and pricing information for a jewelry item -/
structure JewelryItem where
  name : String
  capacity : Nat
  current : Nat
  price : Nat
  discount1 : Nat
  discount1Threshold : Nat
  discount2 : Nat
  discount2Threshold : Nat

/-- Calculates the total cost for restocking jewelry items -/
def calculateTotalCost (items : List JewelryItem) : Rat :=
  let itemCosts := items.map (fun item =>
    let quantity := item.capacity - item.current
    let basePrice := quantity * item.price
    let discountedPrice :=
      if quantity >= item.discount2Threshold then
        basePrice * (1 - item.discount2 / 100)
      else if quantity >= item.discount1Threshold then
        basePrice * (1 - item.discount1 / 100)
      else
        basePrice
    discountedPrice)
  let subtotal := itemCosts.sum
  let shippingFee := subtotal * (2 / 100)
  subtotal + shippingFee

/-- Theorem stating that the total cost to restock the jewelry showroom is $257.04 -/
theorem jewelry_restock_cost :
  let necklaces : JewelryItem := ⟨"Necklace", 20, 8, 5, 10, 10, 15, 15⟩
  let rings : JewelryItem := ⟨"Ring", 40, 25, 8, 5, 20, 12, 30⟩
  let bangles : JewelryItem := ⟨"Bangle", 30, 17, 6, 8, 15, 18, 25⟩
  calculateTotalCost [necklaces, rings, bangles] = 257.04 := by
  sorry

end jewelry_restock_cost_l3482_348217


namespace ice_cream_volume_l3482_348205

/-- The volume of ice cream in a cone with hemisphere and cylindrical layer -/
theorem ice_cream_volume (h_cone : ℝ) (r : ℝ) (h_cylinder : ℝ) : 
  h_cone = 10 ∧ r = 3 ∧ h_cylinder = 2 →
  (1/3 * π * r^2 * h_cone) + (2/3 * π * r^3) + (π * r^2 * h_cylinder) = 66 * π := by
  sorry

end ice_cream_volume_l3482_348205


namespace faulty_token_identifiable_l3482_348259

/-- Represents the possible outcomes of a weighing --/
inductive WeighingResult
  | Equal : WeighingResult
  | LeftHeavier : WeighingResult
  | RightHeavier : WeighingResult

/-- Represents a token with a nominal value and an actual weight --/
structure Token where
  nominal_value : ℕ
  actual_weight : ℕ

/-- Represents a set of four tokens --/
def TokenSet := (Token × Token × Token × Token)

/-- Represents a weighing action on the balance scale --/
def Weighing := (List Token) → (List Token) → WeighingResult

/-- Represents a strategy for determining the faulty token --/
def Strategy := TokenSet → Weighing → Weighing → Option Token

/-- States that exactly one token in the set has an incorrect weight --/
def ExactlyOneFaulty (ts : TokenSet) : Prop := sorry

/-- States that a strategy correctly identifies the faulty token --/
def StrategyCorrect (s : Strategy) : Prop := sorry

theorem faulty_token_identifiable :
  ∃ (s : Strategy), StrategyCorrect s :=
sorry

end faulty_token_identifiable_l3482_348259


namespace volleyball_game_employees_l3482_348284

/-- Calculates the number of employees participating in a volleyball game given the number of managers, teams, and people per team. -/
def employees_participating (managers : ℕ) (teams : ℕ) (people_per_team : ℕ) : ℕ :=
  teams * people_per_team - managers

/-- Theorem stating that with 23 managers, 6 teams, and 5 people per team, there are 7 employees participating. -/
theorem volleyball_game_employees :
  employees_participating 23 6 5 = 7 := by
  sorry

end volleyball_game_employees_l3482_348284


namespace fifth_term_is_16_l3482_348297

/-- A geometric sequence with first term 1 and common ratio 2 -/
def geometric_sequence (n : ℕ) : ℝ := 1 * 2^(n - 1)

/-- The fifth term of the geometric sequence is 16 -/
theorem fifth_term_is_16 : geometric_sequence 5 = 16 := by
  sorry

end fifth_term_is_16_l3482_348297


namespace arithmetic_sequence_common_difference_range_l3482_348273

theorem arithmetic_sequence_common_difference_range (a : ℕ → ℝ) (d : ℝ) :
  (a 1 = -3) →
  (∀ n : ℕ, a (n + 1) = a n + d) →
  (∀ n : ℕ, n ≥ 5 → a n > 0) →
  d ∈ Set.Ioo (3/4) 1 := by
sorry

end arithmetic_sequence_common_difference_range_l3482_348273


namespace consecutive_integers_sum_l3482_348202

theorem consecutive_integers_sum (n : ℕ) (x : ℤ) : 
  (n > 0) → 
  (x + n - 1 = 9) → 
  (n * (2 * x + n - 1) / 2 = 24) → 
  n = 3 :=
by sorry

end consecutive_integers_sum_l3482_348202


namespace product_mod_fifteen_l3482_348203

theorem product_mod_fifteen : 59 * 67 * 78 ≡ 9 [ZMOD 15] := by sorry

end product_mod_fifteen_l3482_348203


namespace alan_bought_20_eggs_l3482_348222

/-- The number of eggs Alan bought at the market -/
def eggs : ℕ := sorry

/-- The price of each egg in dollars -/
def egg_price : ℕ := 2

/-- The number of chickens Alan bought -/
def chickens : ℕ := 6

/-- The price of each chicken in dollars -/
def chicken_price : ℕ := 8

/-- The total amount Alan spent at the market in dollars -/
def total_spent : ℕ := 88

/-- Theorem stating that Alan bought 20 eggs -/
theorem alan_bought_20_eggs : eggs = 20 := by
  sorry

end alan_bought_20_eggs_l3482_348222


namespace cut_rectangle_decreases_area_and_perimeter_l3482_348207

-- Define a rectangle
structure Rectangle where
  length : ℝ
  width : ℝ
  length_pos : length > 0
  width_pos : width > 0

-- Define the area and perimeter functions
def area (r : Rectangle) : ℝ := r.length * r.width
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

-- State the theorem
theorem cut_rectangle_decreases_area_and_perimeter 
  (R : Rectangle) 
  (S : Rectangle) 
  (h_cut : S.length ≤ R.length ∧ S.width ≤ R.width) 
  (h_proper_subset : S.length < R.length ∨ S.width < R.width) : 
  area S < area R ∧ perimeter S < perimeter R := by
  sorry

end cut_rectangle_decreases_area_and_perimeter_l3482_348207


namespace pyramid_face_area_l3482_348241

-- Define the pyramid
structure SquareBasedPyramid where
  baseEdge : ℝ
  lateralEdge : ℝ

-- Define the problem
theorem pyramid_face_area (p : SquareBasedPyramid) 
  (h_base : p.baseEdge = 8)
  (h_lateral : p.lateralEdge = 7) : 
  Real.sqrt ((4 * p.baseEdge * Real.sqrt (p.lateralEdge ^ 2 - (p.baseEdge / 2) ^ 2)) ^ 2) = 16 * Real.sqrt 33 := by
  sorry


end pyramid_face_area_l3482_348241


namespace initial_pigeons_l3482_348230

theorem initial_pigeons (initial final joined : ℕ) : 
  initial > 0 → 
  joined = 1 → 
  final = initial + joined → 
  final = 2 → 
  initial = 1 := by
sorry

end initial_pigeons_l3482_348230


namespace no_infinite_line_family_l3482_348266

theorem no_infinite_line_family :
  ¬ ∃ (k : ℕ → ℝ), 
    (∀ n, k n ≠ 0) ∧ 
    (∀ n, k (n + 1) = (1 - 1 / k n) - (1 - k n)) ∧
    (∀ n, k n * k (n + 1) ≥ 0) :=
by sorry

end no_infinite_line_family_l3482_348266


namespace max_radius_difference_l3482_348282

/-- The ellipse Γ in a 2D coordinate system -/
def Γ : Set (ℝ × ℝ) :=
  {p | p.1^2 / 2 + p.2^2 = 1}

/-- The first quadrant -/
def firstQuadrant : Set (ℝ × ℝ) :=
  {p | p.1 > 0 ∧ p.2 > 0}

/-- Point P on the ellipse Γ in the first quadrant -/
def P : (ℝ × ℝ) :=
  sorry

/-- Left focus F₁ of the ellipse Γ -/
def F₁ : (ℝ × ℝ) :=
  sorry

/-- Right focus F₂ of the ellipse Γ -/
def F₂ : (ℝ × ℝ) :=
  sorry

/-- Point Q₁ where extended PF₁ intersects Γ -/
def Q₁ : (ℝ × ℝ) :=
  sorry

/-- Point Q₂ where extended PF₂ intersects Γ -/
def Q₂ : (ℝ × ℝ) :=
  sorry

/-- Radius r₁ of the inscribed circle in triangle PF₁Q₂ -/
def r₁ : ℝ :=
  sorry

/-- Radius r₂ of the inscribed circle in triangle PF₂Q₁ -/
def r₂ : ℝ :=
  sorry

/-- Theorem stating that the maximum value of r₁ - r₂ is 1/3 -/
theorem max_radius_difference :
  P ∈ Γ ∩ firstQuadrant →
  ∃ (max : ℝ), max = (1 : ℝ) / 3 ∧ ∀ (p : ℝ × ℝ), p ∈ Γ ∩ firstQuadrant → r₁ - r₂ ≤ max :=
sorry

end max_radius_difference_l3482_348282


namespace fraction_equality_l3482_348249

theorem fraction_equality (x y : ℚ) :
  (2/5)^2 + (1/7)^2 = 25*x * ((1/3)^2 + (1/8)^2) / (73*y) →
  Real.sqrt x / Real.sqrt y = 356 / 175 := by
  sorry

end fraction_equality_l3482_348249


namespace no_thirty_degree_angle_l3482_348281

structure Cube where
  vertices : Finset (Fin 8)

def skew_lines (c : Cube) (p1 p2 p3 p4 : Fin 8) : Prop :=
  p1 ∈ c.vertices ∧ p2 ∈ c.vertices ∧ p3 ∈ c.vertices ∧ p4 ∈ c.vertices ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4

def angle_between_lines (l1 l2 : Fin 8 × Fin 8) : ℝ :=
  sorry -- Definition of angle calculation between two lines in a cube

theorem no_thirty_degree_angle (c : Cube) :
  ∀ (p1 p2 p3 p4 : Fin 8),
    skew_lines c p1 p2 p3 p4 →
    angle_between_lines (p1, p2) (p3, p4) ≠ 30 :=
sorry

end no_thirty_degree_angle_l3482_348281


namespace polynomial_sum_simplification_l3482_348275

theorem polynomial_sum_simplification (x : ℝ) : 
  (2*x^4 + 3*x^3 - 5*x^2 + 9*x - 8) + (-x^5 + x^4 - 2*x^3 + 4*x^2 - 6*x + 14) = 
  -x^5 + 3*x^4 + x^3 - x^2 + 3*x + 6 := by
  sorry

end polynomial_sum_simplification_l3482_348275


namespace dot_product_properties_l3482_348290

def angle_between (a b : ℝ × ℝ) : ℝ := sorry

theorem dot_product_properties 
  (a b : ℝ × ℝ) 
  (h1 : Real.sqrt (a.1^2 + a.2^2) = 10)
  (h2 : Real.sqrt (b.1^2 + b.2^2) = 12)
  (h3 : angle_between a b = 2 * π / 3) : 
  (a.1 * b.1 + a.2 * b.2 = -60) ∧ 
  (3 * a.1 * (1/5 * b.1) + 3 * a.2 * (1/5 * b.2) = -36) ∧
  ((3 * b.1 - 2 * a.1) * (4 * a.1 + b.1) + (3 * b.2 - 2 * a.2) * (4 * a.2 + b.2) = -968) := by
  sorry

end dot_product_properties_l3482_348290


namespace usual_price_equals_sale_price_l3482_348265

/-- Represents the laundry detergent scenario -/
structure DetergentScenario where
  loads_per_bottle : ℕ
  sale_price_per_bottle : ℚ
  cost_per_load : ℚ

/-- The usual price of a bottle of detergent is equal to the sale price -/
theorem usual_price_equals_sale_price (scenario : DetergentScenario)
  (h1 : scenario.loads_per_bottle = 80)
  (h2 : scenario.sale_price_per_bottle = 20)
  (h3 : scenario.cost_per_load = 1/4) :
  scenario.sale_price_per_bottle = scenario.loads_per_bottle * scenario.cost_per_load := by
  sorry

#check usual_price_equals_sale_price

end usual_price_equals_sale_price_l3482_348265


namespace count_integers_satisfying_inequality_l3482_348280

theorem count_integers_satisfying_inequality :
  ∃ (S : Finset ℤ), (∀ n : ℤ, n ∈ S ↔ (n - 3) * (n + 5) < 0) ∧ Finset.card S = 7 := by
  sorry

end count_integers_satisfying_inequality_l3482_348280


namespace parallel_planes_condition_l3482_348218

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation
variable (parallel : Plane → Plane → Prop)
variable (lineParallelToPlane : Line → Plane → Prop)

-- Define the subset relation for lines in planes
variable (subset : Line → Plane → Prop)

-- Define specific planes and lines
variable (α β : Plane)
variable (a b : Line)

-- State the theorem
theorem parallel_planes_condition 
  (h1 : subset a α)
  (h2 : subset b α) :
  (∀ (α β : Plane), parallel α β → lineParallelToPlane a β ∧ lineParallelToPlane b β) ∧ 
  (∃ (α β : Plane) (a b : Line), 
    subset a α ∧ 
    subset b α ∧ 
    lineParallelToPlane a β ∧ 
    lineParallelToPlane b β ∧ 
    ¬parallel α β) :=
sorry

end parallel_planes_condition_l3482_348218


namespace martin_ice_cream_cost_l3482_348267

/-- Represents the cost of ice cream scoops in dollars -/
structure IceCreamPrices where
  kiddie : ℕ
  regular : ℕ
  double : ℕ

/-- Represents the Martin family's ice cream order -/
structure MartinOrder where
  regular : ℕ
  kiddie : ℕ
  double : ℕ

/-- Calculates the total cost of the Martin family's ice cream order -/
def calculateTotalCost (prices : IceCreamPrices) (order : MartinOrder) : ℕ :=
  prices.regular * order.regular +
  prices.kiddie * order.kiddie +
  prices.double * order.double

/-- Theorem stating that the total cost for the Martin family's ice cream order is $32 -/
theorem martin_ice_cream_cost :
  ∃ (prices : IceCreamPrices) (order : MartinOrder),
    prices.kiddie = 3 ∧
    prices.regular = 4 ∧
    prices.double = 6 ∧
    order.regular = 2 ∧
    order.kiddie = 2 ∧
    order.double = 3 ∧
    calculateTotalCost prices order = 32 :=
  sorry

end martin_ice_cream_cost_l3482_348267


namespace vacation_cost_problem_l3482_348248

/-- The vacation cost problem -/
theorem vacation_cost_problem (sarah_paid derek_paid rita_paid : ℚ)
  (h_sarah : sarah_paid = 150)
  (h_derek : derek_paid = 210)
  (h_rita : rita_paid = 240)
  (s d : ℚ) :
  let total_paid := sarah_paid + derek_paid + rita_paid
  let equal_share := total_paid / 3
  let sarah_owes := equal_share - sarah_paid
  let derek_owes := equal_share - derek_paid
  s = sarah_owes ∧ d = derek_owes →
  s - d = 60 := by
sorry

end vacation_cost_problem_l3482_348248


namespace geometric_sequence_a7_l3482_348274

/-- A geometric sequence with the given properties -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r) ∧ 
  a 1 = 8 ∧
  a 4 = a 3 * a 5

theorem geometric_sequence_a7 (a : ℕ → ℝ) (h : geometric_sequence a) : 
  a 7 = 1 / 8 := by
sorry

end geometric_sequence_a7_l3482_348274


namespace conference_handshakes_eq_360_l3482_348270

/-- Represents the number of handshakes in a conference with specific groupings -/
def conference_handshakes (total : ℕ) (group_a : ℕ) (group_b1 : ℕ) (group_b2 : ℕ) : ℕ :=
  let handshakes_a_b1 := group_b1 * (group_a - group_a / 2)
  let handshakes_a_b2 := group_b2 * group_a
  let handshakes_b2 := group_b2 * (group_b2 - 1) / 2
  handshakes_a_b1 + handshakes_a_b2 + handshakes_b2

/-- The theorem stating that the number of handshakes in the given conference scenario is 360 -/
theorem conference_handshakes_eq_360 :
  conference_handshakes 40 25 5 10 = 360 := by
  sorry

end conference_handshakes_eq_360_l3482_348270
