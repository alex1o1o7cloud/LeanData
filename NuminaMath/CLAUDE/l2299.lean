import Mathlib

namespace parabola_focus_x_coord_l2299_229930

/-- A parabola defined by parametric equations -/
structure ParametricParabola where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The focus of a parabola -/
structure ParabolaFocus where
  x : ℝ
  y : ℝ

/-- Theorem: The x-coordinate of the focus of the parabola given by x = 4t² and y = 4t is 1 -/
theorem parabola_focus_x_coord (p : ParametricParabola) 
  (h1 : p.x = fun t => 4 * t^2)
  (h2 : p.y = fun t => 4 * t) : 
  ∃ f : ParabolaFocus, f.x = 1 := by
  sorry

end parabola_focus_x_coord_l2299_229930


namespace alpha_range_l2299_229934

theorem alpha_range (α : Real) :
  (Complex.exp (Complex.I * α) + 2 * Complex.I * Complex.cos α = 2 * Complex.I) ↔
  ∃ k : ℤ, α = 2 * k * Real.pi := by
sorry

end alpha_range_l2299_229934


namespace min_value_theorem_min_value_is_four_min_value_achievable_l2299_229955

theorem min_value_theorem (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 2 → 1 / m + 2 / n ≤ 1 / x + 2 / y :=
by sorry

theorem min_value_is_four (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) :
  1 / m + 2 / n ≥ 4 :=
by sorry

theorem min_value_achievable (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + y = 2 ∧ 1 / x + 2 / y = 4 :=
by sorry

end min_value_theorem_min_value_is_four_min_value_achievable_l2299_229955


namespace composition_f_equals_inverse_e_l2299_229919

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.exp x else Real.log x

theorem composition_f_equals_inverse_e : f (f (1 / Real.exp 1)) = 1 / Real.exp 1 := by
  sorry

end composition_f_equals_inverse_e_l2299_229919


namespace no_eighteen_consecutive_good_l2299_229915

/-- A natural number is "good" if it has exactly two prime divisors -/
def isGood (n : ℕ) : Prop :=
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ n.divisors = {1, p, q, n})

/-- Theorem: There do not exist 18 consecutive natural numbers that are all "good" -/
theorem no_eighteen_consecutive_good :
  ¬ ∃ k : ℕ, ∀ i : ℕ, i < 18 → isGood (k + i) := by
  sorry

end no_eighteen_consecutive_good_l2299_229915


namespace root_product_equality_l2299_229962

theorem root_product_equality (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + 1 = 0) → 
  (β^2 + p*β + 1 = 0) → 
  (γ^2 + q*γ + 1 = 0) → 
  (δ^2 + q*δ + 1 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = q^2 - p^2 := by
sorry

end root_product_equality_l2299_229962


namespace bmw_length_l2299_229990

theorem bmw_length : 
  let straight_segments : ℕ := 7
  let straight_length : ℝ := 2
  let diagonal_segments : ℕ := 2
  let diagonal_length : ℝ := Real.sqrt 2
  straight_segments * straight_length + diagonal_segments * diagonal_length = 14 + 2 * Real.sqrt 2 := by
  sorry

end bmw_length_l2299_229990


namespace inequality_proof_l2299_229924

theorem inequality_proof (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a / (b + 2*c + 3*d)) + (b / (c + 2*d + 3*a)) + (c / (d + 2*a + 3*b)) + (d / (a + 2*b + 3*c)) ≥ 2/3 := by
  sorry

end inequality_proof_l2299_229924


namespace centroid_distance_relation_l2299_229980

/-- Given a triangle ABC with centroid G and any point P in the plane, 
    prove that the sum of squared distances from P to the vertices of the triangle 
    is equal to the sum of squared distances from G to the vertices 
    plus three times the squared distance from G to P. -/
theorem centroid_distance_relation (A B C G P : ℝ × ℝ) : 
  G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) →
  (A.1 - P.1)^2 + (A.2 - P.2)^2 + 
  (B.1 - P.1)^2 + (B.2 - P.2)^2 + 
  (C.1 - P.1)^2 + (C.2 - P.2)^2 = 
  (A.1 - G.1)^2 + (A.2 - G.2)^2 + 
  (B.1 - G.1)^2 + (B.2 - G.2)^2 + 
  (C.1 - G.1)^2 + (C.2 - G.2)^2 + 
  3 * ((G.1 - P.1)^2 + (G.2 - P.2)^2) :=
by sorry


end centroid_distance_relation_l2299_229980


namespace parallelogram_adjacent_side_l2299_229935

/-- The length of the other adjacent side of a parallelogram with perimeter 16 and one side length 5 is 3. -/
theorem parallelogram_adjacent_side (perimeter : ℝ) (side_a : ℝ) (side_b : ℝ) 
  (h1 : perimeter = 16) 
  (h2 : side_a = 5) 
  (h3 : perimeter = 2 * (side_a + side_b)) : 
  side_b = 3 := by
  sorry

end parallelogram_adjacent_side_l2299_229935


namespace smallest_of_seven_consecutive_evens_l2299_229927

/-- Given a sequence of seven consecutive even integers with a sum of 700,
    the smallest number in the sequence is 94. -/
theorem smallest_of_seven_consecutive_evens (seq : List ℤ) : 
  seq.length = 7 ∧ 
  (∀ i ∈ seq, ∃ k : ℤ, i = 2 * k) ∧ 
  (∀ i j, i ∈ seq → j ∈ seq → i ≠ j → (i - j).natAbs = 2) ∧
  seq.sum = 700 →
  seq.minimum? = some 94 := by
sorry

end smallest_of_seven_consecutive_evens_l2299_229927


namespace book_price_increase_l2299_229938

theorem book_price_increase (new_price : ℝ) (increase_percentage : ℝ) (original_price : ℝ) :
  new_price = 420 ∧ 
  increase_percentage = 40 ∧ 
  new_price = original_price * (1 + increase_percentage / 100) → 
  original_price = 300 := by
sorry

end book_price_increase_l2299_229938


namespace quadratic_factorization_l2299_229948

theorem quadratic_factorization (m n : ℤ) : 
  (∀ x, x^2 - 7*x + n = (x - 3) * (x + m)) → m - n = -16 := by
  sorry

end quadratic_factorization_l2299_229948


namespace sum_of_prime_factors_88200_l2299_229940

def sum_of_prime_factors (n : ℕ) : ℕ := (Nat.factors n).toFinset.sum id

theorem sum_of_prime_factors_88200 :
  sum_of_prime_factors 88200 = 17 := by sorry

end sum_of_prime_factors_88200_l2299_229940


namespace probability_of_exact_score_l2299_229949

def num_questions : ℕ := 20
def num_choices : ℕ := 4
def correct_answers : ℕ := 10

def probability_correct : ℚ := 1 / num_choices
def probability_incorrect : ℚ := 1 - probability_correct

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem probability_of_exact_score :
  (binomial_coefficient num_questions correct_answers : ℚ) *
  (probability_correct ^ correct_answers) *
  (probability_incorrect ^ (num_questions - correct_answers)) =
  93350805 / 1073741824 := by sorry

end probability_of_exact_score_l2299_229949


namespace mrs_hilt_reading_rate_l2299_229932

/-- Mrs. Hilt's daily reading rate -/
def daily_reading_rate (total_books : ℕ) (total_days : ℕ) : ℚ :=
  total_books / total_days

/-- Theorem: Mrs. Hilt reads 5 books per day -/
theorem mrs_hilt_reading_rate :
  daily_reading_rate 15 3 = 5 := by
  sorry

end mrs_hilt_reading_rate_l2299_229932


namespace trajectory_and_max_distance_l2299_229937

-- Define the fixed point F
def F : ℝ × ℝ := (1, 0)

-- Define the line l: x = 2
def l (x : ℝ) : Prop := x = 2

-- Define the distance ratio condition
def distance_ratio (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  (((x - F.1)^2 + (y - F.2)^2).sqrt / |2 - x|) = Real.sqrt 2 / 2

-- Define the ellipse equation
def ellipse (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  x^2 / 2 + y^2 = 1

-- Define the line for maximum distance calculation
def max_distance_line (x y : ℝ) : Prop :=
  x / Real.sqrt 2 + y = 1

-- Theorem statement
theorem trajectory_and_max_distance :
  -- Part 1: Trajectory is an ellipse
  (∀ M : ℝ × ℝ, distance_ratio M ↔ ellipse M) ∧
  -- Part 2: Maximum distance exists
  (∃ d : ℝ, ∀ M : ℝ × ℝ, ellipse M →
    let (x, y) := M
    abs (x / Real.sqrt 2 + y - 1) / Real.sqrt ((1 / 2) + 1) ≤ d) ∧
  -- Part 3: Maximum distance value
  (let max_d := (2 * Real.sqrt 3 + Real.sqrt 6) / 3
   ∃ M : ℝ × ℝ, ellipse M ∧
     let (x, y) := M
     abs (x / Real.sqrt 2 + y - 1) / Real.sqrt ((1 / 2) + 1) = max_d) :=
by sorry


end trajectory_and_max_distance_l2299_229937


namespace stratified_sampling_theorem_l2299_229907

/-- Represents the number of employees in each job category -/
structure EmployeeCount where
  total : ℕ
  senior : ℕ
  middle : ℕ
  general : ℕ

/-- Represents the number of sampled employees in each job category -/
structure SampledCount where
  senior : ℕ
  middle : ℕ
  general : ℕ

/-- Checks if the sampling is stratified correctly -/
def is_stratified_sampling (ec : EmployeeCount) (sample_size : ℕ) (sc : SampledCount) : Prop :=
  sc.senior * ec.total = sample_size * ec.senior ∧
  sc.middle * ec.total = sample_size * ec.middle ∧
  sc.general * ec.total = sample_size * ec.general

theorem stratified_sampling_theorem (ec : EmployeeCount) (sample_size : ℕ) :
  ec.total = ec.senior + ec.middle + ec.general →
  ∃ (sc : SampledCount), 
    sc.senior + sc.middle + sc.general = sample_size ∧
    is_stratified_sampling ec sample_size sc := by
  sorry

#check stratified_sampling_theorem

end stratified_sampling_theorem_l2299_229907


namespace survival_probability_estimate_l2299_229960

/-- Represents the survival data for a sample of seedlings -/
structure SeedlingSample where
  transplanted : ℕ
  survived : ℕ
  survivalRate : ℚ

/-- The data set of seedling survival samples -/
def seedlingData : List SeedlingSample := [
  ⟨20, 15, 75/100⟩,
  ⟨40, 33, 33/40⟩,
  ⟨100, 78, 39/50⟩,
  ⟨200, 158, 79/100⟩,
  ⟨400, 321, 801/1000⟩,
  ⟨1000, 801, 801/1000⟩
]

/-- Estimates the overall probability of seedling survival -/
def estimateSurvivalProbability (data : List SeedlingSample) : ℚ :=
  sorry

/-- Theorem stating that the estimated survival probability is approximately 0.80 -/
theorem survival_probability_estimate :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ 
  |estimateSurvivalProbability seedlingData - 4/5| < ε :=
sorry

end survival_probability_estimate_l2299_229960


namespace smallest_total_hits_is_twelve_l2299_229996

/-- Represents a baseball player's batting statistics -/
structure BattingStats where
  initialHits : ℕ
  initialAtBats : ℕ
  newHits : ℕ
  newAtBats : ℕ
  initialAverage : ℚ
  newAverage : ℚ

/-- Calculates the smallest number of total hits given initial and new batting averages -/
def smallestTotalHits (stats : BattingStats) : ℕ :=
  stats.initialHits + stats.newHits

/-- Theorem: The smallest number of total hits is 12 given the specified conditions -/
theorem smallest_total_hits_is_twelve :
  ∃ (stats : BattingStats),
    stats.initialAverage = 360 / 1000 ∧
    stats.newAverage = 400 / 1000 ∧
    stats.newAtBats = stats.initialAtBats + 5 ∧
    smallestTotalHits stats = 12 ∧
    ∀ (otherStats : BattingStats),
      otherStats.initialAverage = 360 / 1000 ∧
      otherStats.newAverage = 400 / 1000 ∧
      otherStats.newAtBats = otherStats.initialAtBats + 5 →
      smallestTotalHits otherStats ≥ 12 :=
by sorry


end smallest_total_hits_is_twelve_l2299_229996


namespace bad_carrots_count_l2299_229989

theorem bad_carrots_count (haley_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) : 
  haley_carrots = 39 → mom_carrots = 38 → good_carrots = 64 → 
  haley_carrots + mom_carrots - good_carrots = 13 := by
  sorry

end bad_carrots_count_l2299_229989


namespace triangle_range_theorem_l2299_229994

theorem triangle_range_theorem (a b x : ℝ) (B : ℝ) (has_two_solutions : Prop) :
  a = x →
  b = 2 →
  B = π / 3 →
  has_two_solutions →
  2 < x ∧ x < 4 * Real.sqrt 3 / 3 := by
  sorry

end triangle_range_theorem_l2299_229994


namespace complex_equation_solution_l2299_229998

theorem complex_equation_solution :
  ∀ z : ℂ, z + Complex.abs z = 2 + Complex.I → z = (3/4 : ℝ) + Complex.I := by
sorry

end complex_equation_solution_l2299_229998


namespace mixed_groups_count_l2299_229901

theorem mixed_groups_count (total_children : ℕ) (total_groups : ℕ) (group_size : ℕ)
  (boy_boy_photos : ℕ) (girl_girl_photos : ℕ) :
  total_children = 300 →
  total_groups = 100 →
  group_size = 3 →
  total_children = total_groups * group_size →
  boy_boy_photos = 100 →
  girl_girl_photos = 56 →
  (∃ (mixed_groups : ℕ),
    mixed_groups = 72 ∧
    mixed_groups * 2 = total_groups * group_size - boy_boy_photos - girl_girl_photos) :=
by sorry

end mixed_groups_count_l2299_229901


namespace polygon_interior_angles_l2299_229984

theorem polygon_interior_angles (n : ℕ) (d : ℝ) (largest_angle : ℝ) : 
  n ≥ 3 →
  d = 3 →
  largest_angle = 150 →
  (n : ℝ) * (2 * largest_angle - d * (n - 1)) / 2 = 180 * (n - 2) →
  n = 24 :=
by sorry

end polygon_interior_angles_l2299_229984


namespace philip_paintings_l2299_229939

/-- The number of paintings a painter will have after a certain number of days -/
def total_paintings (initial_paintings : ℕ) (paintings_per_day : ℕ) (days : ℕ) : ℕ :=
  initial_paintings + paintings_per_day * days

/-- Theorem: Philip will have 80 paintings after 30 days -/
theorem philip_paintings : total_paintings 20 2 30 = 80 := by
  sorry

end philip_paintings_l2299_229939


namespace base_equation_solution_l2299_229951

/-- Represents a digit in base b --/
def Digit (b : ℕ) := Fin b

/-- Converts a natural number to its representation in base b --/
def toBase (n : ℕ) (b : ℕ) : List (Digit b) :=
  sorry

/-- Adds two numbers in base b --/
def addBase (x y : List (Digit b)) : List (Digit b) :=
  sorry

/-- Checks if a list of digits is equal to another list of digits --/
def digitListEq (x y : List (Digit b)) : Prop :=
  sorry

theorem base_equation_solution :
  ∀ b : ℕ, b > 1 →
    (digitListEq (addBase (toBase 295 b) (toBase 467 b)) (toBase 762 b)) ↔ b = 10 := by
  sorry

end base_equation_solution_l2299_229951


namespace race_head_start_l2299_229914

/-- Given two runners A and B, where A's speed is 20/16 times B's speed,
    this theorem proves that A should give B a head start of 1/5 of the
    total race length for the race to end in a dead heat. -/
theorem race_head_start (v_B : ℝ) (L : ℝ) (h_pos_v : v_B > 0) (h_pos_L : L > 0) :
  let v_A := (20 / 16) * v_B
  let x := 1 / 5
  L / v_A = (L - x * L) / v_B := by
  sorry

#check race_head_start

end race_head_start_l2299_229914


namespace find_a_l2299_229923

theorem find_a : ∃ (a : ℝ), (∀ (x : ℝ), (2 * x - a ≤ -1) ↔ (x ≤ 1)) → a = 3 :=
by sorry

end find_a_l2299_229923


namespace pythagorean_theorem_geometric_dissection_l2299_229958

/-- Pythagorean theorem using geometric dissection -/
theorem pythagorean_theorem_geometric_dissection 
  (a b c : ℝ) 
  (h_right_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_hypotenuse : c = max a b)
  (h_inner_square : ∃ (s : ℝ), s = |b - a| ∧ s^2 = (b - a)^2)
  (h_area_equality : c^2 = 2*a*b + (b - a)^2) : 
  a^2 + b^2 = c^2 := by
sorry

end pythagorean_theorem_geometric_dissection_l2299_229958


namespace fifty_eight_impossible_l2299_229918

/-- Represents the population of Rivertown -/
structure RivertownPopulation where
  people : ℕ
  dogs : ℕ
  cats : ℕ
  rabbits : ℕ
  chickens : ℕ
  people_dog_ratio : people = 5 * dogs
  cat_rabbit_ratio : cats = 2 * rabbits
  chicken_people_ratio : chickens = 4 * people

/-- The total population of Rivertown -/
def totalPopulation (pop : RivertownPopulation) : ℕ :=
  pop.people + pop.dogs + pop.cats + pop.rabbits + pop.chickens

/-- Theorem stating that 58 cannot be the total population of Rivertown -/
theorem fifty_eight_impossible (pop : RivertownPopulation) : totalPopulation pop ≠ 58 := by
  sorry

end fifty_eight_impossible_l2299_229918


namespace standard_deviation_shift_l2299_229956

-- Define the standard deviation function for a list of real numbers
def standardDeviation (xs : List ℝ) : ℝ := sorry

-- Define a function to add a constant to each element of a list
def addConstant (xs : List ℝ) (c : ℝ) : List ℝ := sorry

-- Theorem statement
theorem standard_deviation_shift (a b c : ℝ) :
  standardDeviation [a + 2, b + 2, c + 2] = 2 →
  standardDeviation [a, b, c] = 2 := by sorry

end standard_deviation_shift_l2299_229956


namespace trig_expression_simplification_l2299_229999

theorem trig_expression_simplification :
  (Real.cos (5 * π / 180))^2 - (Real.sin (5 * π / 180))^2 = 
  2 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180) := by
  sorry

end trig_expression_simplification_l2299_229999


namespace num_sandwiches_al_can_order_l2299_229912

/-- Represents the number of different types of bread offered at the deli. -/
def num_breads : Nat := 5

/-- Represents the number of different types of meat offered at the deli. -/
def num_meats : Nat := 7

/-- Represents the number of different types of cheese offered at the deli. -/
def num_cheeses : Nat := 6

/-- Represents the number of restricted sandwich combinations. -/
def num_restricted : Nat := 16

/-- Theorem stating the number of different sandwiches Al could order. -/
theorem num_sandwiches_al_can_order :
  (num_breads * num_meats * num_cheeses) - num_restricted = 194 := by
  sorry

end num_sandwiches_al_can_order_l2299_229912


namespace race_time_B_b_finish_time_l2299_229942

/-- Calculates the time taken by runner B to finish a race given the conditions --/
theorem race_time_B (race_distance : ℝ) (time_A : ℝ) (beat_distance : ℝ) : ℝ :=
  let distance_B_in_time_A := race_distance - beat_distance
  let speed_B := distance_B_in_time_A / time_A
  race_distance / speed_B

/-- Proves that B finishes the race in 25 seconds given the specified conditions --/
theorem b_finish_time (race_distance : ℝ) (time_A : ℝ) (beat_distance : ℝ) :
  race_time_B race_distance time_A beat_distance = 25 :=
by
  -- Assuming race_distance = 110, time_A = 20, and beat_distance = 22
  have h1 : race_distance = 110 := by sorry
  have h2 : time_A = 20 := by sorry
  have h3 : beat_distance = 22 := by sorry
  
  -- The proof would go here
  sorry

end race_time_B_b_finish_time_l2299_229942


namespace vladimir_digits_puzzle_l2299_229963

/-- Represents a three-digit number formed by digits a, b, c in that order -/
def form_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem vladimir_digits_puzzle :
  ∀ a b c : ℕ,
  a > b → b > c → c > 0 →
  form_number a b c = form_number c b a + form_number c a b →
  a = 9 ∧ b = 5 ∧ c = 4 := by
sorry

end vladimir_digits_puzzle_l2299_229963


namespace square_equality_necessary_not_sufficient_l2299_229910

theorem square_equality_necessary_not_sufficient :
  (∀ x y : ℝ, x = y → x^2 = y^2) ∧
  (∃ x y : ℝ, x^2 = y^2 ∧ x ≠ y) := by
  sorry

end square_equality_necessary_not_sufficient_l2299_229910


namespace three_top_numbers_count_l2299_229921

/-- A function that checks if a number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- A function that returns the units digit of a number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- A function that checks if three consecutive numbers satisfy the "Three Top Numbers" conditions -/
def isThreeTopNumbers (n : ℕ) : Prop :=
  isTwoDigit n ∧ isTwoDigit (n + 1) ∧ isTwoDigit (n + 2) ∧
  isTwoDigit (n + (n + 1) + (n + 2)) ∧
  (unitsDigit (n + (n + 1) + (n + 2)) > unitsDigit n) ∧
  (unitsDigit (n + (n + 1) + (n + 2)) > unitsDigit (n + 1)) ∧
  (unitsDigit (n + (n + 1) + (n + 2)) > unitsDigit (n + 2))

/-- The theorem stating that there are exactly 5 sets of "Three Top Numbers" -/
theorem three_top_numbers_count :
  ∃! (s : Finset ℕ), (∀ n ∈ s, isThreeTopNumbers n) ∧ s.card = 5 := by
  sorry

end three_top_numbers_count_l2299_229921


namespace total_tickets_proof_l2299_229952

/-- The number of tickets Tom spent at the 'dunk a clown' booth -/
def tickets_spent_at_booth : ℕ := 28

/-- The number of rides Tom went on -/
def number_of_rides : ℕ := 3

/-- The cost of each ride in tickets -/
def cost_per_ride : ℕ := 4

/-- The total number of tickets Tom bought at the state fair -/
def total_tickets : ℕ := tickets_spent_at_booth + number_of_rides * cost_per_ride

theorem total_tickets_proof : total_tickets = 40 := by
  sorry

end total_tickets_proof_l2299_229952


namespace sin_cos_identity_l2299_229982

theorem sin_cos_identity (x : ℝ) (h : Real.sin (x + π / 3) = 1 / 3) :
  Real.sin (5 * π / 3 - x) - Real.cos (2 * x - π / 3) = 4 / 9 := by
  sorry

end sin_cos_identity_l2299_229982


namespace rational_set_not_just_positive_and_negative_l2299_229902

theorem rational_set_not_just_positive_and_negative : 
  ∃ q : ℚ, q ∉ {x : ℚ | x > 0} ∪ {x : ℚ | x < 0} := by
  sorry

end rational_set_not_just_positive_and_negative_l2299_229902


namespace circle_through_three_points_l2299_229929

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The standard equation of a circle -/
def CircleEquation (center : Point) (radius : ℝ) : Prop :=
  ∀ (x y : ℝ), (x - center.x)^2 + (y - center.y)^2 = radius^2

theorem circle_through_three_points :
  let A : Point := ⟨-4, 0⟩
  let B : Point := ⟨0, 2⟩
  let O : Point := ⟨0, 0⟩
  let center : Point := ⟨-2, 1⟩
  let radius : ℝ := Real.sqrt 5
  (CircleEquation center radius) ∧
  (center.x - A.x)^2 + (center.y - A.y)^2 = radius^2 ∧
  (center.x - B.x)^2 + (center.y - B.y)^2 = radius^2 ∧
  (center.x - O.x)^2 + (center.y - O.y)^2 = radius^2 :=
by
  sorry

#check circle_through_three_points

end circle_through_three_points_l2299_229929


namespace largest_perfect_square_factor_of_1800_l2299_229974

theorem largest_perfect_square_factor_of_1800 : 
  (∃ (n : ℕ), n^2 = 900 ∧ n^2 ∣ 1800 ∧ ∀ (m : ℕ), m^2 ∣ 1800 → m^2 ≤ 900) := by
  sorry

end largest_perfect_square_factor_of_1800_l2299_229974


namespace jasmine_total_weight_l2299_229947

/-- Calculates the total weight in pounds that Jasmine has to carry given the weight of chips and cookies, and the quantities purchased. -/
theorem jasmine_total_weight (chip_weight : ℕ) (cookie_weight : ℕ) (chip_quantity : ℕ) (cookie_multiplier : ℕ) : 
  chip_weight = 20 →
  cookie_weight = 9 →
  chip_quantity = 6 →
  cookie_multiplier = 4 →
  (chip_weight * chip_quantity + cookie_weight * (cookie_multiplier * chip_quantity)) / 16 = 21 := by
  sorry

#check jasmine_total_weight

end jasmine_total_weight_l2299_229947


namespace max_distance_between_cubic_and_quadratic_roots_l2299_229917

open Complex Set

theorem max_distance_between_cubic_and_quadratic_roots : ∃ (max_dist : ℝ),
  max_dist = 3 * Real.sqrt 7 ∧
  ∀ (a b : ℂ),
    (a^3 - 27 = 0) →
    (b^2 - 6*b + 9 = 0) →
    abs (a - b) ≤ max_dist ∧
    ∃ (a₀ b₀ : ℂ),
      (a₀^3 - 27 = 0) ∧
      (b₀^2 - 6*b₀ + 9 = 0) ∧
      abs (a₀ - b₀) = max_dist :=
by sorry

end max_distance_between_cubic_and_quadratic_roots_l2299_229917


namespace triangle_square_perimeter_difference_l2299_229941

theorem triangle_square_perimeter_difference (d : ℕ) : 
  (∃ (t s : ℝ), 
    t > 0 ∧ s > 0 ∧  -- positive side lengths
    3 * t - 4 * s = 4020 ∧  -- perimeter difference
    t = |s - 12| + d ∧  -- side length relationship
    4 * s > 0)  -- square perimeter > 0
  ↔ d > 1352 :=
sorry

end triangle_square_perimeter_difference_l2299_229941


namespace square_value_l2299_229945

theorem square_value : ∃ (square : ℝ), (6400000 : ℝ) / 400 = 1.6 * square ∧ square = 10000 := by
  sorry

end square_value_l2299_229945


namespace sum_and_product_problem_l2299_229975

theorem sum_and_product_problem (x y : ℝ) 
  (sum_eq : x + y = 15) 
  (product_eq : x * y = 36) : 
  (1 / x + 1 / y = 5 / 12) ∧ (x^2 + y^2 = 153) := by
  sorry

end sum_and_product_problem_l2299_229975


namespace five_lines_eleven_intersections_impossible_five_lines_nine_intersections_possible_l2299_229979

/-- The maximum number of intersection points for n lines in a plane,
    where no three lines intersect at one point -/
def max_intersections (n : ℕ) : ℕ := n.choose 2

/-- Represents a configuration of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  num_intersections : ℕ
  no_triple_intersections : Bool

/-- Theorem stating the impossibility of 5 lines with 11 intersections -/
theorem five_lines_eleven_intersections_impossible :
  ∀ (config : LineConfiguration),
    config.num_lines = 5 ∧ 
    config.no_triple_intersections = true →
    config.num_intersections ≠ 11 :=
sorry

/-- Theorem stating the possibility of 5 lines with 9 intersections -/
theorem five_lines_nine_intersections_possible :
  ∃ (config : LineConfiguration),
    config.num_lines = 5 ∧ 
    config.no_triple_intersections = true ∧
    config.num_intersections = 9 :=
sorry

end five_lines_eleven_intersections_impossible_five_lines_nine_intersections_possible_l2299_229979


namespace train_length_proof_l2299_229933

/-- Calculates the length of a train given bridge length, crossing time, and train speed. -/
def train_length (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ) : ℝ :=
  train_speed * crossing_time - bridge_length

/-- Proves that given the specific conditions, the train length is 844 meters. -/
theorem train_length_proof (bridge_length : ℝ) (crossing_time : ℝ) (train_speed : ℝ)
  (h1 : bridge_length = 200)
  (h2 : crossing_time = 36)
  (h3 : train_speed = 29) :
  train_length bridge_length crossing_time train_speed = 844 := by
  sorry

#eval train_length 200 36 29

end train_length_proof_l2299_229933


namespace fuel_efficiency_savings_l2299_229976

theorem fuel_efficiency_savings
  (old_efficiency : ℝ)
  (new_efficiency_improvement : ℝ)
  (fuel_cost_increase : ℝ)
  (h1 : new_efficiency_improvement = 0.3)
  (h2 : fuel_cost_increase = 0.25)
  : ∃ (savings : ℝ), abs (savings - 0.0385) < 0.0001 :=
by
  sorry

end fuel_efficiency_savings_l2299_229976


namespace complex_product_positive_implies_zero_l2299_229992

theorem complex_product_positive_implies_zero (a : ℝ) :
  (Complex.I * (a - Complex.I)).re > 0 → a = 0 := by sorry

end complex_product_positive_implies_zero_l2299_229992


namespace bowling_team_average_weight_l2299_229925

theorem bowling_team_average_weight 
  (original_players : ℕ) 
  (original_average : ℝ) 
  (new_player1_weight : ℝ) 
  (new_player2_weight : ℝ) :
  original_players = 7 →
  original_average = 103 →
  new_player1_weight = 110 →
  new_player2_weight = 60 →
  let total_weight := original_players * original_average + new_player1_weight + new_player2_weight
  let new_total_players := original_players + 2
  let new_average := total_weight / new_total_players
  new_average = 99 := by
  sorry

end bowling_team_average_weight_l2299_229925


namespace sum_of_xyz_l2299_229971

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 40) (h2 : x * z = 80) (h3 : y * z = 120) :
  x + y + z = 22 * Real.sqrt 15 / 3 := by
  sorry

end sum_of_xyz_l2299_229971


namespace vectors_parallel_when_m_neg_one_l2299_229950

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

/-- Vector a parameterized by m -/
def a (m : ℝ) : ℝ × ℝ := (2*m - 1, m)

/-- Vector b -/
def b : ℝ × ℝ := (3, 1)

/-- Theorem stating that vectors a and b are parallel when m = -1 -/
theorem vectors_parallel_when_m_neg_one :
  are_parallel (a (-1)) b := by sorry

end vectors_parallel_when_m_neg_one_l2299_229950


namespace chocolate_chip_recipe_l2299_229977

theorem chocolate_chip_recipe (total_recipes : ℕ) (total_cups : ℕ) 
  (h1 : total_recipes = 23) (h2 : total_cups = 46) :
  total_cups / total_recipes = 2 := by
  sorry

end chocolate_chip_recipe_l2299_229977


namespace susans_gift_is_eight_l2299_229988

/-- The number of apples Sean had initially -/
def initial_apples : ℕ := 9

/-- The total number of apples Sean had after Susan's gift -/
def total_apples : ℕ := 17

/-- The number of apples Susan gave to Sean -/
def susans_gift : ℕ := total_apples - initial_apples

theorem susans_gift_is_eight : susans_gift = 8 := by
  sorry

end susans_gift_is_eight_l2299_229988


namespace symmetry_x_axis_of_point_A_l2299_229964

/-- A point in three-dimensional space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetry_x_axis (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := -p.z }

theorem symmetry_x_axis_of_point_A :
  let A : Point3D := { x := 1, y := 2, z := 1 }
  symmetry_x_axis A = { x := 1, y := 2, z := -1 } := by
  sorry

end symmetry_x_axis_of_point_A_l2299_229964


namespace gis_main_functions_l2299_229978

/-- Represents the main functions of geographic information technology -/
inductive GISFunction
  | input : GISFunction
  | manage : GISFunction
  | analyze : GISFunction
  | express : GISFunction

/-- Represents the type of data handled by geographic information technology -/
def GeospatialData : Type := Unit

/-- The set of main functions of geographic information technology -/
def mainFunctions : Set GISFunction :=
  {GISFunction.input, GISFunction.manage, GISFunction.analyze, GISFunction.express}

/-- States that the main functions of geographic information technology
    are to input, manage, analyze, and express geospatial data -/
theorem gis_main_functions :
  ∀ f : GISFunction, f ∈ mainFunctions →
  ∃ (d : GeospatialData), (f = GISFunction.input ∨ f = GISFunction.manage ∨
                           f = GISFunction.analyze ∨ f = GISFunction.express) :=
sorry

end gis_main_functions_l2299_229978


namespace yolanda_total_points_l2299_229967

/-- Calculate the total points scored by a basketball player over a season. -/
def total_points_scored (games : ℕ) (free_throws two_pointers three_pointers : ℕ) : ℕ :=
  games * (free_throws * 1 + two_pointers * 2 + three_pointers * 3)

/-- Theorem: Yolanda's total points scored over the entire season is 345. -/
theorem yolanda_total_points : 
  total_points_scored 15 4 5 3 = 345 := by
  sorry

end yolanda_total_points_l2299_229967


namespace fault_line_movement_l2299_229970

/-- The total movement of a fault line over two years, given its movement in each year. -/
theorem fault_line_movement (movement_past_year : ℝ) (movement_year_before : ℝ) 
  (h1 : movement_past_year = 1.25)
  (h2 : movement_year_before = 5.25) : 
  movement_past_year + movement_year_before = 6.50 := by
  sorry

end fault_line_movement_l2299_229970


namespace ellipse_circle_tangent_l2299_229987

/-- Given an ellipse M and a circle N with specific properties, prove their equations and the equation of their common tangent line. -/
theorem ellipse_circle_tangent (a b c : ℝ) (k m : ℝ) :
  a > 0 ∧ b > 0 ∧ a > b ∧  -- conditions on a, b
  c / a = 1 / 2 ∧  -- eccentricity
  a^2 / c - c = 3 ∧  -- distance condition
  c > 0 →  -- c is positive (implied by being a distance)
  -- Prove:
  ((∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 4 + y^2 / 3 = 1) ∧  -- equation of M
   (∀ x y : ℝ, (x - c)^2 + y^2 = a^2 + c^2 ↔ (x - 1)^2 + y^2 = 5) ∧  -- equation of N
   ((k = 1/2 ∧ m = 2) ∨ (k = -1/2 ∧ m = -2)) ∧  -- equation of tangent line l
   (∀ x : ℝ, (x^2 / 4 + (k * x + m)^2 / 3 = 1 →  -- l is tangent to M
              ∃! y : ℝ, y = k * x + m ∧ x^2 / 4 + y^2 / 3 = 1) ∧
    ((k * 1 + m)^2 + 1^2 = 5)))  -- l is tangent to N
  := by sorry

end ellipse_circle_tangent_l2299_229987


namespace divisibility_of_2_power_n_minus_1_l2299_229961

theorem divisibility_of_2_power_n_minus_1 :
  ∃ (n : ℕ+), ∃ (k : ℕ), 2^n.val - 1 = 17 * k ∧
  ∀ (m : ℕ), 10 ≤ m → m ≤ 20 → m ≠ 17 → ¬∃ (l : ℕ+), ∃ (j : ℕ), 2^l.val - 1 = m * j :=
sorry

end divisibility_of_2_power_n_minus_1_l2299_229961


namespace isosceles_trapezoid_right_angle_points_l2299_229991

/-- An isosceles trapezoid with the given properties -/
structure IsoscelesTrapezoid where
  a : ℝ  -- length of AB
  c : ℝ  -- length of CD
  h : ℝ  -- perpendicular distance from A to CD
  a_positive : 0 < a
  c_positive : 0 < c
  h_positive : 0 < h
  c_le_a : c ≤ a  -- As CD is parallel to and shorter than AB

/-- The point X on the axis of symmetry -/
def X (t : IsoscelesTrapezoid) := {x : ℝ // 0 ≤ x ∧ x ≤ t.h}

/-- The theorem stating the conditions for X to exist and its distance from AB -/
theorem isosceles_trapezoid_right_angle_points (t : IsoscelesTrapezoid) :
  ∃ (x : X t), (x.val = t.h / 2 - Real.sqrt (t.h^2 - t.a * t.c) / 2 ∨
                x.val = t.h / 2 + Real.sqrt (t.h^2 - t.a * t.c) / 2) ↔
  t.h^2 ≥ t.a * t.c :=
sorry

end isosceles_trapezoid_right_angle_points_l2299_229991


namespace prime_sum_theorem_l2299_229972

theorem prime_sum_theorem (x y : ℕ) : 
  Nat.Prime x → Nat.Prime y → x < y → x + y = 36 → 4 * x + y = 51 := by sorry

end prime_sum_theorem_l2299_229972


namespace unit_prices_min_type_A_boxes_l2299_229997

-- Define the types of gift boxes
inductive GiftBox
| A
| B

-- Define the unit prices as variables
variable (price_A price_B : ℕ)

-- Define the conditions of the problem
axiom first_purchase : 10 * price_A + 15 * price_B = 2800
axiom second_purchase : 6 * price_A + 5 * price_B = 1200

-- Define the total number of boxes and maximum cost
def total_boxes : ℕ := 40
def max_cost : ℕ := 4500

-- Theorem for the unit prices
theorem unit_prices : price_A = 100 ∧ price_B = 120 := by sorry

-- Function to calculate the total cost
def total_cost (num_A : ℕ) : ℕ :=
  num_A * price_A + (total_boxes - num_A) * price_B

-- Theorem for the minimum number of type A boxes
theorem min_type_A_boxes : 
  ∀ num_A : ℕ, num_A ≥ 15 → total_cost num_A ≤ max_cost := by sorry

end unit_prices_min_type_A_boxes_l2299_229997


namespace equation_solution_l2299_229995

theorem equation_solution (a b : ℝ) (x₁ x₂ x₃ : ℝ) : 
  a > 0 → 
  b > 0 → 
  (∀ x : ℝ, Real.sqrt (|x|) + Real.sqrt (|x + a|) = b ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  x₁ < x₂ →
  x₂ < x₃ →
  x₃ = b →
  a + b = 144 := by
sorry

end equation_solution_l2299_229995


namespace parabola_intersection_l2299_229922

/-- The function f(x) = x² --/
def f (x : ℝ) : ℝ := x^2

/-- Theorem: Given two points on the parabola y = x², with the first point at x = 1
    and the second at x = 4, if we trisect the line segment between these points
    and draw a horizontal line through the first trisection point (closer to the first point),
    then this line intersects the parabola at x = -2. --/
theorem parabola_intersection (x₁ x₂ x₃ : ℝ) (y₁ y₂ y₃ : ℝ) :
  x₁ = 1 →
  x₂ = 4 →
  y₁ = f x₁ →
  y₂ = f x₂ →
  let xc := (2 * x₁ + x₂) / 3
  let yc := f xc
  y₃ = yc →
  y₃ = f x₃ →
  x₃ ≠ xc →
  x₃ = -2 :=
by sorry

end parabola_intersection_l2299_229922


namespace male_to_female_ratio_l2299_229903

/-- Represents the Math club with its member composition -/
structure MathClub where
  total_members : ℕ
  female_members : ℕ
  male_members : ℕ
  total_is_sum : total_members = female_members + male_members

/-- The specific Math club instance from the problem -/
def problem_club : MathClub :=
  { total_members := 18
    female_members := 6
    male_members := 12
    total_is_sum := by rfl }

/-- The ratio of male to female members is 2:1 -/
theorem male_to_female_ratio (club : MathClub) 
  (h1 : club.total_members = 18) 
  (h2 : club.female_members = 6) : 
  club.male_members / club.female_members = 2 := by
  sorry

#check male_to_female_ratio problem_club rfl rfl

end male_to_female_ratio_l2299_229903


namespace starting_lineup_count_l2299_229936

def team_size : ℕ := 12
def lineup_size : ℕ := 5
def non_captain_size : ℕ := lineup_size - 1

theorem starting_lineup_count : 
  team_size * (Nat.choose (team_size - 1) non_captain_size) = 3960 := by
  sorry

end starting_lineup_count_l2299_229936


namespace route_down_length_l2299_229966

/-- Proves that the length of the route down the mountain is 15 miles -/
theorem route_down_length (time_up time_down : ℝ) (rate_up rate_down : ℝ) 
  (h1 : time_up = time_down)
  (h2 : rate_down = 1.5 * rate_up)
  (h3 : rate_up = 5)
  (h4 : time_up = 2) : 
  rate_down * time_down = 15 := by
  sorry

end route_down_length_l2299_229966


namespace socks_bought_l2299_229973

/-- Given John's sock inventory changes, prove the number of new socks bought. -/
theorem socks_bought (initial : ℕ) (thrown_away : ℕ) (final : ℕ) 
  (h1 : initial = 33)
  (h2 : thrown_away = 19)
  (h3 : final = 27) :
  final - (initial - thrown_away) = 13 := by
  sorry

end socks_bought_l2299_229973


namespace sum_of_squares_with_given_means_l2299_229983

theorem sum_of_squares_with_given_means (x y z : ℝ) 
  (h_positive : x > 0 ∧ y > 0 ∧ z > 0)
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 576 := by
sorry

end sum_of_squares_with_given_means_l2299_229983


namespace angle_sum_theorem_l2299_229943

theorem angle_sum_theorem (α β : Real) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (eq1 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (eq2 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π / 2 := by
sorry

end angle_sum_theorem_l2299_229943


namespace beatrice_has_highest_answer_l2299_229906

def albert_calculation (x : ℕ) : ℕ := 2 * ((3 * x + 5) - 3)

def beatrice_calculation (x : ℕ) : ℕ := 2 * ((x * x + 3) - 7)

def carlos_calculation (x : ℕ) : ℚ := ((5 * x - 4 + 6) : ℚ) / 2

theorem beatrice_has_highest_answer :
  let start := 15
  beatrice_calculation start > albert_calculation start ∧
  (beatrice_calculation start : ℚ) > carlos_calculation start := by
sorry

#eval albert_calculation 15
#eval beatrice_calculation 15
#eval carlos_calculation 15

end beatrice_has_highest_answer_l2299_229906


namespace factorial_ratio_l2299_229981

theorem factorial_ratio : Nat.factorial 12 / Nat.factorial 11 = 12 := by
  sorry

end factorial_ratio_l2299_229981


namespace teresa_jogging_distance_l2299_229909

def speed : ℝ := 5
def time : ℝ := 5
def distance : ℝ := speed * time

theorem teresa_jogging_distance : distance = 25 := by
  sorry

end teresa_jogging_distance_l2299_229909


namespace solution_set_f_leq_6_range_a_for_nonempty_solution_l2299_229926

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| + |2*x - 3|

-- Theorem for part I
theorem solution_set_f_leq_6 :
  {x : ℝ | f x ≤ 6} = {x : ℝ | -1 ≤ x ∧ x ≤ 2} :=
by sorry

-- Theorem for part II
theorem range_a_for_nonempty_solution :
  {a : ℝ | ∃ x, f x < |a - 1|} = {a : ℝ | a < -3 ∨ a > 5} :=
by sorry

end solution_set_f_leq_6_range_a_for_nonempty_solution_l2299_229926


namespace hannah_strawberries_l2299_229928

/-- The number of strawberries Hannah has at the end of April -/
def strawberries_at_end_of_april (daily_harvest : ℕ) (days_in_april : ℕ) (given_away : ℕ) (stolen : ℕ) : ℕ :=
  daily_harvest * days_in_april - (given_away + stolen)

theorem hannah_strawberries :
  strawberries_at_end_of_april 5 30 20 30 = 100 := by
  sorry

end hannah_strawberries_l2299_229928


namespace binary_1001101_equals_octal_115_l2299_229965

def binary_to_decimal (b : List Bool) : ℕ :=
  (List.enumFrom 0 b).foldl (λ acc (i, x) => acc + if x then 2^i else 0) 0

def octal_to_decimal (o : List ℕ) : ℕ :=
  (List.enumFrom 0 o).foldl (λ acc (i, x) => acc + x * 8^i) 0

theorem binary_1001101_equals_octal_115 :
  binary_to_decimal [true, false, true, true, false, false, true] =
  octal_to_decimal [5, 1, 1] := by
  sorry

end binary_1001101_equals_octal_115_l2299_229965


namespace cherry_pie_pitting_time_l2299_229920

/-- Represents the time needed to pit cherries for each pound -/
structure PittingTime where
  first : ℕ  -- Time in minutes for the first pound
  second : ℕ -- Time in minutes for the second pound
  third : ℕ  -- Time in minutes for the third pound

/-- Calculates the total time in hours to pit cherries for a cherry pie -/
def total_pitting_time (pt : PittingTime) : ℚ :=
  (pt.first + pt.second + pt.third) / 60

/-- Theorem: Given the conditions, it takes 2 hours to pit all cherries for the pie -/
theorem cherry_pie_pitting_time :
  ∀ (pt : PittingTime),
    (∃ (n : ℕ), pt.first = 10 * (80 / 20) ∧
                pt.second = 8 * (80 / 20) ∧
                pt.third = 12 * (80 / 20) ∧
                n = 3) →
    total_pitting_time pt = 2 := by
  sorry


end cherry_pie_pitting_time_l2299_229920


namespace sports_club_overlap_l2299_229913

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ)
  (h_total : total = 80)
  (h_badminton : badminton = 48)
  (h_tennis : tennis = 46)
  (h_neither : neither = 7)
  : badminton + tennis - (total - neither) = 21 := by
  sorry

end sports_club_overlap_l2299_229913


namespace smallest_congruent_n_l2299_229968

theorem smallest_congruent_n (a b : ℤ) (h1 : a ≡ 23 [ZMOD 60]) (h2 : b ≡ 95 [ZMOD 60]) :
  ∃ n : ℤ, 150 ≤ n ∧ n ≤ 191 ∧ a - b ≡ n [ZMOD 60] ∧
  ∀ m : ℤ, 150 ≤ m ∧ m < n → ¬(a - b ≡ m [ZMOD 60]) ∧ n = 168 := by
  sorry

end smallest_congruent_n_l2299_229968


namespace x_squared_eq_one_necessary_not_sufficient_l2299_229911

theorem x_squared_eq_one_necessary_not_sufficient (x : ℝ) :
  (x = 1 → x^2 = 1) ∧ ¬(x^2 = 1 → x = 1) := by
  sorry

end x_squared_eq_one_necessary_not_sufficient_l2299_229911


namespace parents_in_auditorium_l2299_229946

/-- Given a school play with girls and boys, and both parents of each kid attending,
    calculate the total number of parents in the auditorium. -/
theorem parents_in_auditorium (girls boys : ℕ) (h1 : girls = 6) (h2 : boys = 8) :
  2 * (girls + boys) = 28 := by
  sorry

end parents_in_auditorium_l2299_229946


namespace population_decrease_l2299_229916

theorem population_decrease (k : ℝ) (P₀ : ℝ) (n : ℕ) 
  (h1 : -1 < k) (h2 : k < 0) (h3 : P₀ > 0) : 
  P₀ * (1 + k)^(n + 1) < P₀ * (1 + k)^n := by
  sorry

#check population_decrease

end population_decrease_l2299_229916


namespace unique_pair_cube_prime_l2299_229957

theorem unique_pair_cube_prime : 
  ∃! (x y : ℕ), x > 0 ∧ y > 0 ∧ 
  ∃ (p : ℕ), Prime p ∧ (x * y^3) / (x + y) = p^3 ∧ 
  x = 2 ∧ y = 14 := by
sorry

end unique_pair_cube_prime_l2299_229957


namespace factorization_2a_squared_minus_2a_l2299_229900

theorem factorization_2a_squared_minus_2a (a : ℝ) : 2*a^2 - 2*a = 2*a*(a-1) := by
  sorry

end factorization_2a_squared_minus_2a_l2299_229900


namespace street_length_l2299_229986

theorem street_length (forest_area : ℝ) (street_area : ℝ) (trees_per_sqm : ℝ) (total_trees : ℝ) :
  forest_area = 3 * street_area →
  trees_per_sqm = 4 →
  total_trees = 120000 →
  street_area = (100 : ℝ) ^ 2 :=
by
  sorry

end street_length_l2299_229986


namespace cos_squared_alpha_plus_pi_fourth_l2299_229953

theorem cos_squared_alpha_plus_pi_fourth (α : ℝ) (h : Real.sin (2 * α) = 2/3) :
  Real.cos (α + Real.pi/4)^2 = 1/6 := by
  sorry

end cos_squared_alpha_plus_pi_fourth_l2299_229953


namespace triathlon_speed_l2299_229931

/-- Triathlon completion problem -/
theorem triathlon_speed (swim_distance : Real) (bike_distance : Real) (run_distance : Real)
  (total_time : Real) (swim_speed : Real) (run_speed : Real) :
  swim_distance = 0.5 ∧ 
  bike_distance = 20 ∧ 
  run_distance = 4 ∧ 
  total_time = 1.75 ∧ 
  swim_speed = 1 ∧ 
  run_speed = 4 →
  (bike_distance / (total_time - (swim_distance / swim_speed) - (run_distance / run_speed))) = 80 := by
  sorry

#check triathlon_speed

end triathlon_speed_l2299_229931


namespace max_value_of_complex_expression_l2299_229959

theorem max_value_of_complex_expression (w : ℂ) (h : Complex.abs w = 2) :
  Complex.abs ((w - 2)^2 * (w + 2)) ≤ 12 ∧
  ∃ w : ℂ, Complex.abs w = 2 ∧ Complex.abs ((w - 2)^2 * (w + 2)) = 12 := by
  sorry

end max_value_of_complex_expression_l2299_229959


namespace orange_crates_count_l2299_229985

theorem orange_crates_count :
  ∀ (num_crates : ℕ),
    (∀ (crate : ℕ), crate ≤ num_crates → 150 * num_crates + 16 * 30 = 2280) →
    num_crates = 12 :=
by
  sorry

end orange_crates_count_l2299_229985


namespace rahul_savings_l2299_229969

/-- Rahul's savings problem -/
theorem rahul_savings (NSC PPF : ℕ) (h1 : 3 * (NSC / 3) = 2 * (PPF / 2)) (h2 : PPF = 72000) :
  NSC + PPF = 180000 := by
  sorry

end rahul_savings_l2299_229969


namespace jerry_original_butterflies_l2299_229954

/-- The number of butterflies Jerry let go -/
def butterflies_released : ℕ := 11

/-- The number of butterflies Jerry still has -/
def butterflies_remaining : ℕ := 82

/-- The original number of butterflies Jerry had -/
def original_butterflies : ℕ := butterflies_released + butterflies_remaining

theorem jerry_original_butterflies : original_butterflies = 93 := by
  sorry

end jerry_original_butterflies_l2299_229954


namespace quadratic_factor_sum_l2299_229908

theorem quadratic_factor_sum (a w c d : ℤ) : 
  (∀ x : ℚ, 6 * x^2 + x - 12 = (a * x + w) * (c * x + d)) →
  |a| + |w| + |c| + |d| = 22 := by
sorry

end quadratic_factor_sum_l2299_229908


namespace money_needed_proof_l2299_229905

def car_wash_count : ℕ := 5
def car_wash_price : ℚ := 8.5
def dog_walk_count : ℕ := 4
def dog_walk_price : ℚ := 6.75
def lawn_mow_count : ℕ := 3
def lawn_mow_price : ℚ := 12.25
def bicycle_price : ℚ := 150.25
def helmet_price : ℚ := 35.75
def lock_price : ℚ := 24.5

def total_money_made : ℚ := 
  car_wash_count * car_wash_price + 
  dog_walk_count * dog_walk_price + 
  lawn_mow_count * lawn_mow_price

def total_cost : ℚ := 
  bicycle_price + helmet_price + lock_price

theorem money_needed_proof : 
  total_cost - total_money_made = 104.25 := by sorry

end money_needed_proof_l2299_229905


namespace enrollment_theorem_l2299_229944

-- Define the schools and their enrollments
def schools : Fin 4 → ℕ
| 0 => 1300  -- Varsity
| 1 => 1500  -- Northwest
| 2 => 1800  -- Central
| 3 => 1600  -- Greenbriar
| _ => 0     -- This case should never occur

-- Calculate the average enrollment
def average_enrollment : ℚ := (schools 0 + schools 1 + schools 2 + schools 3) / 4

-- Calculate the positive difference between a school's enrollment and the average
def positive_difference (i : Fin 4) : ℚ := |schools i - average_enrollment|

-- Theorem stating the average enrollment and positive differences
theorem enrollment_theorem :
  average_enrollment = 1550 ∧
  positive_difference 0 = 250 ∧
  positive_difference 1 = 50 ∧
  positive_difference 2 = 250 ∧
  positive_difference 3 = 50 :=
by sorry

end enrollment_theorem_l2299_229944


namespace xiaolin_mean_calculation_l2299_229904

theorem xiaolin_mean_calculation 
  (a b c : ℝ) 
  (h1 : a < b) 
  (h2 : b < c) 
  (x : ℝ) 
  (hx : x = (a + b) / 2) 
  (y : ℝ) 
  (hy : y = (x + c) / 2) : 
  y < (a + b + c) / 3 := by
sorry

end xiaolin_mean_calculation_l2299_229904


namespace alice_sales_above_quota_l2299_229993

def alice_quota : ℕ := 2000

def shoe_prices : List (String × ℕ) := [
  ("Adidas", 45),
  ("Nike", 60),
  ("Reeboks", 35),
  ("Puma", 50),
  ("Converse", 40)
]

def sales : List (String × ℕ) := [
  ("Nike", 12),
  ("Adidas", 10),
  ("Reeboks", 15),
  ("Puma", 8),
  ("Converse", 14)
]

def total_sales : ℕ := (sales.map (fun (s : String × ℕ) =>
  match shoe_prices.find? (fun (p : String × ℕ) => p.1 = s.1) with
  | some price => s.2 * price.2
  | none => 0
)).sum

theorem alice_sales_above_quota :
  total_sales - alice_quota = 655 := by sorry

end alice_sales_above_quota_l2299_229993
