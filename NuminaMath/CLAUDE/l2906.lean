import Mathlib

namespace prob_king_jack_queen_value_l2906_290683

/-- A standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Number of Jacks in a standard deck -/
def NumJacks : ℕ := 4

/-- Number of Queens in a standard deck -/
def NumQueens : ℕ := 4

/-- Probability of drawing a King, then a Jack, then a Queen from a standard deck without replacement -/
def prob_king_jack_queen : ℚ :=
  (NumKings : ℚ) / StandardDeck *
  (NumJacks : ℚ) / (StandardDeck - 1) *
  (NumQueens : ℚ) / (StandardDeck - 2)

theorem prob_king_jack_queen_value :
  prob_king_jack_queen = 8 / 16575 := by
  sorry

end prob_king_jack_queen_value_l2906_290683


namespace medicine_price_proof_l2906_290679

/-- Proves that the original price of a medicine is $150 given the specified conditions --/
theorem medicine_price_proof (cashback_rate : Real) (rebate : Real) (final_cost : Real) :
  cashback_rate = 0.1 →
  rebate = 25 →
  final_cost = 110 →
  ∃ (original_price : Real),
    original_price - (cashback_rate * original_price + rebate) = final_cost ∧
    original_price = 150 := by
  sorry

#check medicine_price_proof

end medicine_price_proof_l2906_290679


namespace hyperbola_eccentricity_l2906_290682

/-- Given a hyperbola with the equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if its focus (c, 0) is symmetric about the asymptote y = (b/a)x and 
    the symmetric point of the focus lies on the other asymptote y = -(b/a)x,
    then its eccentricity is 2. -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ x y : ℝ, y = (b/a) * x ∧ (x - c)^2 + y^2 = (x + c)^2 + y^2) →
  (∃ x y : ℝ, y = -(b/a) * x ∧ (x + c)^2 + y^2 = 0) →
  c / a = 2 :=
by sorry

end hyperbola_eccentricity_l2906_290682


namespace garden_breadth_l2906_290694

/-- 
Given a rectangular garden with perimeter 800 meters and length 300 meters,
prove that its breadth is 100 meters.
-/
theorem garden_breadth (perimeter length breadth : ℝ) 
  (h1 : perimeter = 800)
  (h2 : length = 300)
  (h3 : perimeter = 2 * (length + breadth)) : 
  breadth = 100 := by
  sorry

end garden_breadth_l2906_290694


namespace initial_bureaus_correct_l2906_290650

/-- The number of offices -/
def num_offices : ℕ := 14

/-- The additional bureaus needed for equal distribution -/
def additional_bureaus : ℕ := 10

/-- The initial number of bureaus -/
def initial_bureaus : ℕ := 8

/-- Theorem stating that the initial number of bureaus is correct -/
theorem initial_bureaus_correct :
  ∃ (x : ℕ), (initial_bureaus + additional_bureaus = num_offices * x) ∧
             (∀ y : ℕ, initial_bureaus ≠ num_offices * y) :=
by sorry

end initial_bureaus_correct_l2906_290650


namespace intersection_dot_product_l2906_290687

-- Define the line l: 4x + 3y - 5 = 0
def line_l (x y : ℝ) : Prop := 4 * x + 3 * y - 5 = 0

-- Define the circle C: x² + y² - 4 = 0
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0

-- Define the intersection points A and B
def is_intersection (x y : ℝ) : Prop := line_l x y ∧ circle_C x y

-- Define the origin O
def origin : ℝ × ℝ := (0, 0)

-- Theorem statement
theorem intersection_dot_product :
  ∃ (A B : ℝ × ℝ),
    is_intersection A.1 A.2 ∧
    is_intersection B.1 B.2 ∧
    A ≠ B ∧
    (A.1 * B.1 + A.2 * B.2 = -2) :=
sorry

end intersection_dot_product_l2906_290687


namespace house_size_problem_l2906_290632

theorem house_size_problem (sara_house nada_house : ℝ) : 
  sara_house = 1000 ∧ 
  sara_house = 2 * nada_house + 100 → 
  nada_house = 450 := by
sorry

end house_size_problem_l2906_290632


namespace exists_number_divisible_by_digit_sum_l2906_290684

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate to check if all digits of a number are non-zero -/
def all_digits_nonzero (n : ℕ) : Prop := sorry

/-- Number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Theorem: For all positive integers n, there exists an n-digit number z
    such that none of its digits are 0 and z is divisible by the sum of its digits -/
theorem exists_number_divisible_by_digit_sum :
  ∀ n : ℕ, n > 0 → ∃ z : ℕ,
    num_digits z = n ∧
    all_digits_nonzero z ∧
    z % sum_of_digits z = 0 :=
by sorry

end exists_number_divisible_by_digit_sum_l2906_290684


namespace min_value_theorem_l2906_290614

-- Define the condition function
def condition (a b : ℝ) : Prop :=
  ∀ x : ℝ, (Real.log a + b) * Real.exp x - a^2 * Real.exp x * x ≥ 0

-- State the theorem
theorem min_value_theorem (a b : ℝ) (h : condition a b) : 
  ∃ (min : ℝ), min = 1 ∧ ∀ (c : ℝ), b / a ≥ c := by sorry

end min_value_theorem_l2906_290614


namespace number_of_arrangements_l2906_290653

/-- Represents a person in the group photo --/
inductive Person
  | StudentA
  | StudentB
  | StudentC
  | StudentD
  | StudentE
  | TeacherX
  | TeacherY

/-- Represents a valid arrangement of people in the group photo --/
def ValidArrangement : Type := List Person

/-- Checks if students A, B, and C are standing together in the arrangement --/
def studentsABCTogether (arrangement : ValidArrangement) : Prop := sorry

/-- Checks if teachers X and Y are not standing next to each other in the arrangement --/
def teachersNotAdjacent (arrangement : ValidArrangement) : Prop := sorry

/-- The set of all valid arrangements satisfying the given conditions --/
def validArrangements : Set ValidArrangement :=
  {arrangement | studentsABCTogether arrangement ∧ teachersNotAdjacent arrangement}

/-- The main theorem stating that the number of valid arrangements is 504 --/
theorem number_of_arrangements (h : Fintype validArrangements) :
  Fintype.card validArrangements = 504 := by sorry

end number_of_arrangements_l2906_290653


namespace cost_difference_l2906_290643

/-- The price difference between two types of candy in kopecks per kilogram -/
def price_difference : ℕ := 80

/-- The total amount of candy bought by each person in grams -/
def total_amount : ℕ := 150

/-- The cost of Andrey's purchase in kopecks -/
def andrey_cost (x : ℕ) : ℚ :=
  (150 * x + 8000 : ℚ) / 1000

/-- The cost of Yura's purchase in kopecks -/
def yura_cost (x : ℕ) : ℚ :=
  (150 * x + 6000 : ℚ) / 1000

/-- The theorem stating the difference in cost between Andrey's and Yura's purchases -/
theorem cost_difference (x : ℕ) :
  andrey_cost x - yura_cost x = 2 / 1000 := by sorry

end cost_difference_l2906_290643


namespace solution_set_nonempty_iff_m_in_range_inequality_holds_for_interval_iff_m_in_range_l2906_290601

-- Part 1
theorem solution_set_nonempty_iff_m_in_range (m : ℝ) :
  (∃ x : ℝ, m * x^2 + m * x + m - 6 < 0) ↔ m < 8 :=
sorry

-- Part 2
theorem inequality_holds_for_interval_iff_m_in_range (m : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-2) 1 → m * x^2 - m * x < -m + 2) ↔ m < 2/7 :=
sorry

end solution_set_nonempty_iff_m_in_range_inequality_holds_for_interval_iff_m_in_range_l2906_290601


namespace willies_stickers_l2906_290680

/-- Willie's sticker problem -/
theorem willies_stickers (initial_stickers given_away : ℕ) 
  (h1 : initial_stickers = 36)
  (h2 : given_away = 7) :
  initial_stickers - given_away = 29 := by
  sorry

end willies_stickers_l2906_290680


namespace basketball_purchase_theorem_l2906_290652

/-- Represents the prices and quantities of basketballs --/
structure BasketballPurchase where
  priceA : ℕ  -- Price of brand A basketball
  priceB : ℕ  -- Price of brand B basketball
  quantityA : ℕ  -- Quantity of brand A basketballs
  quantityB : ℕ  -- Quantity of brand B basketballs

/-- Represents the conditions of the basketball purchase problem --/
def BasketballProblem (p : BasketballPurchase) : Prop :=
  p.priceB = p.priceA + 40 ∧
  4800 / p.priceA = (3/2) * (4000 / p.priceB) ∧
  p.quantityA + p.quantityB = 90 ∧
  p.quantityB ≥ 2 * p.quantityA ∧
  p.priceA * p.quantityA + p.priceB * p.quantityB ≤ 17200

/-- The theorem to be proved --/
theorem basketball_purchase_theorem (p : BasketballPurchase) 
  (h : BasketballProblem p) : 
  p.priceA = 160 ∧ 
  p.priceB = 200 ∧ 
  (∃ n : ℕ, n = 11 ∧ 
    ∀ m : ℕ, (20 ≤ m ∧ m ≤ 30) ↔ 
      BasketballProblem ⟨p.priceA, p.priceB, m, 90 - m⟩) ∧
  (∀ a : ℕ, 30 < a ∧ a < 50 → 
    (a < 40 → p.quantityA = 30) ∧ 
    (a > 40 → p.quantityA = 20)) :=
sorry


end basketball_purchase_theorem_l2906_290652


namespace increased_speed_calculation_l2906_290606

/-- Proves that given a distance of 100 km, a usual speed of 20 km/hr,
    and a travel time reduction of 1 hour with increased speed,
    the increased speed is 25 km/hr. -/
theorem increased_speed_calculation (distance : ℝ) (usual_speed : ℝ) (time_reduction : ℝ) :
  distance = 100 ∧ usual_speed = 20 ∧ time_reduction = 1 →
  (distance / (distance / usual_speed - time_reduction)) = 25 := by
  sorry

end increased_speed_calculation_l2906_290606


namespace temperature_difference_product_of_N_values_l2906_290610

theorem temperature_difference (B : ℝ) (N : ℝ) : 
  (∃ A : ℝ, A = B + N) → 
  (|((B + N) - 4) - (B + 5)| = 1) →
  (N = 10 ∨ N = 8) :=
by sorry

theorem product_of_N_values :
  ∃ N₁ N₂ : ℝ, 
    (∃ B : ℝ, (∃ A : ℝ, A = B + N₁) ∧ |((B + N₁) - 4) - (B + 5)| = 1) ∧
    (∃ B : ℝ, (∃ A : ℝ, A = B + N₂) ∧ |((B + N₂) - 4) - (B + 5)| = 1) ∧
    N₁ * N₂ = 80 :=
by sorry

end temperature_difference_product_of_N_values_l2906_290610


namespace final_deleted_files_l2906_290617

def deleted_pictures : ℕ := 5
def deleted_songs : ℕ := 12
def deleted_text_files : ℕ := 10
def deleted_video_files : ℕ := 6
def restored_pictures : ℕ := 3
def restored_video_files : ℕ := 4

theorem final_deleted_files :
  deleted_pictures + deleted_songs + deleted_text_files + deleted_video_files
  - (restored_pictures + restored_video_files) = 26 := by
  sorry

end final_deleted_files_l2906_290617


namespace division_problem_l2906_290688

theorem division_problem : ∃ (dividend : Nat) (divisor : Nat),
  dividend = 10004678 ∧ 
  divisor = 142 ∧ 
  100 ≤ divisor ∧ 
  divisor < 1000 ∧
  10000000 ≤ dividend ∧ 
  dividend < 100000000 ∧
  dividend / divisor = 70709 := by
  sorry

end division_problem_l2906_290688


namespace girls_tryout_count_l2906_290678

theorem girls_tryout_count (boys : ℕ) (called_back : ℕ) (didnt_make_cut : ℕ) :
  boys = 4 →
  called_back = 26 →
  didnt_make_cut = 17 →
  ∃ girls : ℕ, girls + boys = called_back + didnt_make_cut ∧ girls = 39 :=
by
  sorry

end girls_tryout_count_l2906_290678


namespace min_words_for_passing_score_l2906_290642

/-- Represents the German vocabulary exam parameters and conditions -/
structure GermanExam where
  total_words : ℕ := 800
  correct_points : ℚ := 1
  incorrect_penalty : ℚ := 1/4
  target_score_percentage : ℚ := 90/100

/-- Calculates the exam score based on the number of words learned -/
def examScore (exam : GermanExam) (words_learned : ℕ) : ℚ :=
  (words_learned : ℚ) * exam.correct_points - 
  ((exam.total_words - words_learned) : ℚ) * exam.incorrect_penalty

/-- Theorem stating that learning at least 736 words ensures a score of at least 90% -/
theorem min_words_for_passing_score (exam : GermanExam) :
  ∀ words_learned : ℕ, words_learned ≥ 736 →
  examScore exam words_learned ≥ (exam.target_score_percentage * exam.total_words) :=
by sorry

#check min_words_for_passing_score

end min_words_for_passing_score_l2906_290642


namespace smallest_positive_shift_l2906_290697

-- Define a function f with period 20
def f : ℝ → ℝ := sorry

-- Define the periodicity property
axiom f_periodic : ∀ x : ℝ, f (x - 20) = f x

-- Define the property for the scaled and shifted function
def scaled_shifted_property (a : ℝ) : Prop :=
  ∀ x : ℝ, f ((x - a) / 4) = f (x / 4)

-- Theorem statement
theorem smallest_positive_shift :
  ∃ a : ℝ, a > 0 ∧ scaled_shifted_property a ∧
  ∀ b : ℝ, b > 0 ∧ scaled_shifted_property b → a ≤ b :=
sorry

end smallest_positive_shift_l2906_290697


namespace P_equals_Q_l2906_290604

-- Define set P
def P : Set ℝ := {m : ℝ | -1 < m ∧ m ≤ 0}

-- Define set Q
def Q : Set ℝ := {m : ℝ | ∀ x : ℝ, m * x^2 + 4 * m * x - 4 < 0}

-- Theorem statement
theorem P_equals_Q : P = Q := by sorry

end P_equals_Q_l2906_290604


namespace items_not_washed_l2906_290624

theorem items_not_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (pants : ℕ) (jackets : ℕ) (washed : ℕ)
  (h1 : short_sleeve = 9)
  (h2 : long_sleeve = 21)
  (h3 : pants = 15)
  (h4 : jackets = 8)
  (h5 : washed = 43) :
  short_sleeve + long_sleeve + pants + jackets - washed = 10 := by
  sorry

end items_not_washed_l2906_290624


namespace luncheon_cost_theorem_l2906_290689

/-- The cost of items in a luncheon -/
structure LuncheonCost where
  sandwich : ℝ
  coffee : ℝ
  pie : ℝ
  cookie : ℝ

/-- Given conditions of the luncheon costs -/
def luncheon_conditions (cost : LuncheonCost) : Prop :=
  5 * cost.sandwich + 9 * cost.coffee + 2 * cost.pie + 3 * cost.cookie = 5.85 ∧
  6 * cost.sandwich + 12 * cost.coffee + 2 * cost.pie + 4 * cost.cookie = 7.20

/-- Theorem stating the cost of one of each item -/
theorem luncheon_cost_theorem (cost : LuncheonCost) :
  luncheon_conditions cost →
  cost.sandwich + cost.coffee + cost.pie + cost.cookie = 1.35 :=
by sorry

end luncheon_cost_theorem_l2906_290689


namespace novel_writing_stats_l2906_290600

-- Define the given conditions
def total_words : ℕ := 50000
def total_hours : ℕ := 100
def hours_per_day : ℕ := 5

-- Theorem to prove
theorem novel_writing_stats :
  (total_words / total_hours = 500) ∧
  (total_hours / hours_per_day = 20) := by
  sorry

end novel_writing_stats_l2906_290600


namespace frank_max_average_time_l2906_290663

/-- The maximum average time per maze Frank wants to maintain -/
def maxAverageTime (previousMazes : ℕ) (averagePreviousTime : ℕ) (currentTime : ℕ) (remainingTime : ℕ) : ℚ :=
  let totalPreviousTime := previousMazes * averagePreviousTime
  let totalCurrentTime := currentTime + remainingTime
  let totalTime := totalPreviousTime + totalCurrentTime
  let totalMazes := previousMazes + 1
  totalTime / totalMazes

/-- Theorem stating the maximum average time Frank wants to maintain -/
theorem frank_max_average_time :
  maxAverageTime 4 50 45 55 = 60 := by
  sorry

end frank_max_average_time_l2906_290663


namespace boys_clay_maple_basketball_l2906_290699

/-- Represents a school in the sports camp -/
inductive School
| Jonas
| Clay
| Maple

/-- Represents an activity in the sports camp -/
inductive Activity
| Basketball
| Swimming

/-- Represents the gender of a student -/
inductive Gender
| Boy
| Girl

/-- Data about the sports camp -/
structure SportsData where
  total_students : ℕ
  total_boys : ℕ
  total_girls : ℕ
  jonas_students : ℕ
  clay_students : ℕ
  maple_students : ℕ
  jonas_boys : ℕ
  swimming_girls : ℕ
  clay_swimming_boys : ℕ

/-- Theorem stating the number of boys from Clay and Maple who attended basketball -/
theorem boys_clay_maple_basketball (data : SportsData)
  (h1 : data.total_students = 120)
  (h2 : data.total_boys = 70)
  (h3 : data.total_girls = 50)
  (h4 : data.jonas_students = 50)
  (h5 : data.clay_students = 40)
  (h6 : data.maple_students = 30)
  (h7 : data.jonas_boys = 28)
  (h8 : data.swimming_girls = 16)
  (h9 : data.clay_swimming_boys = 10) :
  (data.total_boys - data.jonas_boys - data.clay_swimming_boys) = 30 := by
  sorry

end boys_clay_maple_basketball_l2906_290699


namespace constant_expression_l2906_290686

-- Define the logarithm with base √2
noncomputable def log_sqrt2 (x : ℝ) : ℝ := Real.log x / Real.log (Real.sqrt 2)

-- State the theorem
theorem constant_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x^2 + y^2 = 18*x*y) :
  log_sqrt2 (x - y) - (log_sqrt2 x + log_sqrt2 y) / 2 = 4 := by
  sorry

end constant_expression_l2906_290686


namespace clerical_staff_percentage_l2906_290657

def total_employees : ℕ := 3600
def initial_clerical_ratio : ℚ := 1 / 6
def clerical_reduction_ratio : ℚ := 1 / 4

theorem clerical_staff_percentage : 
  let initial_clerical := (initial_clerical_ratio * total_employees : ℚ)
  let reduced_clerical := initial_clerical - (clerical_reduction_ratio * initial_clerical)
  let remaining_employees := total_employees - (initial_clerical - reduced_clerical)
  (reduced_clerical / remaining_employees) * 100 = 450 / 3450 * 100 := by
  sorry

end clerical_staff_percentage_l2906_290657


namespace problem_solution_l2906_290668

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | x^2 - x - a = 0}
def B : Set ℝ := {2, -5}

-- Define the theorem
theorem problem_solution :
  ∃ (a : ℝ),
    (2 ∈ A a) ∧
    (a = 2) ∧
    (A a = {-1, 2}) ∧
    (let U := A a ∪ B;
     U = {-5, -1, 2} ∧
     (U \ A a) ∪ (U \ B) = {-5, -1}) :=
by
  sorry


end problem_solution_l2906_290668


namespace sunshine_orchard_pumpkins_l2906_290609

theorem sunshine_orchard_pumpkins (moonglow_pumpkins : ℕ) (sunshine_pumpkins : ℕ) : 
  moonglow_pumpkins = 14 →
  sunshine_pumpkins = 3 * moonglow_pumpkins + 12 →
  sunshine_pumpkins = 54 := by
  sorry

end sunshine_orchard_pumpkins_l2906_290609


namespace stating_triangle_field_q_formula_l2906_290674

/-- Represents a right-angled triangle field with two people walking along its edges -/
structure TriangleField where
  /-- Length of LM -/
  r : ℝ
  /-- Length of LX -/
  p : ℝ
  /-- Length of XN -/
  q : ℝ
  /-- r is positive -/
  r_pos : r > 0
  /-- p is positive -/
  p_pos : p > 0
  /-- q is positive -/
  q_pos : q > 0
  /-- LMN is a right-angled triangle -/
  right_angle : (p + q)^2 + r^2 = (p + r - q)^2

/-- 
Theorem stating that in a TriangleField, the length q can be expressed as pr / (2p + r)
-/
theorem triangle_field_q_formula (tf : TriangleField) : tf.q = (tf.p * tf.r) / (2 * tf.p + tf.r) := by
  sorry

end stating_triangle_field_q_formula_l2906_290674


namespace decimal_sum_to_fraction_l2906_290681

theorem decimal_sum_to_fraction :
  (0.1 : ℚ) + 0.02 + 0.003 + 0.0004 + 0.00005 = 2469 / 20000 := by
  sorry

end decimal_sum_to_fraction_l2906_290681


namespace remainder_sum_l2906_290676

theorem remainder_sum (a b : ℤ) 
  (ha : a % 80 = 75) 
  (hb : b % 90 = 85) : 
  (a + b) % 40 = 0 := by
sorry

end remainder_sum_l2906_290676


namespace sqrt_30_bounds_l2906_290685

theorem sqrt_30_bounds : 5 < Real.sqrt 30 ∧ Real.sqrt 30 < 6 := by sorry

end sqrt_30_bounds_l2906_290685


namespace smallest_w_l2906_290665

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : 
  w > 0 ∧ 
  is_factor (2^5) (936 * w) ∧ 
  is_factor (3^3) (936 * w) ∧ 
  is_factor (10^2) (936 * w) →
  w ≥ 300 ∧ 
  ∃ (v : ℕ), v = 300 ∧ 
    v > 0 ∧ 
    is_factor (2^5) (936 * v) ∧ 
    is_factor (3^3) (936 * v) ∧ 
    is_factor (10^2) (936 * v) :=
by sorry

end smallest_w_l2906_290665


namespace comic_book_stacking_permutations_l2906_290644

theorem comic_book_stacking_permutations :
  let spiderman_books : ℕ := 7
  let archie_books : ℕ := 4
  let garfield_books : ℕ := 5
  let batman_books : ℕ := 3
  let total_books : ℕ := spiderman_books + archie_books + garfield_books + batman_books
  let non_batman_types : ℕ := 3  -- Spiderman, Archie, and Garfield

  (spiderman_books.factorial * archie_books.factorial * garfield_books.factorial * batman_books.factorial) *
  non_batman_types.factorial = 55085760 :=
by
  sorry

end comic_book_stacking_permutations_l2906_290644


namespace track_width_l2906_290635

theorem track_width (r₁ r₂ : ℝ) (h : r₁ > r₂) :
  2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi →
  r₁ - r₂ = 10 := by
  sorry

end track_width_l2906_290635


namespace solve_linear_equation_l2906_290655

theorem solve_linear_equation (m : ℝ) (x : ℝ) : 
  (m * x + 1 = 2) → (x = -1) → (m = -1) := by
  sorry

end solve_linear_equation_l2906_290655


namespace opposite_of_three_l2906_290647

theorem opposite_of_three (a : ℝ) : (2 * a + 3) + 3 = 0 → a = -3 := by
  sorry

end opposite_of_three_l2906_290647


namespace intersection_M_N_l2906_290628

-- Define set M
def M : Set ℝ := {x | x^2 ≥ x}

-- Define set N
def N : Set ℝ := {y | ∃ x, y = 3^x + 1}

-- Theorem statement
theorem intersection_M_N : M ∩ N = {x | x > 1} := by
  sorry

end intersection_M_N_l2906_290628


namespace division_problem_l2906_290616

theorem division_problem (smaller larger : ℕ) : 
  larger - smaller = 1395 →
  larger = 1656 →
  larger % smaller = 15 →
  larger / smaller = 6 := by
sorry

end division_problem_l2906_290616


namespace pentagon_triangles_l2906_290630

/-- The number of triangles formed in a pentagon when drawing diagonals from one vertex --/
def triangles_in_pentagon : ℕ := 3

/-- The number of vertices in a pentagon --/
def pentagon_vertices : ℕ := 5

/-- The number of diagonals drawn from one vertex in a pentagon --/
def diagonals_from_vertex : ℕ := 2

theorem pentagon_triangles :
  triangles_in_pentagon = diagonals_from_vertex + 1 :=
by sorry

end pentagon_triangles_l2906_290630


namespace latte_price_calculation_l2906_290670

-- Define the prices and quantities
def total_cost : ℚ := 25
def drip_coffee_price : ℚ := 2.25
def drip_coffee_quantity : ℕ := 2
def espresso_price : ℚ := 3.50
def espresso_quantity : ℕ := 1
def latte_quantity : ℕ := 2
def vanilla_syrup_price : ℚ := 0.50
def vanilla_syrup_quantity : ℕ := 1
def cold_brew_price : ℚ := 2.50
def cold_brew_quantity : ℕ := 2
def cappuccino_price : ℚ := 3.50
def cappuccino_quantity : ℕ := 1

-- Define the theorem
theorem latte_price_calculation :
  ∃ (latte_price : ℚ),
    latte_price * latte_quantity +
    drip_coffee_price * drip_coffee_quantity +
    espresso_price * espresso_quantity +
    vanilla_syrup_price * vanilla_syrup_quantity +
    cold_brew_price * cold_brew_quantity +
    cappuccino_price * cappuccino_quantity = total_cost ∧
    latte_price = 4 := by
  sorry

end latte_price_calculation_l2906_290670


namespace franklin_valentines_l2906_290671

/-- The number of Valentines Mrs. Franklin gave away -/
def valentines_given : ℕ := 42

/-- The number of Valentines Mrs. Franklin has left -/
def valentines_left : ℕ := 16

/-- The initial number of Valentines Mrs. Franklin had -/
def initial_valentines : ℕ := valentines_given + valentines_left

theorem franklin_valentines : initial_valentines = 58 := by
  sorry

end franklin_valentines_l2906_290671


namespace total_rainfall_proof_l2906_290667

/-- Given rainfall data for three days, proves the total rainfall. -/
theorem total_rainfall_proof (sunday_rain : ℝ) (monday_rain : ℝ) (tuesday_rain : ℝ)
  (h1 : sunday_rain = 4)
  (h2 : monday_rain = sunday_rain + 3)
  (h3 : tuesday_rain = 2 * monday_rain) :
  sunday_rain + monday_rain + tuesday_rain = 25 := by
  sorry

end total_rainfall_proof_l2906_290667


namespace arithmetic_calculation_l2906_290634

theorem arithmetic_calculation : 4 * 6 * 8 + 18 / 3^2 = 194 := by
  sorry

end arithmetic_calculation_l2906_290634


namespace degree_not_determined_by_characteristic_l2906_290613

/-- A type representing a characteristic of a polynomial -/
def PolynomialCharacteristic := Type

/-- A function that computes a characteristic of a polynomial -/
noncomputable def compute_characteristic (P : Polynomial ℝ) : PolynomialCharacteristic :=
  sorry

/-- Theorem stating that the degree of a polynomial cannot be uniquely determined from its characteristic -/
theorem degree_not_determined_by_characteristic :
  ∃ (P1 P2 : Polynomial ℝ), 
    P1.degree ≠ P2.degree ∧ 
    compute_characteristic P1 = compute_characteristic P2 := by
  sorry

end degree_not_determined_by_characteristic_l2906_290613


namespace function_identity_proof_l2906_290691

theorem function_identity_proof (f : ℕ+ → ℕ+) 
  (h : ∀ (m n : ℕ+), (f m)^2 + f n ∣ (m^2 + n)^2) : 
  ∀ (n : ℕ+), f n = n := by
  sorry

end function_identity_proof_l2906_290691


namespace sin_less_than_x_in_interval_exp_x_plus_one_greater_than_neg_e_squared_l2906_290690

-- Option A
theorem sin_less_than_x_in_interval (x : ℝ) (h : x ∈ Set.Ioo 0 Real.pi) : x > Real.sin x := by
  sorry

-- Option C
theorem exp_x_plus_one_greater_than_neg_e_squared (x : ℝ) : (x + 1) * Real.exp x > -(1 / Real.exp 2) := by
  sorry

end sin_less_than_x_in_interval_exp_x_plus_one_greater_than_neg_e_squared_l2906_290690


namespace work_time_ratio_l2906_290664

/-- Proves that the ratio of Celeste's work time to Bianca's work time is 2:1 given the specified conditions. -/
theorem work_time_ratio (bianca_time : ℝ) (celeste_multiplier : ℝ) :
  bianca_time = 12.5 →
  bianca_time * celeste_multiplier + (bianca_time * celeste_multiplier - 8.5) + bianca_time = 54 →
  celeste_multiplier = 2 := by
  sorry

#check work_time_ratio

end work_time_ratio_l2906_290664


namespace second_trial_point_theorem_l2906_290619

/-- Represents the fractional method for optimization experiments -/
structure FractionalMethod where
  range_start : ℝ
  range_end : ℝ
  rounds : ℕ

/-- Calculates the possible second trial points for the fractional method -/
def second_trial_points (fm : FractionalMethod) : Set ℝ :=
  let interval_length := fm.range_end - fm.range_start
  let num_divisions := 2^fm.rounds
  let step := interval_length / num_divisions
  {fm.range_start + 3 * step, fm.range_end - 3 * step}

/-- Theorem stating that for the given experimental setup, 
    the second trial point is either 40 or 60 -/
theorem second_trial_point_theorem (fm : FractionalMethod) 
  (h1 : fm.range_start = 10) 
  (h2 : fm.range_end = 90) 
  (h3 : fm.rounds = 4) : 
  second_trial_points fm = {40, 60} := by
  sorry

end second_trial_point_theorem_l2906_290619


namespace cubic_polynomials_common_roots_l2906_290660

theorem cubic_polynomials_common_roots (c d : ℝ) : 
  (∃ u v : ℝ, u ≠ v ∧ 
    u^3 + c*u^2 + 10*u + 4 = 0 ∧ 
    u^3 + d*u^2 + 13*u + 5 = 0 ∧
    v^3 + c*v^2 + 10*v + 4 = 0 ∧ 
    v^3 + d*v^2 + 13*v + 5 = 0) → 
  c = 7 ∧ d = 8 := by
sorry

end cubic_polynomials_common_roots_l2906_290660


namespace candied_yams_order_l2906_290636

theorem candied_yams_order (total_shoppers : ℕ) (buying_frequency : ℕ) (packages_per_box : ℕ) : 
  total_shoppers = 375 →
  buying_frequency = 3 →
  packages_per_box = 25 →
  (total_shoppers / buying_frequency) / packages_per_box = 5 := by
  sorry

end candied_yams_order_l2906_290636


namespace equation_solution_l2906_290602

theorem equation_solution :
  ∃! x : ℚ, (4 * x^2 + 3 * x + 2) / (x - 2) = 4 * x + 2 :=
by
  use (-2/3)
  sorry

end equation_solution_l2906_290602


namespace max_time_digit_sum_l2906_290675

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours < 24
  minute_valid : minutes < 60

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a given time -/
def timeDigitSum (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The theorem stating the maximum sum of digits in a 24-hour time display -/
theorem max_time_digit_sum :
  (∃ (t : Time24), ∀ (t' : Time24), timeDigitSum t' ≤ timeDigitSum t) ∧
  (∀ (t : Time24), timeDigitSum t ≤ 24) :=
sorry

end max_time_digit_sum_l2906_290675


namespace volume_to_surface_area_ratio_l2906_290608

/-- Represents a shape created by joining seven unit cubes -/
structure SevenCubeShape where
  /-- The volume of the shape in cubic units -/
  volume : ℕ
  /-- The surface area of the shape in square units -/
  surface_area : ℕ
  /-- The shape is composed of seven unit cubes -/
  is_seven_cubes : volume = 7
  /-- The surface area is calculated based on the configuration of the seven cubes -/
  surface_area_calc : surface_area = 30

/-- Theorem stating that the ratio of volume to surface area for the SevenCubeShape is 7:30 -/
theorem volume_to_surface_area_ratio (shape : SevenCubeShape) :
  (shape.volume : ℚ) / shape.surface_area = 7 / 30 := by
  sorry

end volume_to_surface_area_ratio_l2906_290608


namespace no_real_solutions_l2906_290626

theorem no_real_solutions : ∀ x : ℝ, ¬(Real.sqrt (9 - 3*x) = x * Real.sqrt (9 - 9*x)) := by
  sorry

end no_real_solutions_l2906_290626


namespace hyperbola_foci_distance_l2906_290656

/-- A hyperbola with given asymptotes and a point it passes through -/
structure Hyperbola where
  /-- First asymptote: y = 2x + 3 -/
  asymptote1 : ℝ → ℝ
  /-- Second asymptote: y = -2x - 1 -/
  asymptote2 : ℝ → ℝ
  /-- The hyperbola passes through this point -/
  point : ℝ × ℝ
  /-- The first asymptote has the form y = 2x + 3 -/
  h₁ : ∀ x, asymptote1 x = 2 * x + 3
  /-- The second asymptote has the form y = -2x - 1 -/
  h₂ : ∀ x, asymptote2 x = -2 * x - 1
  /-- The point (4, 5) lies on the hyperbola -/
  h₃ : point = (4, 5)

/-- The distance between the foci of the hyperbola is 6√2 -/
theorem hyperbola_foci_distance (h : Hyperbola) : 
  ∃ (f₁ f₂ : ℝ × ℝ), (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 72 := by
  sorry

end hyperbola_foci_distance_l2906_290656


namespace conic_is_ellipse_l2906_290646

-- Define the equation
def conic_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + (y-2)^2) + Real.sqrt ((x-6)^2 + (y+4)^2) = 14

-- Theorem statement
theorem conic_is_ellipse :
  ∃ (a b c d e f : ℝ), 
    (∀ x y : ℝ, conic_equation x y ↔ a*x^2 + b*y^2 + c*x*y + d*x + e*y + f = 0) ∧
    b^2 * c^2 - 4 * a * b * (a * e^2 + b * d^2 - c * d * e + f * (c^2 - 4 * a * b)) > 0 ∧
    a + b ≠ 0 ∧ a * b > 0 :=
sorry

end conic_is_ellipse_l2906_290646


namespace odd_function_properties_l2906_290623

noncomputable def f (a b x : ℝ) : ℝ := (-2^x + b) / (2^(x+1) + a)

theorem odd_function_properties (a b : ℝ) :
  (∀ x : ℝ, f a b x = -f a b (-x)) →
  (a = 2 ∧ b = 1) ∧
  (∀ x y : ℝ, x < y → f 2 1 x > f 2 1 y) ∧
  (∀ k : ℝ, (∀ x : ℝ, x ≥ 1 → f 2 1 (k * 3^x) + f 2 1 (3^x - 9^x + 2) > 0) ↔ k < 4/3) :=
by sorry

end odd_function_properties_l2906_290623


namespace calculate_units_produced_l2906_290622

/-- Given fixed cost, marginal cost, and total cost, calculate the number of units produced. -/
theorem calculate_units_produced 
  (fixed_cost : ℝ) 
  (marginal_cost : ℝ) 
  (total_cost : ℝ) 
  (h1 : fixed_cost = 12000)
  (h2 : marginal_cost = 200)
  (h3 : total_cost = 16000) :
  (total_cost - fixed_cost) / marginal_cost = 20 :=
by sorry

end calculate_units_produced_l2906_290622


namespace greatest_integer_less_than_negative_22_over_3_l2906_290693

theorem greatest_integer_less_than_negative_22_over_3 :
  ⌊-22 / 3⌋ = -8 := by sorry

end greatest_integer_less_than_negative_22_over_3_l2906_290693


namespace left_square_side_length_l2906_290666

/-- Given three squares with specific relationships between their side lengths,
    prove that the side length of the left square is 8 cm. -/
theorem left_square_side_length (x y z : ℝ) 
  (sum_condition : x + y + z = 52)
  (middle_square_condition : y = x + 17)
  (right_square_condition : z = y - 6) :
  x = 8 := by
  sorry

end left_square_side_length_l2906_290666


namespace black_hen_day_probability_l2906_290603

/-- Represents the color of a hen -/
inductive HenColor
| Black
| White

/-- Represents a program type -/
inductive ProgramType
| Day
| Evening

/-- Represents the state of available spots -/
structure AvailableSpots :=
  (day : Nat)
  (evening : Nat)

/-- Represents a hen's application -/
structure Application :=
  (color : HenColor)
  (program : ProgramType)

/-- The probability of at least one black hen in the daytime program -/
def prob_black_hen_day (total_spots : Nat) (day_spots : Nat) (evening_spots : Nat) 
                       (black_hens : Nat) (white_hens : Nat) : ℚ :=
  sorry

theorem black_hen_day_probability :
  let total_spots := 5
  let day_spots := 2
  let evening_spots := 3
  let black_hens := 3
  let white_hens := 1
  prob_black_hen_day total_spots day_spots evening_spots black_hens white_hens = 59 / 64 :=
by sorry

end black_hen_day_probability_l2906_290603


namespace pen_refill_purchase_comparison_l2906_290654

theorem pen_refill_purchase_comparison (p₁ p₂ : ℝ) (hp₁ : p₁ > 0) (hp₂ : p₂ > 0) :
  (2 * p₁ * p₂) / (p₁ + p₂) ≤ (p₁ + p₂) / 2 ∧
  (2 * p₁ * p₂) / (p₁ + p₂) = (p₁ + p₂) / 2 ↔ p₁ = p₂ := by
  sorry

end pen_refill_purchase_comparison_l2906_290654


namespace square_perimeter_l2906_290669

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 450 → 
  side^2 = area → 
  perimeter = 4 * side → 
  perimeter = 60 * Real.sqrt 2 := by
sorry

end square_perimeter_l2906_290669


namespace factorization_equality_l2906_290658

theorem factorization_equality (x y : ℝ) : x^2 * y - 4 * y = y * (x + 2) * (x - 2) := by
  sorry

end factorization_equality_l2906_290658


namespace binomial_seven_four_l2906_290696

theorem binomial_seven_four : Nat.choose 7 4 = 35 := by
  sorry

end binomial_seven_four_l2906_290696


namespace solve_installment_problem_l2906_290698

def installment_problem (cash_price : ℕ) (down_payment : ℕ) (first_four_payment : ℕ) (next_four_payment : ℕ) (total_months : ℕ) (installment_markup : ℕ) : Prop :=
  let total_installment_price := cash_price + installment_markup
  let paid_so_far := down_payment + 4 * first_four_payment + 4 * next_four_payment
  let remaining_amount := total_installment_price - paid_so_far
  let last_four_months := total_months - 8
  remaining_amount / last_four_months = 30

theorem solve_installment_problem :
  installment_problem 450 100 40 35 12 70 :=
sorry

end solve_installment_problem_l2906_290698


namespace parallel_condition_theorem_l2906_290611

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation between lines and between a line and a plane
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)

-- State the theorem
theorem parallel_condition_theorem 
  (a b : Line) (α : Plane) 
  (h_different : a ≠ b) 
  (h_contained : contained_in b α) :
  (∀ x y : Line, ∀ p : Plane, 
    contained_in y p → 
    parallel_lines x y → 
    parallel_line_plane x p) ∧
  (∃ x y : Line, ∃ p : Plane,
    contained_in y p ∧ 
    parallel_line_plane x p ∧ 
    ¬parallel_lines x y) →
  (parallel_line_plane a α → parallel_lines a b) ∧
  ¬(parallel_lines a b → parallel_line_plane a α) :=
sorry

end parallel_condition_theorem_l2906_290611


namespace computers_produced_per_month_l2906_290661

/-- Represents the number of computers produced in a month -/
def computers_per_month (computers_per_interval : ℝ) (days_per_month : ℕ) : ℝ :=
  computers_per_interval * (days_per_month * 24 * 2)

/-- Theorem stating that 4200 computers are produced per month -/
theorem computers_produced_per_month :
  computers_per_month 3.125 28 = 4200 := by
  sorry

end computers_produced_per_month_l2906_290661


namespace racing_cars_lcm_l2906_290615

theorem racing_cars_lcm (lap_time_A lap_time_B : ℕ) 
  (h1 : lap_time_A = 28) 
  (h2 : lap_time_B = 24) : 
  Nat.lcm lap_time_A lap_time_B = 168 := by
  sorry

end racing_cars_lcm_l2906_290615


namespace matts_plantation_length_l2906_290639

/-- Represents Matt's peanut plantation and its production process -/
structure PeanutPlantation where
  width : ℝ
  length : ℝ
  peanuts_per_sqft : ℝ
  peanuts_to_butter_ratio : ℝ
  butter_price_per_kg : ℝ
  total_revenue : ℝ

/-- Calculates the length of one side of Matt's plantation -/
def calculate_plantation_length (p : PeanutPlantation) : ℝ :=
  p.width

/-- Theorem stating that given the conditions, the length of Matt's plantation is 500 feet -/
theorem matts_plantation_length (p : PeanutPlantation) 
  (h1 : p.length = 500)
  (h2 : p.peanuts_per_sqft = 50)
  (h3 : p.peanuts_to_butter_ratio = 20 / 5)
  (h4 : p.butter_price_per_kg = 10)
  (h5 : p.total_revenue = 31250) :
  calculate_plantation_length p = 500 := by
  sorry

end matts_plantation_length_l2906_290639


namespace cube_sum_minus_product_2003_l2906_290633

theorem cube_sum_minus_product_2003 :
  {(x, y, z) : ℤ × ℤ × ℤ | x^3 + y^3 + z^3 - 3*x*y*z = 2003} =
  {(668, 668, 667), (668, 667, 668), (667, 668, 668)} := by
  sorry

end cube_sum_minus_product_2003_l2906_290633


namespace max_table_coverage_max_table_side_optimal_l2906_290692

/-- The side length of each square tablecloth in centimeters -/
def tablecloth_side : ℝ := 144

/-- The number of tablecloths available -/
def num_tablecloths : ℕ := 3

/-- The maximum side length of the square table that can be covered -/
def max_table_side : ℝ := 183

/-- Theorem stating that the maximum side length of a square table that can be completely
    covered by three square tablecloths, each with a side length of 144 cm, is 183 cm -/
theorem max_table_coverage :
  ∀ (table_side : ℝ),
  table_side ≤ max_table_side →
  (table_side ^ 2 : ℝ) ≤ num_tablecloths * tablecloth_side ^ 2 :=
by sorry

/-- Theorem stating that 183 cm is the largest possible side length for the table -/
theorem max_table_side_optimal :
  ∀ (larger_side : ℝ),
  larger_side > max_table_side →
  (larger_side ^ 2 : ℝ) > num_tablecloths * tablecloth_side ^ 2 :=
by sorry

end max_table_coverage_max_table_side_optimal_l2906_290692


namespace fraction_conforms_to_standard_notation_l2906_290629

/-- Rules for standard algebraic notation -/
structure AlgebraicNotationRules where
  no_multiplication_sign : Bool
  mixed_numbers_as_fractions : Bool
  division_as_fraction : Bool

/-- An algebraic expression -/
inductive AlgebraicExpression
  | Multiply : ℕ → Char → AlgebraicExpression
  | MixedNumber : ℕ → ℚ → Char → AlgebraicExpression
  | Fraction : Char → Char → ℕ → AlgebraicExpression
  | Divide : Char → ℕ → Char → AlgebraicExpression

/-- Function to check if an expression conforms to standard algebraic notation -/
def conforms_to_standard_notation (rules : AlgebraicNotationRules) (expr : AlgebraicExpression) : Prop :=
  match expr with
  | AlgebraicExpression.Fraction _ _ _ => true
  | _ => false

/-- Theorem stating that -b/a² conforms to standard algebraic notation -/
theorem fraction_conforms_to_standard_notation (rules : AlgebraicNotationRules) :
  conforms_to_standard_notation rules (AlgebraicExpression.Fraction 'b' 'a' 2) :=
sorry

end fraction_conforms_to_standard_notation_l2906_290629


namespace range_of_a_for_quadratic_inequality_l2906_290631

theorem range_of_a_for_quadratic_inequality :
  {a : ℝ | ∀ x : ℝ, a * x^2 - a * x - 2 ≤ 0} = {a : ℝ | -8 ≤ a ∧ a ≤ 0} := by sorry

end range_of_a_for_quadratic_inequality_l2906_290631


namespace patrick_current_age_l2906_290618

/-- Patrick's age is half of Robert's age -/
def patrick_age_relation (patrick_age robert_age : ℕ) : Prop :=
  patrick_age = robert_age / 2

/-- Robert will be 30 years old in 2 years -/
def robert_future_age (robert_age : ℕ) : Prop :=
  robert_age + 2 = 30

/-- The theorem stating Patrick's current age -/
theorem patrick_current_age :
  ∃ (patrick_age robert_age : ℕ),
    patrick_age_relation patrick_age robert_age ∧
    robert_future_age robert_age ∧
    patrick_age = 14 := by
  sorry

end patrick_current_age_l2906_290618


namespace odometer_skipping_four_l2906_290662

/-- Represents an odometer that skips the digit 4 -/
def SkippingOdometer : Type := ℕ

/-- Converts a regular number to its representation on the skipping odometer -/
def toSkippingOdometer (n : ℕ) : SkippingOdometer :=
  sorry

/-- Converts a skipping odometer reading back to the actual distance -/
def fromSkippingOdometer (s : SkippingOdometer) : ℕ :=
  sorry

/-- The theorem stating the relationship between the odometer reading and actual distance -/
theorem odometer_skipping_four (reading : SkippingOdometer) :
  reading = toSkippingOdometer 2005 →
  fromSkippingOdometer reading = 1462 :=
sorry

end odometer_skipping_four_l2906_290662


namespace candice_spending_l2906_290638

def total_money : ℕ := 100
def mildred_spent : ℕ := 25
def money_left : ℕ := 40

theorem candice_spending : 
  total_money - mildred_spent - money_left = 35 := by
sorry

end candice_spending_l2906_290638


namespace constant_curvature_curves_l2906_290620

/-- A plane curve is a continuous function from ℝ to ℝ² --/
def PlaneCurve := ℝ → ℝ × ℝ

/-- The curvature of a plane curve at a point --/
noncomputable def curvature (γ : PlaneCurve) (t : ℝ) : ℝ := sorry

/-- A curve has constant curvature if its curvature is the same at all points --/
def has_constant_curvature (γ : PlaneCurve) : Prop :=
  ∃ k : ℝ, ∀ t : ℝ, curvature γ t = k

/-- A straight line --/
def is_straight_line (γ : PlaneCurve) : Prop :=
  ∃ a b : ℝ × ℝ, ∀ t : ℝ, γ t = a + t • (b - a)

/-- A circle --/
def is_circle (γ : PlaneCurve) : Prop :=
  ∃ c : ℝ × ℝ, ∃ r : ℝ, r > 0 ∧ ∀ t : ℝ, ‖γ t - c‖ = r

/-- Theorem: The only plane curves with constant curvature are straight lines and circles --/
theorem constant_curvature_curves (γ : PlaneCurve) :
  has_constant_curvature γ ↔ is_straight_line γ ∨ is_circle γ :=
sorry

end constant_curvature_curves_l2906_290620


namespace smallest_n_theorem_l2906_290649

/-- The smallest positive integer n for which the equation 15x^2 - nx + 630 = 0 has integral solutions -/
def smallest_n : ℕ := 195

/-- The equation 15x^2 - nx + 630 = 0 has integral solutions -/
def has_integral_solutions (n : ℕ) : Prop :=
  ∃ x : ℤ, 15 * x^2 - n * x + 630 = 0

theorem smallest_n_theorem :
  (has_integral_solutions smallest_n) ∧
  (∀ m : ℕ, m < smallest_n → ¬(has_integral_solutions m)) := by
  sorry

end smallest_n_theorem_l2906_290649


namespace population_average_age_l2906_290672

/-- Proves that given a population with a specific ratio of women to men and their respective average ages, the average age of the entire population can be calculated. -/
theorem population_average_age 
  (total_population : ℕ) 
  (women_ratio : ℚ) 
  (men_ratio : ℚ) 
  (women_avg_age : ℚ) 
  (men_avg_age : ℚ) 
  (h1 : women_ratio + men_ratio = 1) 
  (h2 : women_ratio = 11 / 21) 
  (h3 : men_ratio = 10 / 21) 
  (h4 : women_avg_age = 34) 
  (h5 : men_avg_age = 32) : 
  (women_ratio * women_avg_age + men_ratio * men_avg_age : ℚ) = 33 + 1 / 21 := by
  sorry

#check population_average_age

end population_average_age_l2906_290672


namespace fourth_root_equivalence_l2906_290627

theorem fourth_root_equivalence (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 * y^(1/3))^(1/4) = x^(1/2) * y^(1/12) := by
  sorry

end fourth_root_equivalence_l2906_290627


namespace greatest_common_divisor_with_same_remainder_l2906_290641

theorem greatest_common_divisor_with_same_remainder (a b c : ℕ) (ha : a = 54) (hb : b = 87) (hc : c = 172) :
  ∃ (d : ℕ), d > 0 ∧ 
  (∃ (r : ℕ), a % d = r ∧ b % d = r ∧ c % d = r) ∧
  (∀ (k : ℕ), k > d → ¬(∃ (s : ℕ), a % k = s ∧ b % k = s ∧ c % k = s)) →
  d = 1 :=
by sorry

end greatest_common_divisor_with_same_remainder_l2906_290641


namespace min_value_expression_l2906_290645

theorem min_value_expression (a b c : ℝ) (h1 : b > c) (h2 : c > a) (h3 : a > 0) (h4 : b ≠ 0) :
  ((a + b)^3 + (b - c)^2 + (c - a)^3) / b^2 ≥ 2 := by
  sorry

end min_value_expression_l2906_290645


namespace johns_average_speed_l2906_290677

/-- John's cycling and walking trip -/
def johns_trip (uphill_distance : ℝ) (uphill_time : ℝ) (downhill_time : ℝ) (walk_distance : ℝ) (walk_time : ℝ) : Prop :=
  let total_distance := 2 * uphill_distance + walk_distance
  let total_time := uphill_time + downhill_time + walk_time
  (total_distance / (total_time / 60)) = 6

theorem johns_average_speed :
  johns_trip 3 45 15 2 20 := by
  sorry

end johns_average_speed_l2906_290677


namespace painting_price_increase_l2906_290648

theorem painting_price_increase (X : ℝ) : 
  (1 + X / 100) * (1 - 0.15) = 1.02 → X = 20 := by
  sorry

end painting_price_increase_l2906_290648


namespace max_value_of_five_numbers_l2906_290651

theorem max_value_of_five_numbers (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →  -- distinct and ordered
  (a + b + c + d + e) / 5 = 12 →  -- average is 12
  c = 17 →  -- median is 17
  e ≤ 24 :=  -- maximum possible value is 24
by sorry

end max_value_of_five_numbers_l2906_290651


namespace ticket_distribution_theorem_l2906_290607

/-- The number of ways to distribute 3 different tickets to 3 students out of a group of 10 -/
def ticket_distribution_ways : ℕ := 10 * 9 * 8

/-- Theorem: The number of ways to distribute 3 different tickets to 3 students out of a group of 10 is 720 -/
theorem ticket_distribution_theorem : ticket_distribution_ways = 720 := by
  sorry

end ticket_distribution_theorem_l2906_290607


namespace union_of_A_and_B_l2906_290621

-- Define the sets A and B
def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {1, 2}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} := by sorry

end union_of_A_and_B_l2906_290621


namespace milk_consumption_l2906_290640

/-- The amount of regular milk consumed by Mitch's family in 1 week -/
def regular_milk : ℝ := 0.5

/-- The amount of soy milk consumed by Mitch's family in 1 week -/
def soy_milk : ℝ := 0.1

/-- The total amount of milk consumed by Mitch's family in 1 week -/
def total_milk : ℝ := regular_milk + soy_milk

theorem milk_consumption :
  total_milk = 0.6 := by sorry

end milk_consumption_l2906_290640


namespace rectangle_length_calculation_l2906_290605

/-- Represents a rectangular piece of land -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- The area of a rectangle -/
def area (r : Rectangle) : ℝ := r.width * r.length

/-- Theorem: For a rectangle with area 215.6 m² and width 14 m, the length is 15.4 m -/
theorem rectangle_length_calculation (r : Rectangle) 
  (h_area : area r = 215.6) 
  (h_width : r.width = 14) : 
  r.length = 15.4 := by
  sorry

#check rectangle_length_calculation

end rectangle_length_calculation_l2906_290605


namespace pen_to_book_ratio_l2906_290625

theorem pen_to_book_ratio (pencils pens books : ℕ) 
  (h1 : pencils = 140)
  (h2 : books = 30)
  (h3 : pencils * 4 = pens * 14)
  (h4 : pencils * 3 = books * 14) : 
  4 * books = 3 * pens := by
  sorry

end pen_to_book_ratio_l2906_290625


namespace awards_distribution_l2906_290637

/-- The number of ways to distribute n distinct awards to k students,
    where each student receives at least one award. -/
def distribute_awards (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n distinct items. -/
def choose (n r : ℕ) : ℕ := sorry

theorem awards_distribution :
  distribute_awards 5 4 = 240 :=
by
  sorry

end awards_distribution_l2906_290637


namespace mean_chocolate_sales_l2906_290673

def week1_sales : ℕ := 75
def week2_sales : ℕ := 67
def week3_sales : ℕ := 75
def week4_sales : ℕ := 70
def week5_sales : ℕ := 68
def num_weeks : ℕ := 5

def total_sales : ℕ := week1_sales + week2_sales + week3_sales + week4_sales + week5_sales

theorem mean_chocolate_sales :
  (total_sales : ℚ) / num_weeks = 71 := by sorry

end mean_chocolate_sales_l2906_290673


namespace dish_temperature_l2906_290612

/-- Calculates the final temperature of a dish in an oven -/
def final_temperature (start_temp : ℝ) (heating_rate : ℝ) (cooking_time : ℝ) : ℝ :=
  start_temp + heating_rate * cooking_time

/-- Proves that the dish reaches 100 degrees given the specified conditions -/
theorem dish_temperature : final_temperature 20 5 16 = 100 := by
  sorry

end dish_temperature_l2906_290612


namespace same_solution_equations_l2906_290695

theorem same_solution_equations (x : ℝ) (d : ℝ) : 
  (3 * x + 8 = 5) ∧ (d * x - 15 = -7) → d = -8 := by
sorry

end same_solution_equations_l2906_290695


namespace sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2906_290659

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  let roots := {x : ℝ | f x = 0}
  (∃ x y : ℝ, x ∈ roots ∧ y ∈ roots ∧ x ≠ y) →
  (∀ z : ℝ, z ∈ roots → z = x ∨ z = y) →
  x + y = -b / a :=
by sorry

theorem sum_of_roots_specific_quadratic :
  let f : ℝ → ℝ := λ x ↦ 3 * x^2 - 15 * x + 20
  let roots := {x : ℝ | f x = 0}
  ∃ C D : ℝ, C ∈ roots ∧ D ∈ roots ∧ C ≠ D ∧
    (∀ z : ℝ, z ∈ roots → z = C ∨ z = D) ∧
    C + D = 5 :=
by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l2906_290659
