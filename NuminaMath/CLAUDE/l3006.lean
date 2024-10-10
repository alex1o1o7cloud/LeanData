import Mathlib

namespace election_winner_votes_l3006_300624

/-- The number of candidates in the election -/
def num_candidates : ℕ := 4

/-- The percentage of votes received by the winning candidate -/
def winner_percentage : ℚ := 468 / 1000

/-- The percentage of votes received by the second-place candidate -/
def second_percentage : ℚ := 326 / 1000

/-- The margin of victory in number of votes -/
def margin : ℕ := 752

/-- The total number of votes cast in the election -/
def total_votes : ℕ := 5296

/-- The number of votes received by the winning candidate -/
def winner_votes : ℕ := 2479

theorem election_winner_votes :
  num_candidates = 4 ∧
  winner_percentage = 468 / 1000 ∧
  second_percentage = 326 / 1000 ∧
  margin = 752 ∧
  total_votes = 5296 →
  winner_votes = 2479 :=
by sorry

end election_winner_votes_l3006_300624


namespace menelaus_condition_l3006_300666

-- Define the points
variable (A B C D P Q R S O : Point)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define points on sides
def point_on_segment (P A B : Point) : Prop := sorry

-- Define intersection of lines
def lines_intersect (P R Q S O : Point) : Prop := sorry

-- Define quadrilateral with incircle
def has_incircle (A P O S : Point) : Prop := sorry

-- Define the ratio of segments
def segment_ratio (A P B : Point) : ℝ := sorry

-- Main theorem
theorem menelaus_condition 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_P : point_on_segment P A B)
  (h_Q : point_on_segment Q B C)
  (h_R : point_on_segment R C D)
  (h_S : point_on_segment S D A)
  (h_intersect : lines_intersect P R Q S O)
  (h_incircle1 : has_incircle A P O S)
  (h_incircle2 : has_incircle B Q O P)
  (h_incircle3 : has_incircle C R O Q)
  (h_incircle4 : has_incircle D S O R) :
  (segment_ratio A P B) * (segment_ratio B Q C) * 
  (segment_ratio C R D) * (segment_ratio D S A) = 1 := by
  sorry

end menelaus_condition_l3006_300666


namespace ali_final_money_l3006_300692

-- Define the initial state of Ali's wallet
def initial_wallet : ℚ :=
  7 * 5 + 1 * 10 + 3 * 20 + 1 * 50 + 8 * 1 + 10 * (1/4)

-- Morning transaction
def morning_transaction (wallet : ℚ) : ℚ :=
  wallet - (50 + 20 + 5) + (3 + 8 * (1/4) + 10 * (1/10))

-- Coffee shop transaction
def coffee_transaction (wallet : ℚ) : ℚ :=
  wallet - (15/4)

-- Afternoon transaction
def afternoon_transaction (wallet : ℚ) : ℚ :=
  wallet + 42

-- Evening transaction
def evening_transaction (wallet : ℚ) : ℚ :=
  wallet - (45/4)

-- Final wallet state after all transactions
def final_wallet : ℚ :=
  evening_transaction (afternoon_transaction (coffee_transaction (morning_transaction initial_wallet)))

-- Theorem statement
theorem ali_final_money :
  final_wallet = 247/2 := by sorry

end ali_final_money_l3006_300692


namespace three_possible_values_for_d_l3006_300631

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Checks if four digits are distinct -/
def distinct (a b c d : Digit) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d

/-- Represents the equation AABC + CBBA = DCCD -/
def satisfies_equation (a b c d : Digit) : Prop :=
  1000 * a.val + 100 * a.val + 10 * b.val + c.val +
  1000 * c.val + 100 * b.val + 10 * b.val + a.val =
  1000 * d.val + 100 * c.val + 10 * c.val + d.val

/-- The main theorem stating there are exactly 3 possible values for D -/
theorem three_possible_values_for_d :
  ∃ (s : Finset Digit),
    s.card = 3 ∧
    (∀ d : Digit, d ∈ s ↔ 
      ∃ (a b c : Digit), distinct a b c d ∧ satisfies_equation a b c d) :=
sorry

end three_possible_values_for_d_l3006_300631


namespace equation_solution_l3006_300667

theorem equation_solution (x : ℝ) (h : 1 - 9 / x + 9 / x^2 = 0) :
  3 / x = (3 - Real.sqrt 5) / 2 ∨ 3 / x = (3 + Real.sqrt 5) / 2 := by
  sorry

end equation_solution_l3006_300667


namespace abc_fraction_value_l3006_300644

theorem abc_fraction_value (a b c : ℝ) 
  (h1 : a * b / (a + b) = 4)
  (h2 : b * c / (b + c) = 5)
  (h3 : c * a / (c + a) = 7) :
  a * b * c / (a * b + b * c + c * a) = 280 / 83 := by
  sorry

end abc_fraction_value_l3006_300644


namespace inequality_proof_l3006_300660

theorem inequality_proof (x y : ℝ) (h1 : y ≥ 0) (h2 : y * (y + 1) ≤ (x + 1)^2) :
  y * (y - 1) ≤ x^2 := by
  sorry

end inequality_proof_l3006_300660


namespace constant_molecular_weight_l3006_300649

-- Define the molecular weight of the compound
def molecular_weight : ℝ := 918

-- Define a function that returns the molecular weight for any number of moles
def weight_for_moles (moles : ℝ) : ℝ := molecular_weight

-- Theorem stating that the molecular weight is constant regardless of the number of moles
theorem constant_molecular_weight (moles : ℝ) :
  weight_for_moles moles = molecular_weight := by
  sorry

end constant_molecular_weight_l3006_300649


namespace inequality_not_always_hold_l3006_300650

theorem inequality_not_always_hold (a b : ℝ) (h : a > b) : 
  ¬ (∀ c : ℝ, a * c > b * c) := by
sorry

end inequality_not_always_hold_l3006_300650


namespace unique_monic_quadratic_l3006_300636

/-- A monic polynomial of degree 2 -/
def MonicQuadratic (g : ℝ → ℝ) : Prop :=
  ∃ b c : ℝ, ∀ x, g x = x^2 + b*x + c

theorem unique_monic_quadratic (g : ℝ → ℝ) 
  (h_monic : MonicQuadratic g) 
  (h_g0 : g 0 = 2) 
  (h_g1 : g 1 = 6) : 
  ∀ x, g x = x^2 + 3*x + 2 := by
  sorry

end unique_monic_quadratic_l3006_300636


namespace mutually_exclusive_but_not_complementary_l3006_300688

/-- A bag containing red and black balls -/
structure Bag where
  red : ℕ
  black : ℕ

/-- The sample space of drawing two balls from the bag -/
def SampleSpace (b : Bag) := Fin (b.red + b.black) × Fin (b.red + b.black - 1)

/-- Event of drawing exactly one black ball -/
def ExactlyOneBlack (b : Bag) : Set (SampleSpace b) := sorry

/-- Event of drawing exactly two black balls -/
def ExactlyTwoBlack (b : Bag) : Set (SampleSpace b) := sorry

/-- Two events are mutually exclusive -/
def MutuallyExclusive {α : Type*} (A B : Set α) : Prop :=
  A ∩ B = ∅

/-- Two events are complementary -/
def Complementary {α : Type*} (A B : Set α) : Prop :=
  MutuallyExclusive A B ∧ A ∪ B = Set.univ

/-- The main theorem -/
theorem mutually_exclusive_but_not_complementary :
  let b : Bag := ⟨2, 2⟩
  MutuallyExclusive (ExactlyOneBlack b) (ExactlyTwoBlack b) ∧
  ¬Complementary (ExactlyOneBlack b) (ExactlyTwoBlack b) := by sorry

end mutually_exclusive_but_not_complementary_l3006_300688


namespace quarter_squared_decimal_l3006_300632

theorem quarter_squared_decimal : (1 / 4 : ℚ) ^ 2 = 0.0625 := by
  sorry

end quarter_squared_decimal_l3006_300632


namespace trigonometric_equation_solution_l3006_300691

theorem trigonometric_equation_solution :
  ∀ x : ℝ, 5.22 * (Real.sin x)^2 - 2 * Real.sin x * Real.cos x = 3 * (Real.cos x)^2 ↔
  (∃ k : ℤ, x = Real.arctan 0.973 + k * Real.pi) ∨
  (∃ k : ℤ, x = Real.arctan (-0.59) + k * Real.pi) :=
by sorry

end trigonometric_equation_solution_l3006_300691


namespace geometric_progression_a10_l3006_300646

/-- A geometric progression with given conditions -/
def geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_progression_a10 (a : ℕ → ℝ) :
  geometric_progression a → a 2 = 2 → a 6 = 162 → a 10 = 13122 := by
  sorry

end geometric_progression_a10_l3006_300646


namespace infinitely_many_satisfy_property_l3006_300680

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Property that n divides F_{F_n} but not F_n -/
def satisfies_property (n : ℕ) : Prop :=
  n > 0 ∧ (n ∣ fib (fib n)) ∧ ¬(n ∣ fib n)

theorem infinitely_many_satisfy_property :
  ∀ k : ℕ, k > 0 → satisfies_property (12 * k) :=
by sorry

end infinitely_many_satisfy_property_l3006_300680


namespace granola_bar_distribution_l3006_300683

/-- Calculates the number of granola bars per kid given the number of kids, bars per box, and boxes purchased. -/
def granola_bars_per_kid (num_kids : ℕ) (bars_per_box : ℕ) (boxes_purchased : ℕ) : ℕ :=
  (bars_per_box * boxes_purchased) / num_kids

/-- Proves that given 30 kids, 12 bars per box, and 5 boxes purchased, the number of granola bars per kid is 2. -/
theorem granola_bar_distribution : granola_bars_per_kid 30 12 5 = 2 := by
  sorry

#eval granola_bars_per_kid 30 12 5

end granola_bar_distribution_l3006_300683


namespace sports_camp_coach_age_l3006_300641

theorem sports_camp_coach_age (total_members : ℕ) (avg_age : ℕ) 
  (num_girls num_boys num_coaches : ℕ) (avg_age_girls avg_age_boys : ℕ) :
  total_members = 30 →
  avg_age = 20 →
  num_girls = 10 →
  num_boys = 15 →
  num_coaches = 5 →
  avg_age_girls = 18 →
  avg_age_boys = 19 →
  (total_members * avg_age - num_girls * avg_age_girls - num_boys * avg_age_boys) / num_coaches = 27 :=
by sorry

end sports_camp_coach_age_l3006_300641


namespace chinese_team_gold_medal_probability_l3006_300696

/-- The probability of the Chinese team winning the gold medal in Women's Singles Table Tennis -/
theorem chinese_team_gold_medal_probability :
  let prob_A : ℚ := 3/7  -- Probability of Player A winning
  let prob_B : ℚ := 1/4  -- Probability of Player B winning
  -- Assuming the events are mutually exclusive
  prob_A + prob_B = 19/28 := by sorry

end chinese_team_gold_medal_probability_l3006_300696


namespace tan_255_degrees_l3006_300615

theorem tan_255_degrees : Real.tan (255 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end tan_255_degrees_l3006_300615


namespace complex_imaginary_solution_l3006_300662

theorem complex_imaginary_solution (a : ℝ) : 
  (a - (5 : ℂ) / (2 - Complex.I)).im = (a - (5 : ℂ) / (2 - Complex.I)).re → a = 2 := by
  sorry

end complex_imaginary_solution_l3006_300662


namespace square_field_area_l3006_300628

/-- Proves that a square field crossed diagonally in 9 seconds by a man walking at 6 km/h has an area of 112.5 square meters. -/
theorem square_field_area (speed_kmh : ℝ) (time_s : ℝ) (area : ℝ) : 
  speed_kmh = 6 → time_s = 9 → area = 112.5 → 
  let speed_ms := speed_kmh * 1000 / 3600
  let diagonal := speed_ms * time_s
  let side := (diagonal^2 / 2).sqrt
  area = side^2 := by sorry

end square_field_area_l3006_300628


namespace geometric_sequence_a7_l3006_300603

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_a7 (a : ℕ → ℚ) :
  GeometricSequence a →
  a 5 = 1/2 →
  4 * a 3 + a 7 = 2 →
  a 7 = 2/3 := by
sorry

end geometric_sequence_a7_l3006_300603


namespace james_tshirt_purchase_l3006_300695

/-- The total cost for a discounted purchase of t-shirts -/
def discounted_total_cost (num_shirts : ℕ) (original_price : ℚ) (discount_percent : ℚ) : ℚ :=
  num_shirts * original_price * (1 - discount_percent)

/-- Theorem: James pays $60 for 6 t-shirts at 50% off, originally priced at $20 each -/
theorem james_tshirt_purchase : 
  discounted_total_cost 6 20 (1/2) = 60 := by
  sorry

end james_tshirt_purchase_l3006_300695


namespace lines_perpendicular_to_plane_are_parallel_l3006_300648

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end lines_perpendicular_to_plane_are_parallel_l3006_300648


namespace x_is_perfect_square_l3006_300659

theorem x_is_perfect_square (x y : ℕ+) (h : (2 * x * y) ∣ (x^2 + y^2 - x)) : 
  ∃ n : ℕ+, x = n^2 := by
sorry

end x_is_perfect_square_l3006_300659


namespace wire_ratio_proof_l3006_300607

theorem wire_ratio_proof (total_length longer_length shorter_length : ℚ) : 
  total_length = 80 →
  shorter_length = 30 →
  longer_length = total_length - shorter_length →
  shorter_length / longer_length = 3 / 5 := by
  sorry

end wire_ratio_proof_l3006_300607


namespace multiply_by_17_equals_493_l3006_300611

theorem multiply_by_17_equals_493 : ∃ x : ℤ, x * 17 = 493 ∧ x = 29 := by
  sorry

end multiply_by_17_equals_493_l3006_300611


namespace complex_magnitude_proof_l3006_300616

/-- Proves that for a complex number z with argument 60°, 
    if |z-1| is the geometric mean of |z| and |z-2|, then |z| = √2 + 1 -/
theorem complex_magnitude_proof (z : ℂ) :
  Complex.arg z = π / 3 →
  Complex.abs (z - 1) ^ 2 = Complex.abs z * Complex.abs (z - 2) →
  Complex.abs z = Real.sqrt 2 + 1 := by
sorry

end complex_magnitude_proof_l3006_300616


namespace chess_piece_position_l3006_300643

/-- Represents a position on a chess board -/
structure ChessPosition :=
  (column : Nat)
  (row : Nat)

/-- Converts a ChessPosition to a pair of natural numbers -/
def ChessPosition.toPair (pos : ChessPosition) : Nat × Nat :=
  (pos.column, pos.row)

theorem chess_piece_position :
  let piece : ChessPosition := ⟨3, 7⟩
  ChessPosition.toPair piece = (3, 7) := by
  sorry

end chess_piece_position_l3006_300643


namespace joan_initial_dimes_l3006_300625

/-- The number of dimes Joan spent -/
def dimes_spent : ℕ := 2

/-- The number of dimes Joan has left -/
def dimes_left : ℕ := 3

/-- The initial number of dimes Joan had -/
def initial_dimes : ℕ := dimes_spent + dimes_left

theorem joan_initial_dimes : initial_dimes = 5 := by sorry

end joan_initial_dimes_l3006_300625


namespace squareable_numbers_l3006_300665

def isSquareable (n : ℕ) : Prop :=
  ∃ (p : Fin n → Fin n), Function.Bijective p ∧
    ∀ i : Fin n, ∃ k : ℕ, (p i).val.succ + (i.val.succ) = k * k

theorem squareable_numbers : 
  isSquareable 9 ∧ 
  isSquareable 15 ∧ 
  ¬isSquareable 7 ∧ 
  ¬isSquareable 11 :=
sorry

end squareable_numbers_l3006_300665


namespace park_visitors_l3006_300681

theorem park_visitors (visitors_day1 visitors_day2 : ℕ) : 
  visitors_day2 = visitors_day1 + 40 →
  visitors_day1 + visitors_day2 = 440 →
  visitors_day1 = 200 := by
sorry

end park_visitors_l3006_300681


namespace power_fraction_simplification_l3006_300640

theorem power_fraction_simplification :
  (10 ^ 0.7) * (10 ^ 0.4) / ((10 ^ 0.2) * (10 ^ 0.6) * (10 ^ 0.3)) = 1 := by
  sorry

end power_fraction_simplification_l3006_300640


namespace complex_fraction_evaluation_l3006_300684

theorem complex_fraction_evaluation : 
  (2 / (3 + 1/5) + ((3 + 1/4) / 13) / (2/3) + (2 + 5/18 - 17/36) * (18/65)) * (1/3) = 1/2 := by
  sorry

end complex_fraction_evaluation_l3006_300684


namespace min_overlap_brown_eyes_and_lunch_box_l3006_300685

theorem min_overlap_brown_eyes_and_lunch_box 
  (total_students : ℕ) 
  (brown_eyes : ℕ) 
  (lunch_box : ℕ) 
  (h1 : total_students = 40) 
  (h2 : brown_eyes = 18) 
  (h3 : lunch_box = 25) : 
  ∃ (overlap : ℕ), 
    overlap ≥ brown_eyes + lunch_box - total_students ∧ 
    overlap = 3 := by
  sorry

end min_overlap_brown_eyes_and_lunch_box_l3006_300685


namespace rental_shop_problem_l3006_300669

/-- Rental shop problem -/
theorem rental_shop_problem 
  (first_hour_rate : ℝ) 
  (additional_hour_rate : ℝ)
  (sales_tax_rate : ℝ)
  (total_paid : ℝ)
  (h : ℕ)
  (h_def : h = (total_paid / (1 + sales_tax_rate) - first_hour_rate) / additional_hour_rate)
  (first_hour_rate_def : first_hour_rate = 25)
  (additional_hour_rate_def : additional_hour_rate = 10)
  (sales_tax_rate_def : sales_tax_rate = 0.08)
  (total_paid_def : total_paid = 125) :
  h + 1 = 10 := by
sorry


end rental_shop_problem_l3006_300669


namespace eight_digit_non_decreasing_integers_mod_1000_l3006_300618

/-- The number of ways to distribute n indistinguishable objects into k distinguishable bins -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) k

/-- The number of 8-digit positive integers with non-decreasing digits -/
def M : ℕ := stars_and_bars 8 9

theorem eight_digit_non_decreasing_integers_mod_1000 : M % 1000 = 870 := by sorry

end eight_digit_non_decreasing_integers_mod_1000_l3006_300618


namespace road_width_calculation_l3006_300687

/-- Calculates the width of roads on a rectangular lawn given the dimensions and cost --/
theorem road_width_calculation (length width total_cost cost_per_sqm : ℝ) : 
  length = 80 →
  width = 60 →
  total_cost = 3900 →
  cost_per_sqm = 3 →
  let road_area := total_cost / cost_per_sqm
  let road_width := road_area / (length + width)
  road_width = 65 / 7 := by
  sorry

end road_width_calculation_l3006_300687


namespace difference_of_cubes_divisible_by_eight_l3006_300629

theorem difference_of_cubes_divisible_by_eight (a b : ℤ) :
  ∃ k : ℤ, (2*a - 1)^3 - (2*b - 1)^3 = 8 * k := by
sorry

end difference_of_cubes_divisible_by_eight_l3006_300629


namespace reciprocal_problem_l3006_300694

theorem reciprocal_problem :
  (∀ x : ℚ, x ≠ 0 → x * (1 / x) = 1) →
  (1 / 0.125 = 8) ∧ (1 / 1 = 1) := by sorry

end reciprocal_problem_l3006_300694


namespace intersection_sum_l3006_300623

-- Define the constants and variables
variable (n c : ℝ)
variable (x y : ℝ)

-- Define the two lines
def line1 (x : ℝ) : ℝ := n * x + 5
def line2 (x : ℝ) : ℝ := 4 * x + c

-- State the theorem
theorem intersection_sum (h1 : line1 5 = 15) (h2 : line2 5 = 15) : c + n = -3 := by
  sorry

end intersection_sum_l3006_300623


namespace households_with_car_l3006_300682

theorem households_with_car (total : ℕ) (neither : ℕ) (both : ℕ) (bike_only : ℕ) 
  (h1 : total = 90)
  (h2 : neither = 11)
  (h3 : both = 22)
  (h4 : bike_only = 35) :
  total - neither - bike_only = 44 :=
by sorry

end households_with_car_l3006_300682


namespace deposit_amount_is_34_l3006_300609

/-- Represents a bank account with transactions --/
structure BankAccount where
  initial_balance : ℕ
  last_month_deposit : ℕ
  current_balance : ℕ

/-- Calculates the deposit amount this month --/
def deposit_this_month (account : BankAccount) : ℕ :=
  account.current_balance - account.initial_balance

/-- Theorem: The deposit amount this month is $34 --/
theorem deposit_amount_is_34 (account : BankAccount) 
  (h1 : account.initial_balance = 150)
  (h2 : account.last_month_deposit = 17)
  (h3 : account.current_balance = account.initial_balance + 16) :
  deposit_this_month account = 34 := by
  sorry

#eval deposit_this_month { initial_balance := 150, last_month_deposit := 17, current_balance := 166 }

end deposit_amount_is_34_l3006_300609


namespace room_width_proof_l3006_300622

/-- Given a rectangular room with length 5.5 meters, prove that its width is 4 meters
    when the cost of paving is 850 rupees per square meter and the total cost is 18,700 rupees. -/
theorem room_width_proof (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  length = 5.5 →
  cost_per_sqm = 850 →
  total_cost = 18700 →
  total_cost / cost_per_sqm / length = 4 := by
sorry

end room_width_proof_l3006_300622


namespace peaches_before_equals_34_l3006_300621

/-- The number of peaches Mike picked from the orchard -/
def peaches_picked : ℕ := 52

/-- The current total number of peaches at the stand -/
def current_total : ℕ := 86

/-- The number of peaches Mike had left at the stand before picking more -/
def peaches_before : ℕ := current_total - peaches_picked

theorem peaches_before_equals_34 : peaches_before = 34 := by sorry

end peaches_before_equals_34_l3006_300621


namespace x_over_y_is_negative_two_l3006_300661

theorem x_over_y_is_negative_two (x y : ℝ) (h1 : 1 < (x - y) / (x + y))
  (h2 : (x - y) / (x + y) < 3) (h3 : ∃ (n : ℤ), x / y = n) :
  x / y = -2 := by
  sorry

end x_over_y_is_negative_two_l3006_300661


namespace range_of_x2_plus_y2_l3006_300657

theorem range_of_x2_plus_y2 (x y : ℝ) (h : x^2 - 2*x*y + 5*y^2 = 4) :
  ∃ (min max : ℝ), min = 3 - Real.sqrt 5 ∧ max = 3 + Real.sqrt 5 ∧
  (min ≤ x^2 + y^2 ∧ x^2 + y^2 ≤ max) ∧
  ∃ (x1 y1 x2 y2 : ℝ), x1^2 - 2*x1*y1 + 5*y1^2 = 4 ∧
                       x2^2 - 2*x2*y2 + 5*y2^2 = 4 ∧
                       x1^2 + y1^2 = min ∧
                       x2^2 + y2^2 = max :=
by sorry

end range_of_x2_plus_y2_l3006_300657


namespace pet_shop_total_l3006_300610

/-- Given a pet shop with dogs, cats, and bunnies in stock, prove that the total number of dogs and bunnies is 375. -/
theorem pet_shop_total (dogs cats bunnies : ℕ) : 
  dogs = 75 →
  dogs / 3 = cats / 7 →
  dogs / 3 = bunnies / 12 →
  dogs + bunnies = 375 := by
sorry


end pet_shop_total_l3006_300610


namespace max_parts_three_planes_exists_eight_parts_l3006_300654

/-- A plane in 3D space -/
structure Plane3D where
  -- We don't need to define the specifics of a plane for this problem

/-- The number of parts that a set of planes divides 3D space into -/
def num_parts (planes : List Plane3D) : ℕ :=
  sorry -- Definition not provided as it's not necessary for the statement

/-- Theorem: Three planes can divide 3D space into at most 8 parts -/
theorem max_parts_three_planes :
  ∀ (p1 p2 p3 : Plane3D), num_parts [p1, p2, p3] ≤ 8 :=
sorry

/-- Theorem: There exists a configuration of three planes that divides 3D space into exactly 8 parts -/
theorem exists_eight_parts :
  ∃ (p1 p2 p3 : Plane3D), num_parts [p1, p2, p3] = 8 :=
sorry

end max_parts_three_planes_exists_eight_parts_l3006_300654


namespace range_of_f_when_m_eq_1_solution_set_of_f_gt_3x_when_m_eq_neg_1_inequality_equivalence_l3006_300652

-- Define the function f(x) with parameter m
def f (m : ℝ) (x : ℝ) : ℝ := |x + 1| - m * |x - 2|

-- Theorem for the range of f(x) when m = 1
theorem range_of_f_when_m_eq_1 :
  Set.range (f 1) = Set.Icc (-3) 3 := by sorry

-- Theorem for the solution set of f(x) > 3x when m = -1
theorem solution_set_of_f_gt_3x_when_m_eq_neg_1 :
  {x : ℝ | f (-1) x > 3 * x} = Set.Iio 1 := by sorry

-- Additional helper theorem to show the equivalence of the inequality
theorem inequality_equivalence (x : ℝ) :
  f (-1) x > 3 * x ↔ |x + 1| + |x - 2| > 3 * x := by sorry

end range_of_f_when_m_eq_1_solution_set_of_f_gt_3x_when_m_eq_neg_1_inequality_equivalence_l3006_300652


namespace inequality_proof_l3006_300655

theorem inequality_proof (x y : ℝ) (n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y = 1) (h4 : n ≥ 2) :
  (x^n / (x + y^3)) + (y^n / (x^3 + y)) ≥ (2^(4-n)) / 5 := by
  sorry

end inequality_proof_l3006_300655


namespace negation_of_existence_l3006_300672

theorem negation_of_existence (Triangle : Type) (isSymmetrical : Triangle → Prop) :
  (¬ ∃ t : Triangle, isSymmetrical t) ↔ (∀ t : Triangle, ¬ isSymmetrical t) := by sorry

end negation_of_existence_l3006_300672


namespace part_one_part_two_l3006_300613

/-- Given real numbers a and b, define the functions f and g -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x

/-- Define the derivatives of f and g -/
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a
def g' (b : ℝ) (x : ℝ) : ℝ := 2*x + b

/-- Define consistent monotonicity on an interval -/
def consistent_monotonicity (a b : ℝ) (I : Set ℝ) : Prop :=
  ∀ x ∈ I, f' a x * g' b x ≥ 0

/-- Part 1: Prove that if a > 0 and f, g have consistent monotonicity on [-1, +∞), then b ≥ 2 -/
theorem part_one (a b : ℝ) (ha : a > 0)
  (h_cons : consistent_monotonicity a b { x | x ≥ -1 }) : b ≥ 2 := by
  sorry

/-- Part 2: Prove that if a < 0, a ≠ b, and f, g have consistent monotonicity on (min a b, max a b),
    then |a - b| ≤ 1/3 -/
theorem part_two (a b : ℝ) (ha : a < 0) (hab : a ≠ b)
  (h_cons : consistent_monotonicity a b (Set.Ioo (min a b) (max a b))) : |a - b| ≤ 1/3 := by
  sorry

end part_one_part_two_l3006_300613


namespace purely_imaginary_z_l3006_300676

theorem purely_imaginary_z (z : ℂ) : 
  (∃ b : ℝ, z = Complex.I * b) →  -- z is purely imaginary
  (∃ c : ℝ, (z - 3)^2 + Complex.I * 5 = Complex.I * c) →  -- (z-3)^2+5i is purely imaginary
  z = Complex.I * 3 ∨ z = Complex.I * (-3) :=
by sorry

end purely_imaginary_z_l3006_300676


namespace cylinder_volume_ratio_l3006_300620

/-- The ratio of cylinder volumes formed by rolling a 6x9 rectangle -/
theorem cylinder_volume_ratio :
  let l₁ : ℝ := 6
  let l₂ : ℝ := 9
  let v₁ : ℝ := l₁ * l₂^2 / (4 * Real.pi)
  let v₂ : ℝ := l₂ * l₁^2 / (4 * Real.pi)
  max v₁ v₂ / min v₁ v₂ = 3/2 := by
sorry

end cylinder_volume_ratio_l3006_300620


namespace divisibility_by_1995_l3006_300614

theorem divisibility_by_1995 (n : ℕ) : 
  1995 ∣ 256^(2*n) * 7^(2*n) - 168^(2*n) - 32^(2*n) + 3^(2*n) := by
sorry

end divisibility_by_1995_l3006_300614


namespace one_third_repeating_one_seventh_repeating_one_ninth_repeating_l3006_300673

def repeating_decimal (n : ℕ) (d : ℕ) (period : List ℕ) : ℚ :=
  (n : ℚ) / (d : ℚ)

theorem one_third_repeating : repeating_decimal 1 3 [3] = 1 / 3 := by sorry

theorem one_seventh_repeating : repeating_decimal 1 7 [1, 4, 2, 8, 5, 7] = 1 / 7 := by sorry

theorem one_ninth_repeating : repeating_decimal 1 9 [1] = 1 / 9 := by sorry

end one_third_repeating_one_seventh_repeating_one_ninth_repeating_l3006_300673


namespace xiaoxiao_types_faster_l3006_300686

/-- Represents a typist with their typing data -/
structure Typist where
  name : String
  characters : ℕ
  minutes : ℕ

/-- Calculate the typing speed in characters per minute -/
def typingSpeed (t : Typist) : ℚ :=
  t.characters / t.minutes

/-- Determine if one typist is faster than another -/
def isFaster (t1 t2 : Typist) : Prop :=
  typingSpeed t1 > typingSpeed t2

theorem xiaoxiao_types_faster :
  let taoqi : Typist := { name := "淘气", characters := 200, minutes := 5 }
  let xiaoxiao : Typist := { name := "笑笑", characters := 132, minutes := 3 }
  isFaster xiaoxiao taoqi := by
  sorry

end xiaoxiao_types_faster_l3006_300686


namespace barkley_bones_l3006_300600

/-- The number of new dog bones Barkley gets at the beginning of each month -/
def monthly_new_bones : ℕ := sorry

/-- The number of months -/
def months : ℕ := 5

/-- The number of bones available after 5 months -/
def available_bones : ℕ := 8

/-- The number of bones buried after 5 months -/
def buried_bones : ℕ := 42

theorem barkley_bones : monthly_new_bones = 10 := by
  sorry

end barkley_bones_l3006_300600


namespace cube_root_equation_l3006_300693

theorem cube_root_equation (x : ℝ) : 
  (x * (x^3)^(1/2))^(1/3) = 3 → x = 3^(6/5) := by sorry

end cube_root_equation_l3006_300693


namespace average_of_new_sequence_eq_l3006_300651

/-- Given a positive integer a, this function returns the average of seven consecutive integers starting with a. -/
def average_of_seven (a : ℤ) : ℚ :=
  (7 * a + 21) / 7

/-- Given a positive integer a, this function returns the average of seven consecutive integers starting with the average of seven consecutive integers starting with a. -/
def average_of_new_sequence (a : ℤ) : ℚ :=
  let b := average_of_seven a
  (7 * ⌊b⌋ + 21) / 7

/-- Theorem stating that the average of the new sequence is equal to a + 6 -/
theorem average_of_new_sequence_eq (a : ℤ) (h : a > 0) : 
  average_of_new_sequence a = a + 6 := by
  sorry

end average_of_new_sequence_eq_l3006_300651


namespace unique_natural_number_with_specific_properties_l3006_300617

theorem unique_natural_number_with_specific_properties :
  ∀ (x n : ℕ),
    x = 5^n - 1 →
    (∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ x = p * q * r) →
    11 ∣ x →
    x = 3124 := by
  sorry

end unique_natural_number_with_specific_properties_l3006_300617


namespace problem_1_l3006_300675

theorem problem_1 : 2 * Real.tan (45 * π / 180) + (-1/2)^0 + |Real.sqrt 3 - 1| = 2 + Real.sqrt 3 := by
  sorry

end problem_1_l3006_300675


namespace point_reflection_x_axis_l3006_300645

/-- Given a point A(2,3) in the Cartesian coordinate system, 
    its coordinates with respect to the x-axis are (2,-3). -/
theorem point_reflection_x_axis : 
  let A : ℝ × ℝ := (2, 3)
  let reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  reflect_x A = (2, -3) := by
  sorry

end point_reflection_x_axis_l3006_300645


namespace assignFourFromTwentyFive_eq_303600_l3006_300630

/-- The number of ways to select and assign 4 people from a group of 25 to 4 distinct positions -/
def assignFourFromTwentyFive : ℕ := 25 * 24 * 23 * 22

/-- Theorem stating that the number of ways to select and assign 4 people from a group of 25 to 4 distinct positions is 303600 -/
theorem assignFourFromTwentyFive_eq_303600 : assignFourFromTwentyFive = 303600 := by
  sorry

end assignFourFromTwentyFive_eq_303600_l3006_300630


namespace max_value_of_trigonometric_function_l3006_300639

theorem max_value_of_trigonometric_function :
  let y : ℝ → ℝ := λ x => Real.tan (x + 3 * Real.pi / 4) - Real.tan (x + Real.pi / 4) + Real.sin (x + Real.pi / 4)
  ∃ (max_y : ℝ), max_y = Real.sqrt 2 / 2 ∧
    ∀ x, -2 * Real.pi / 3 ≤ x ∧ x ≤ -Real.pi / 2 → y x ≤ max_y :=
by sorry

end max_value_of_trigonometric_function_l3006_300639


namespace jungkook_has_most_apples_l3006_300606

def jungkook_initial : ℕ := 6
def jungkook_additional : ℕ := 3
def yoongi_apples : ℕ := 4
def yuna_apples : ℕ := 5

def jungkook_total : ℕ := jungkook_initial + jungkook_additional

theorem jungkook_has_most_apples :
  jungkook_total > yoongi_apples ∧ jungkook_total > yuna_apples :=
by sorry

end jungkook_has_most_apples_l3006_300606


namespace no_friendly_triplet_in_small_range_exists_friendly_triplet_in_large_range_l3006_300619

-- Define friendly integers
def friendly (a b c : ℤ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ (a ∣ b * c ∨ b ∣ a * c ∨ c ∣ a * b)

theorem no_friendly_triplet_in_small_range (n : ℕ) :
  ¬∃ a b c : ℤ, n^2 < a ∧ a < b ∧ b < c ∧ c < n^2 + n ∧ friendly a b c := by
  sorry

theorem exists_friendly_triplet_in_large_range (n : ℕ) :
  ∃ a b c : ℤ, n^2 < a ∧ a < b ∧ b < c ∧ c < n^2 + n + 3 * Real.sqrt n ∧ friendly a b c := by
  sorry

end no_friendly_triplet_in_small_range_exists_friendly_triplet_in_large_range_l3006_300619


namespace max_sum_arithmetic_progression_l3006_300677

/-- The first term of the arithmetic progression -/
def a₁ : ℤ := 113

/-- The common difference of the arithmetic progression -/
def d : ℤ := -4

/-- The sum of the first n terms of an arithmetic progression -/
def S (n : ℕ) : ℤ := n * (2 * a₁ + (n - 1) * d) / 2

/-- The n-th term of the arithmetic progression -/
def aₙ (n : ℕ) : ℤ := a₁ + (n - 1) * d

/-- The maximum number of terms before the sequence becomes non-positive -/
def max_n : ℕ := 29

theorem max_sum_arithmetic_progression :
  ∀ n : ℕ, S n ≤ S max_n ∧ S max_n = 1653 :=
sorry

end max_sum_arithmetic_progression_l3006_300677


namespace jackson_painting_fraction_l3006_300671

-- Define the time it takes Jackson to paint the entire garage
def total_time : ℚ := 60

-- Define the time we want to calculate the portion for
def partial_time : ℚ := 12

-- Define the fraction of the garage painted in partial_time
def fraction_painted : ℚ := partial_time / total_time

-- Theorem to prove
theorem jackson_painting_fraction :
  fraction_painted = 1 / 5 := by
  sorry

end jackson_painting_fraction_l3006_300671


namespace marble_count_theorem_l3006_300634

/-- Represents the count of marbles of each color in a bag -/
structure MarbleCount where
  red : ℕ
  blue : ℕ
  green : ℕ

/-- The ratio of marbles in the bag -/
def marbleRatio : MarbleCount := { red := 2, blue := 4, green := 6 }

/-- The number of green marbles in the bag -/
def greenMarbleCount : ℕ := 42

/-- Theorem stating the correct count of marbles given the ratio and green marble count -/
theorem marble_count_theorem (ratio : MarbleCount) (green_count : ℕ) :
  ratio = marbleRatio →
  green_count = greenMarbleCount →
  ∃ (count : MarbleCount),
    count.red = 14 ∧
    count.blue = 28 ∧
    count.green = 42 :=
by
  sorry

end marble_count_theorem_l3006_300634


namespace no_polynomial_iteration_fixed_points_l3006_300658

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- The degree of a polynomial -/
def degree (p : IntPolynomial) : ℕ := sorry

/-- A function from integers to integers -/
def IntFunction := ℤ → ℤ

/-- n-fold application of a function -/
def iterate (f : IntFunction) (n : ℕ) : IntFunction := sorry

/-- The number of fixed points of a function -/
def fixedPointCount (f : IntFunction) : ℕ := sorry

/-- Main theorem -/
theorem no_polynomial_iteration_fixed_points :
  ¬ ∃ (P : IntPolynomial) (T : IntFunction),
    degree P ≥ 1 ∧
    (∀ n : ℕ, n ≥ 1 → fixedPointCount (iterate T n) = P n) :=
sorry

end no_polynomial_iteration_fixed_points_l3006_300658


namespace cone_volume_l3006_300674

/-- Given a cone with base area 2π and lateral area 6π, its volume is 8π/3 -/
theorem cone_volume (r : ℝ) (l : ℝ) (h : ℝ) : 
  (π * r^2 = 2*π) → 
  (π * r * l = 6*π) → 
  (h^2 + r^2 = l^2) →
  (1/3 * π * r^2 * h = 8*π/3) :=
by sorry

end cone_volume_l3006_300674


namespace charlies_weight_l3006_300663

theorem charlies_weight (alice_weight charlie_weight : ℚ) 
  (sum_condition : alice_weight + charlie_weight = 240)
  (difference_condition : charlie_weight - alice_weight = charlie_weight / 3) :
  charlie_weight = 144 := by
  sorry

end charlies_weight_l3006_300663


namespace sum_first_100_odd_integers_l3006_300635

theorem sum_first_100_odd_integers : 
  (Finset.range 100).sum (fun i => 2 * (i + 1) - 1) = 10000 := by
  sorry

end sum_first_100_odd_integers_l3006_300635


namespace remainder_51_pow_2015_mod_13_l3006_300697

theorem remainder_51_pow_2015_mod_13 : 51^2015 % 13 = 12 := by
  sorry

end remainder_51_pow_2015_mod_13_l3006_300697


namespace expression_simplification_l3006_300637

theorem expression_simplification :
  ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 12) / 4) = 13.5 := by
  sorry

end expression_simplification_l3006_300637


namespace robin_hair_cut_l3006_300699

/-- Calculates the length of hair cut off given initial length, growth, and final length -/
def hair_cut_length (initial_length growth final_length : ℝ) : ℝ :=
  initial_length + growth - final_length

/-- Theorem stating that given the conditions in the problem, Robin cut off 11 inches of hair -/
theorem robin_hair_cut :
  let initial_length : ℝ := 16
  let growth : ℝ := 12
  let final_length : ℝ := 17
  hair_cut_length initial_length growth final_length = 11 := by
  sorry

end robin_hair_cut_l3006_300699


namespace expenditure_is_negative_l3006_300698

/-- Represents the recording of a monetary transaction -/
inductive MonetaryRecord
| Income (amount : ℤ)
| Expenditure (amount : ℤ)

/-- Converts a MonetaryRecord to its signed integer representation -/
def toSignedAmount (record : MonetaryRecord) : ℤ :=
  match record with
  | MonetaryRecord.Income a => a
  | MonetaryRecord.Expenditure a => -a

theorem expenditure_is_negative (income_amount expenditure_amount : ℤ) 
  (h : toSignedAmount (MonetaryRecord.Income income_amount) = income_amount) :
  toSignedAmount (MonetaryRecord.Expenditure expenditure_amount) = -expenditure_amount := by
  sorry

end expenditure_is_negative_l3006_300698


namespace find_N_l3006_300612

theorem find_N (a b c N : ℚ) : 
  a + b + c = 120 ∧
  a - 10 = N ∧
  b + 10 = N ∧
  7 * c = N →
  N = 56 := by
sorry

end find_N_l3006_300612


namespace multiply_by_112_equals_70000_l3006_300664

theorem multiply_by_112_equals_70000 (x : ℝ) : 112 * x = 70000 → x = 625 := by
  sorry

end multiply_by_112_equals_70000_l3006_300664


namespace quadratic_equation_solution_l3006_300633

theorem quadratic_equation_solution (x : ℝ) : 
  x^2 - 2*x - 4 = 0 ↔ x = 1 + Real.sqrt 5 ∨ x = 1 - Real.sqrt 5 := by
sorry

end quadratic_equation_solution_l3006_300633


namespace absolute_value_of_z_squared_minus_two_z_l3006_300627

theorem absolute_value_of_z_squared_minus_two_z (z : ℂ) : 
  z = 1 + I → Complex.abs (z^2 - 2*z) = 2 := by sorry

end absolute_value_of_z_squared_minus_two_z_l3006_300627


namespace return_journey_percentage_l3006_300642

/-- Represents the distance of a one-way trip -/
def one_way_distance : ℝ := 1

/-- Represents the total round-trip distance -/
def round_trip_distance : ℝ := 2 * one_way_distance

/-- Represents the percentage of the round-trip completed -/
def round_trip_completed_percentage : ℝ := 0.75

/-- Represents the distance traveled in the round-trip -/
def distance_traveled : ℝ := round_trip_completed_percentage * round_trip_distance

/-- Represents the distance traveled on the return journey -/
def return_journey_traveled : ℝ := distance_traveled - one_way_distance

theorem return_journey_percentage :
  return_journey_traveled / one_way_distance = 0.5 := by sorry

end return_journey_percentage_l3006_300642


namespace min_value_expression_l3006_300668

theorem min_value_expression (n : ℕ) (hn : n > 0) :
  (n : ℝ) / 2 + 50 / n ≥ 10 ∧ ∃ m : ℕ, m > 0 ∧ (m : ℝ) / 2 + 50 / m = 10 :=
sorry

end min_value_expression_l3006_300668


namespace race_time_proof_l3006_300656

/-- Given a race with the following conditions:
    - The race distance is 240 meters
    - Runner A beats runner B by either 56 meters or 7 seconds
    This theorem proves that runner A's time to complete the race is 23 seconds. -/
theorem race_time_proof (race_distance : ℝ) (distance_diff : ℝ) (time_diff : ℝ) :
  race_distance = 240 ∧ distance_diff = 56 ∧ time_diff = 7 →
  ∃ (time_A : ℝ), time_A = 23 ∧
    (race_distance / time_A = (race_distance - distance_diff) / (time_A + time_diff)) :=
by sorry

end race_time_proof_l3006_300656


namespace spending_difference_l3006_300605

def akeno_spending : ℕ := 2985

def lev_spending : ℕ := akeno_spending / 3

def ambrocio_spending : ℕ := lev_spending - 177

def total_difference : ℕ := akeno_spending - (lev_spending + ambrocio_spending)

theorem spending_difference :
  total_difference = 1172 :=
by sorry

end spending_difference_l3006_300605


namespace complex_equation_solution_l3006_300601

theorem complex_equation_solution (i z : ℂ) (hi : i * i = -1) (hz : (2 * i) / z = 1 - i) : z = -1 + i := by
  sorry

end complex_equation_solution_l3006_300601


namespace no_all_prime_arrangement_l3006_300638

/-- A card with two digits -/
structure Card where
  digit1 : Nat
  digit2 : Nat
  h_different : digit1 ≠ digit2
  h_range : digit1 < 10 ∧ digit2 < 10

/-- Function to form a two-digit number from two digits -/
def formNumber (tens : Nat) (ones : Nat) : Nat :=
  10 * tens + ones

/-- Predicate to check if a number is prime -/
def isPrime (n : Nat) : Prop :=
  n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- Main theorem statement -/
theorem no_all_prime_arrangement :
  ¬∃ (card1 card2 : Card),
    card1.digit1 ≠ card2.digit1 ∧
    card1.digit1 ≠ card2.digit2 ∧
    card1.digit2 ≠ card2.digit1 ∧
    card1.digit2 ≠ card2.digit2 ∧
    (∀ (d1 d2 : Nat),
      (d1 = card1.digit1 ∨ d1 = card1.digit2 ∨ d1 = card2.digit1 ∨ d1 = card2.digit2) →
      (d2 = card1.digit1 ∨ d2 = card1.digit2 ∨ d2 = card2.digit1 ∨ d2 = card2.digit2) →
      isPrime (formNumber d1 d2)) :=
sorry

end no_all_prime_arrangement_l3006_300638


namespace range_of_a_l3006_300626

theorem range_of_a (a : ℝ) : 
  (¬∀ x : ℝ, |1 - x| - |x - 5| < a → False) → a > 4 := by
  sorry

end range_of_a_l3006_300626


namespace thomas_salary_l3006_300689

/-- Given the average salaries of two groups, prove Thomas's salary -/
theorem thomas_salary (raj_salary roshan_salary thomas_salary : ℕ) : 
  (raj_salary + roshan_salary) / 2 = 4000 →
  (raj_salary + roshan_salary + thomas_salary) / 3 = 5000 →
  thomas_salary = 7000 := by
  sorry

end thomas_salary_l3006_300689


namespace extended_segment_endpoint_l3006_300647

/-- Given a segment with endpoints A and B, extended to point C such that BC = 1/2 * AB,
    prove that C has the calculated coordinates. -/
theorem extended_segment_endpoint (A B C : ℝ × ℝ) : 
  A = (1, -3) → 
  B = (11, 3) → 
  C.1 - B.1 = (B.1 - A.1) / 2 → 
  C.2 - B.2 = (B.2 - A.2) / 2 → 
  C = (16, 6) := by
  sorry

end extended_segment_endpoint_l3006_300647


namespace black_area_after_transformations_l3006_300690

/-- The fraction of black area remaining after one transformation -/
def remaining_fraction : ℚ := 2 / 3

/-- The number of transformations -/
def num_transformations : ℕ := 6

/-- The theorem stating the fraction of black area remaining after six transformations -/
theorem black_area_after_transformations :
  remaining_fraction ^ num_transformations = 64 / 729 := by
  sorry

end black_area_after_transformations_l3006_300690


namespace arithmetic_equation_l3006_300679

theorem arithmetic_equation : 8 / 4 - 3^2 + 4 * 5 = 13 := by sorry

end arithmetic_equation_l3006_300679


namespace max_value_of_a_l3006_300670

-- Define the operation
def matrix_op (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the theorem
theorem max_value_of_a :
  (∀ x : ℝ, matrix_op (x - 1) (a - 2) (a + 1) x ≥ 1) →
  (∀ b : ℝ, (∀ x : ℝ, matrix_op (x - 1) (b - 2) (b + 1) x ≥ 1) → b ≤ 3/2) ∧
  (∃ x : ℝ, matrix_op (x - 1) (3/2 - 2) (3/2 + 1) x ≥ 1) :=
by sorry

end max_value_of_a_l3006_300670


namespace sin_half_theta_l3006_300653

theorem sin_half_theta (θ : Real) (h1 : |Real.cos θ| = 1/5) (h2 : 5*Real.pi/2 < θ) (h3 : θ < 3*Real.pi) :
  Real.sin (θ/2) = -Real.sqrt 15 / 5 := by
  sorry

end sin_half_theta_l3006_300653


namespace right_handed_players_count_l3006_300678

theorem right_handed_players_count (total_players : ℕ) (throwers : ℕ) :
  total_players = 70 →
  throwers = 28 →
  (total_players - throwers) % 3 = 0 →
  56 = throwers + (total_players - throwers) - (total_players - throwers) / 3 := by
  sorry

end right_handed_players_count_l3006_300678


namespace bacteria_growth_l3006_300602

theorem bacteria_growth (quadruple_time : ℕ) (total_time : ℕ) (final_count : ℕ) 
  (h1 : quadruple_time = 20)
  (h2 : total_time = 4 * 60)
  (h3 : final_count = 1048576)
  (h4 : (total_time / quadruple_time : ℚ) = 12) :
  ∃ (initial_count : ℚ), 
    initial_count * (4 ^ (total_time / quadruple_time)) = final_count ∧ 
    initial_count = 1 / 16 := by
  sorry

end bacteria_growth_l3006_300602


namespace local_min_condition_l3006_300608

open Real

/-- A function f with a parameter b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - 3*b*x + b

/-- The theorem statement -/
theorem local_min_condition (b : ℝ) :
  (∃ x ∈ Set.Ioo 0 1, IsLocalMin (f b) x) → b ∈ Set.Ioo 0 1 := by
  sorry

end local_min_condition_l3006_300608


namespace fraction_division_l3006_300604

theorem fraction_division : (3 / 4) / (5 / 8) = 6 / 5 := by
  sorry

end fraction_division_l3006_300604
