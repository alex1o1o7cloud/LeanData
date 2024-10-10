import Mathlib

namespace successive_projections_l1324_132448

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Projection of a point onto the xOy plane -/
def proj_xOy (p : Point3D) : Point3D :=
  { x := p.x, y := p.y, z := 0 }

/-- Projection of a point onto the yOz plane -/
def proj_yOz (p : Point3D) : Point3D :=
  { x := 0, y := p.y, z := p.z }

/-- Projection of a point onto the xOz plane -/
def proj_xOz (p : Point3D) : Point3D :=
  { x := p.x, y := 0, z := p.z }

/-- The origin (0, 0, 0) -/
def origin : Point3D :=
  { x := 0, y := 0, z := 0 }

theorem successive_projections (M : Point3D) :
  proj_xOz (proj_yOz (proj_xOy M)) = origin := by
  sorry

end successive_projections_l1324_132448


namespace complex_expression_equality_l1324_132490

theorem complex_expression_equality : 
  let z₁ : ℂ := (-2 * Real.sqrt 3 + I) / (1 + 2 * Real.sqrt 3 * I)
  let z₂ : ℂ := (Real.sqrt 2 / (1 - I)) ^ 2017
  z₁ + z₂ = Real.sqrt 2 / 2 + (Real.sqrt 2 / 2 + 1) * I :=
by sorry

end complex_expression_equality_l1324_132490


namespace camp_cedar_boys_l1324_132486

theorem camp_cedar_boys (boys : ℕ) (girls : ℕ) (counselors : ℕ) : 
  girls = 3 * boys →
  counselors = 20 →
  boys + girls = 8 * counselors →
  boys = 40 := by
sorry

end camp_cedar_boys_l1324_132486


namespace ratio_sum_problem_l1324_132484

theorem ratio_sum_problem (a b c : ℝ) : 
  (a / b = 5 / 3) ∧ (c / b = 4 / 3) ∧ (b = 27) → a + b + c = 108 := by
  sorry

end ratio_sum_problem_l1324_132484


namespace fourth_sample_id_l1324_132485

/-- Represents a systematic sampling scenario -/
structure SystematicSampling where
  total_students : ℕ
  sample_size : ℕ
  known_ids : List ℕ

/-- Calculates the sampling interval -/
def sampling_interval (s : SystematicSampling) : ℕ :=
  s.total_students / s.sample_size

/-- Checks if a given ID is part of the sample -/
def is_in_sample (s : SystematicSampling) (id : ℕ) : Prop :=
  ∃ k : ℕ, id = s.known_ids.head! + k * sampling_interval s

/-- The main theorem to prove -/
theorem fourth_sample_id (s : SystematicSampling)
  (h1 : s.total_students = 44)
  (h2 : s.sample_size = 4)
  (h3 : s.known_ids = [6, 28, 39]) :
  is_in_sample s 17 := by
  sorry

#check fourth_sample_id

end fourth_sample_id_l1324_132485


namespace original_square_side_length_l1324_132423

def is_valid_square (n : ℕ) : Prop :=
  ∃ (k : ℕ), k > 0 ∧ (n + k)^2 - n^2 = 47

theorem original_square_side_length :
  ∃! (n : ℕ), is_valid_square n ∧ n > 0 :=
by
  -- The proof would go here
  sorry

end original_square_side_length_l1324_132423


namespace concert_attendance_l1324_132462

theorem concert_attendance (total_tickets : ℕ) 
  (before_start : ℚ) (after_first_song : ℚ) (during_middle : ℕ) 
  (h1 : total_tickets = 900)
  (h2 : before_start = 3/4)
  (h3 : after_first_song = 5/9)
  (h4 : during_middle = 80) : 
  total_tickets - (before_start * total_tickets + 
    after_first_song * (total_tickets - before_start * total_tickets) + 
    during_middle) = 20 := by
sorry

end concert_attendance_l1324_132462


namespace fib_2006_mod_10_l1324_132453

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fib_2006_mod_10 : fib 2006 % 10 = 3 := by
  sorry

end fib_2006_mod_10_l1324_132453


namespace gcf_of_36_48_72_l1324_132415

theorem gcf_of_36_48_72 : Nat.gcd 36 (Nat.gcd 48 72) = 12 := by
  sorry

end gcf_of_36_48_72_l1324_132415


namespace rex_cards_left_is_150_l1324_132429

/-- The number of Pokemon cards collected by Nicole -/
def nicole_cards : ℕ := 400

/-- The number of Pokemon cards collected by Cindy -/
def cindy_cards : ℕ := 2 * nicole_cards

/-- The combined total of Nicole and Cindy's cards -/
def combined_total : ℕ := nicole_cards + cindy_cards

/-- The number of Pokemon cards collected by Rex -/
def rex_cards : ℕ := combined_total / 2

/-- The number of people Rex divides his cards among (including himself) -/
def number_of_people : ℕ := 4

/-- The number of cards Rex has left after dividing his cards equally -/
def rex_cards_left : ℕ := rex_cards / number_of_people

theorem rex_cards_left_is_150 : rex_cards_left = 150 := by sorry

end rex_cards_left_is_150_l1324_132429


namespace min_squares_to_exceed_1000_l1324_132477

/-- The function that represents repeated squaring of a number -/
def repeated_square (x : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => x
  | n + 1 => (repeated_square x n) ^ 2

/-- The theorem stating that 3 is the smallest number of squaring operations needed for 5 to exceed 1000 -/
theorem min_squares_to_exceed_1000 :
  (∀ k < 3, repeated_square 5 k ≤ 1000) ∧
  (repeated_square 5 3 > 1000) :=
sorry

end min_squares_to_exceed_1000_l1324_132477


namespace circle_equation_and_tangent_lines_l1324_132470

/-- Circle C with center (a, b) and radius 5 -/
structure CircleC where
  a : ℝ
  b : ℝ
  center_on_line : a + b + 1 = 0
  passes_through_p : ((-2) - a)^2 + (0 - b)^2 = 25
  passes_through_q : (5 - a)^2 + (1 - b)^2 = 25

/-- Tangent line to circle C passing through point A(-3, 0) -/
structure TangentLine where
  k : ℝ

theorem circle_equation_and_tangent_lines (c : CircleC) :
  ((c.a = 2 ∧ c.b = -3) ∧
   (∀ x y : ℝ, (x - 2)^2 + (y + 3)^2 = 25 ↔ (x - c.a)^2 + (y - c.b)^2 = 25)) ∧
  (∃ t : TangentLine,
    (t.k = 0 ∧ ∀ x y : ℝ, y = t.k * (x + 3) ↔ x = -3) ∨
    (t.k = 8/15 ∧ ∀ x y : ℝ, y = t.k * (x + 3) ↔ y = (8/15) * (x + 3))) :=
by sorry

end circle_equation_and_tangent_lines_l1324_132470


namespace y_intercept_of_line_l1324_132482

/-- The y-intercept of the line 3x - 5y = 7 is -7/5 -/
theorem y_intercept_of_line (x y : ℝ) :
  3 * x - 5 * y = 7 → x = 0 → y = -7/5 := by
  sorry

end y_intercept_of_line_l1324_132482


namespace equation_is_quadratic_l1324_132441

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation_in_one_variable (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x^2 = 3 -/
def equation (x : ℝ) : ℝ := x^2 - 3

theorem equation_is_quadratic : is_quadratic_equation_in_one_variable equation := by
  sorry

end equation_is_quadratic_l1324_132441


namespace consecutive_numbers_with_perfect_square_digit_sums_l1324_132406

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Check if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

/-- Theorem: There exist two consecutive natural numbers greater than 1,000,000 
    whose sums of digits are perfect squares -/
theorem consecutive_numbers_with_perfect_square_digit_sums : 
  ∃ n : ℕ, n > 1000000 ∧ 
    is_perfect_square (sum_of_digits n) ∧ 
    is_perfect_square (sum_of_digits (n + 1)) := by sorry

end consecutive_numbers_with_perfect_square_digit_sums_l1324_132406


namespace work_related_emails_l1324_132467

theorem work_related_emails (total : ℕ) (spam_percent : ℚ) (promo_percent : ℚ) (social_percent : ℚ)
  (h_total : total = 1200)
  (h_spam : spam_percent = 27 / 100)
  (h_promo : promo_percent = 18 / 100)
  (h_social : social_percent = 15 / 100) :
  (total : ℚ) * (1 - (spam_percent + promo_percent + social_percent)) = 480 := by
  sorry

end work_related_emails_l1324_132467


namespace sequence_is_increasing_l1324_132427

def is_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n, a n < a (n + 1)

theorem sequence_is_increasing (a : ℕ → ℝ) 
  (h1 : a 1 < 0) 
  (h2 : ∀ n, a (n + 1) / a n = 1 / 3) : 
  is_increasing a :=
sorry

end sequence_is_increasing_l1324_132427


namespace a_plus_reward_is_ten_l1324_132480

/-- Represents the grading system and reward structure for Paul's courses. -/
structure GradingSystem where
  num_courses : ℕ
  reward_b_plus : ℚ
  reward_a : ℚ
  max_reward : ℚ

/-- Calculates the maximum reward Paul can receive given a grading system and A+ reward. -/
def max_reward (gs : GradingSystem) (reward_a_plus : ℚ) : ℚ :=
  let doubled_reward_b_plus := 2 * gs.reward_b_plus
  let doubled_reward_a := 2 * gs.reward_a
  max (gs.num_courses * doubled_reward_a)
      (((gs.num_courses - 1) * doubled_reward_a) + reward_a_plus)

/-- Theorem stating that the A+ reward must be $10 to achieve the maximum possible reward. -/
theorem a_plus_reward_is_ten (gs : GradingSystem) 
    (h_num_courses : gs.num_courses = 10)
    (h_reward_b_plus : gs.reward_b_plus = 5)
    (h_reward_a : gs.reward_a = 10)
    (h_max_reward : gs.max_reward = 190) :
    ∃ (reward_a_plus : ℚ), reward_a_plus = 10 ∧ max_reward gs reward_a_plus = gs.max_reward :=
  sorry


end a_plus_reward_is_ten_l1324_132480


namespace fraction_of_y_l1324_132471

theorem fraction_of_y (y : ℝ) (h : y > 0) : (9 * y / 20 + 3 * y / 10) / y = 3 / 4 := by
  sorry

end fraction_of_y_l1324_132471


namespace binomial_expansion_properties_l1324_132461

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The coefficient of x^r in the expansion of (x + 1/(2x))^n -/
def coeff (n r : ℕ) : ℚ := (binomial n r) * (1 / 2 ^ r)

theorem binomial_expansion_properties (n : ℕ) (h : n ≥ 2) :
  (2 * coeff n 1 = coeff n 0 + coeff n 2 ↔ n = 8) ∧
  (n = 8 → coeff n 2 = 7) := by sorry

end binomial_expansion_properties_l1324_132461


namespace area_of_region_l1324_132421

/-- Rectangle with sides of length 2 -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)
  (is_2x2 : width = 2 ∧ height = 2)

/-- Equilateral triangle with side length 2 -/
structure EquilateralTriangle :=
  (side_length : ℝ)
  (is_side_2 : side_length = 2)

/-- Region R inside rectangle and outside triangle -/
structure Region (rect : Rectangle) (tri : EquilateralTriangle) :=
  (inside_rectangle : Prop)
  (outside_triangle : Prop)
  (distance_from_AD : ℝ → Prop)

/-- The theorem to be proved -/
theorem area_of_region 
  (rect : Rectangle) 
  (tri : EquilateralTriangle) 
  (R : Region rect tri) : 
  ∃ (area : ℝ), 
    area = (4 - Real.sqrt 3) / 6 ∧ 
    (∀ x, R.distance_from_AD x → 2/3 ≤ x ∧ x ≤ 1) :=
sorry

end area_of_region_l1324_132421


namespace nancy_bills_l1324_132418

/-- The number of 5-dollar bills Nancy has -/
def num_bills : ℕ := sorry

/-- The value of each bill in dollars -/
def bill_value : ℕ := 5

/-- The total amount of money Nancy has in dollars -/
def total_money : ℕ := 45

/-- Theorem stating that Nancy has 9 five-dollar bills -/
theorem nancy_bills : num_bills = 45 / 5 := by sorry

end nancy_bills_l1324_132418


namespace yellow_purple_difference_l1324_132407

/-- Represents the composition of candies in a box of rainbow nerds -/
structure RainbowNerdsBox where
  purple : ℕ
  yellow : ℕ
  green : ℕ
  total : ℕ
  green_yellow_relation : green = yellow - 2
  total_sum : total = purple + yellow + green

/-- Theorem stating the difference between yellow and purple candies -/
theorem yellow_purple_difference (box : RainbowNerdsBox) 
  (h_purple : box.purple = 10) 
  (h_total : box.total = 36) : 
  box.yellow - box.purple = 4 := by
  sorry


end yellow_purple_difference_l1324_132407


namespace custom_op_result_l1324_132401

-- Define the custom operation
def custom_op (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem custom_op_result : custom_op (custom_op 12 8) 2 = 7 / 3 := by
  sorry

end custom_op_result_l1324_132401


namespace product_of_specific_primes_l1324_132408

def largest_one_digit_prime : ℕ := 7

def largest_two_digit_primes : List ℕ := [97, 89]

theorem product_of_specific_primes : 
  (largest_one_digit_prime * (largest_two_digit_primes.prod)) = 60431 := by
  sorry

end product_of_specific_primes_l1324_132408


namespace all_four_digit_palindromes_divisible_by_11_l1324_132442

/-- A four-digit palindrome is a number between 1000 and 9999 of the form abba where a and b are digits and a ≠ 0 -/
def FourDigitPalindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 1000 * a + 100 * b + 10 * b + a

theorem all_four_digit_palindromes_divisible_by_11 :
  ∀ n : ℕ, FourDigitPalindrome n → n % 11 = 0 := by
  sorry

#check all_four_digit_palindromes_divisible_by_11

end all_four_digit_palindromes_divisible_by_11_l1324_132442


namespace correct_comparison_l1324_132445

theorem correct_comparison :
  (-5/6 : ℚ) < -4/5 ∧
  ¬(-(-21) < -21) ∧
  ¬(-(abs (-21/2)) > 26/3) ∧
  ¬(-(abs (-23/3)) > -(-23/3)) :=
by sorry

end correct_comparison_l1324_132445


namespace improved_representation_of_100_l1324_132459

theorem improved_representation_of_100 :
  (222 / 2 : ℚ) - (22 / 2 : ℚ) = 100 := by sorry

end improved_representation_of_100_l1324_132459


namespace pencil_distribution_l1324_132457

theorem pencil_distribution (num_pens : ℕ) (num_pencils : ℕ) (num_students : ℕ) :
  num_pens = 1048 →
  num_students = 4 →
  num_pens % num_students = 0 →
  num_pencils % num_students = 0 →
  num_students = Nat.gcd num_pens num_pencils →
  num_pencils % 4 = 0 := by
  sorry

end pencil_distribution_l1324_132457


namespace room_width_calculation_l1324_132488

def room_length : ℝ := 25
def room_height : ℝ := 12
def door_length : ℝ := 6
def door_width : ℝ := 3
def window_length : ℝ := 4
def window_width : ℝ := 3
def num_windows : ℕ := 3
def cost_per_sqft : ℝ := 8
def total_cost : ℝ := 7248

theorem room_width_calculation (x : ℝ) :
  (2 * (room_length * room_height + x * room_height) - 
   (door_length * door_width + ↑num_windows * window_length * window_width)) * 
   cost_per_sqft = total_cost →
  x = 15 := by sorry

end room_width_calculation_l1324_132488


namespace hyperbola_eccentricity_l1324_132464

/-- Represents a hyperbola -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos : a > 0 ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The foci of a hyperbola -/
def foci (h : Hyperbola) : Point × Point := sorry

/-- Check if a point is on the hyperbola -/
def is_on_hyperbola (h : Hyperbola) (p : Point) : Prop := sorry

/-- Check if three points are on the same circle -/
def on_same_circle (p1 p2 p3 : Point) : Prop := sorry

/-- Check if a circle is tangent to a line segment -/
def circle_tangent_to_segment (center radius : Point) (p1 p2 : Point) : Prop := sorry

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

theorem hyperbola_eccentricity (h : Hyperbola) (p : Point) :
  let (f1, f2) := foci h
  is_on_hyperbola h p ∧
  on_same_circle f1 f2 p ∧
  circle_tangent_to_segment origin f1 p f2 →
  eccentricity h = (3 + 6 * Real.sqrt 2) / 7 := by
  sorry

end hyperbola_eccentricity_l1324_132464


namespace find_a_l1324_132472

def U : Set ℕ := {1, 3, 5, 7}

theorem find_a (M : Set ℕ) (a : ℕ) (h1 : M = {1, a}) 
  (h2 : (U \ M) = {5, 7}) : a = 3 := by
  sorry

end find_a_l1324_132472


namespace restaurant_bill_calculation_l1324_132494

/-- Calculate the total cost for a group at a restaurant where adults pay and kids eat free -/
theorem restaurant_bill_calculation (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : 
  total_people = 12 →
  num_kids = 7 →
  adult_meal_cost = 3 →
  (total_people - num_kids) * adult_meal_cost = 15 := by
sorry

end restaurant_bill_calculation_l1324_132494


namespace parallel_vectors_x_value_l1324_132443

/-- Two vectors in R² are parallel if and only if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (-1, 2)
  let b : ℝ × ℝ := (x, -1)
  parallel a b → x = 1/2 := by
  sorry

end parallel_vectors_x_value_l1324_132443


namespace soccer_challenge_kicks_l1324_132476

/-- The number of penalty kicks needed for a soccer team challenge --/
def penalty_kicks (total_players : ℕ) (goalies : ℕ) : ℕ :=
  (total_players - 1) * goalies

theorem soccer_challenge_kicks :
  penalty_kicks 25 5 = 120 :=
by sorry

end soccer_challenge_kicks_l1324_132476


namespace a_10_value_l1324_132444

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_10_value
  (a : ℕ → ℤ)
  (h_arith : arithmetic_sequence a)
  (h_7 : a 7 = 9)
  (h_13 : a 13 = -3) :
  a 10 = 3 :=
sorry

end a_10_value_l1324_132444


namespace imo_problem_6_l1324_132451

theorem imo_problem_6 (n : ℕ) (hn : n ≥ 2) :
  (∀ k : ℕ, k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) →
  (∀ k : ℕ, k ≤ n - 2 → Nat.Prime (k^2 + k + n)) := by
  sorry

end imo_problem_6_l1324_132451


namespace min_socks_for_pair_l1324_132435

/-- Represents the color of a sock -/
inductive SockColor
| White
| Blue
| Grey

/-- Represents a sock with its color and whether it has a hole -/
structure Sock :=
  (color : SockColor)
  (hasHole : Bool)

/-- The contents of the sock box -/
def sockBox : List Sock := sorry

/-- The number of socks in the box -/
def totalSocks : Nat := sockBox.length

/-- The number of socks with holes -/
def socksWithHoles : Nat := 3

/-- The number of white socks -/
def whiteSocks : Nat := 2

/-- The number of blue socks -/
def blueSocks : Nat := 3

/-- The number of grey socks -/
def greySocks : Nat := 4

/-- Theorem stating that 7 is the minimum number of socks needed to guarantee a pair without holes -/
theorem min_socks_for_pair (draw : Nat → Sock) :
  ∃ (n : Nat), n ≤ 7 ∧
  ∃ (i j : Nat), i < j ∧ j < n ∧
  (draw i).color = (draw j).color ∧
  ¬(draw i).hasHole ∧ ¬(draw j).hasHole :=
sorry

end min_socks_for_pair_l1324_132435


namespace water_level_drop_l1324_132460

/-- The water level drop in a cylindrical container when two spheres are removed -/
theorem water_level_drop (container_radius : ℝ) (sphere_diameter : ℝ) : 
  container_radius = 5 →
  sphere_diameter = 5 →
  (π * container_radius^2 * (5/3)) = (2 * (4/3) * π * (sphere_diameter/2)^3) :=
by sorry

end water_level_drop_l1324_132460


namespace trip_distance_is_3_6_miles_l1324_132475

/-- Calculates the trip distance given the initial fee, charge per segment, and total charge -/
def calculate_trip_distance (initial_fee : ℚ) (charge_per_segment : ℚ) (segment_length : ℚ) (total_charge : ℚ) : ℚ :=
  let distance_charge := total_charge - initial_fee
  let num_segments := distance_charge / charge_per_segment
  num_segments * segment_length

/-- Proves that the trip distance is 3.6 miles given the specified conditions -/
theorem trip_distance_is_3_6_miles :
  let initial_fee : ℚ := 5/2
  let charge_per_segment : ℚ := 7/20
  let segment_length : ℚ := 2/5
  let total_charge : ℚ := 113/20
  calculate_trip_distance initial_fee charge_per_segment segment_length total_charge = 18/5 := by
  sorry

#eval (18 : ℚ) / 5

end trip_distance_is_3_6_miles_l1324_132475


namespace prob_different_colors_specific_l1324_132434

/-- The probability of drawing two chips of different colors with replacement -/
def prob_different_colors (blue red yellow : ℕ) : ℚ :=
  let total := blue + red + yellow
  let prob_blue := blue / total
  let prob_red := red / total
  let prob_yellow := yellow / total
  let prob_not_blue := (red + yellow) / total
  let prob_not_red := (blue + yellow) / total
  let prob_not_yellow := (blue + red) / total
  prob_blue * prob_not_blue + prob_red * prob_not_red + prob_yellow * prob_not_yellow

/-- Theorem stating the probability of drawing two chips of different colors -/
theorem prob_different_colors_specific :
  prob_different_colors 6 5 4 = 148 / 225 := by
  sorry

end prob_different_colors_specific_l1324_132434


namespace longest_lifetime_l1324_132473

/-- A binary string is a list of booleans, where true represents 1 and false represents 0. -/
def BinaryString := List Bool

/-- The transformation function f as described in the problem. -/
def f (s : BinaryString) : BinaryString :=
  sorry

/-- The lifetime of a binary string is the number of times f can be applied until no falses remain. -/
def lifetime (s : BinaryString) : Nat :=
  sorry

/-- Generate a binary string of length n with repeated 110 pattern. -/
def repeated110 (n : Nat) : BinaryString :=
  sorry

/-- Theorem: For any n ≥ 2, the binary string with repeated 110 pattern has the longest lifetime. -/
theorem longest_lifetime (n : Nat) (h : n ≥ 2) :
  ∀ s : BinaryString, s.length = n → lifetime (repeated110 n) ≥ lifetime s :=
  sorry

end longest_lifetime_l1324_132473


namespace system_of_equations_l1324_132469

theorem system_of_equations (x y z k : ℝ) : 
  (2 * x - y + 3 * z = 9) → 
  (x + 2 * y - z = k) → 
  (-x + y + 4 * z = 6) → 
  (y = -1) → 
  (k = -3) := by
sorry

end system_of_equations_l1324_132469


namespace log_difference_equals_negative_nine_l1324_132497

theorem log_difference_equals_negative_nine :
  (Real.log 243 / Real.log 3) / (Real.log 27 / Real.log 3) -
  (Real.log 729 / Real.log 3) / (Real.log 81 / Real.log 3) = -9 := by
sorry

end log_difference_equals_negative_nine_l1324_132497


namespace toms_tickets_toms_remaining_tickets_l1324_132446

/-- Tom's arcade tickets problem -/
theorem toms_tickets (whack_a_mole : ℕ) (skee_ball : ℕ) (spent : ℕ) : ℕ :=
  let total := whack_a_mole + skee_ball
  total - spent

/-- Proof of Tom's remaining tickets -/
theorem toms_remaining_tickets : toms_tickets 32 25 7 = 50 := by
  sorry

end toms_tickets_toms_remaining_tickets_l1324_132446


namespace not_on_line_l1324_132436

-- Define the quadratic function f(x)
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the function g(x)
def g (a b c x x_1 x_2 : ℝ) : ℝ := f a b c (x - x_1) + f a b c (x - x_2)

theorem not_on_line (a b c x_1 x_2 : ℝ) 
  (h1 : ∃ x_1 x_2, f a b c x_1 = 0 ∧ f a b c x_2 = 0) -- f has two zeros
  (h2 : f a b c 1 = 2 * a) -- f(1) = 2a
  (h3 : a > c) -- a > c
  (h4 : ∀ x ∈ Set.Icc 0 1, g a b c x x_1 x_2 ≤ 2 / a) -- max of g(x) in [0,1] is 2/a
  (h5 : ∃ x ∈ Set.Icc 0 1, g a b c x x_1 x_2 = 2 / a) -- max of g(x) in [0,1] is achieved
  : a + b ≠ 1 := by
  sorry


end not_on_line_l1324_132436


namespace range_of_a_l1324_132483

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem range_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd_function f)
  (h_period : has_period f 3)
  (h_f1 : f 1 > 1)
  (h_f2015 : f 2015 = (2 * a - 3) / (a + 1)) :
  -1 < a ∧ a < 2/3 :=
sorry

end range_of_a_l1324_132483


namespace constant_value_l1324_132437

/-- A function satisfying the given conditions -/
def f (c : ℝ) : ℝ → ℝ :=
  fun x ↦ sorry

/-- The theorem stating the problem conditions and conclusion -/
theorem constant_value (c : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x + 3 * f (c - x) = x) →
  f 2 = 2 →
  c = 8 := by
  sorry

end constant_value_l1324_132437


namespace polynomial_coefficient_values_l1324_132499

theorem polynomial_coefficient_values (a₅ a₄ a₃ a₂ a₁ a₀ : ℝ) :
  (∀ x : ℝ, x^5 = a₅*(2*x+1)^5 + a₄*(2*x+1)^4 + a₃*(2*x+1)^3 + a₂*(2*x+1)^2 + a₁*(2*x+1) + a₀) →
  a₅ = 1/32 ∧ a₄ = -5/32 := by
sorry

end polynomial_coefficient_values_l1324_132499


namespace range_of_m_for_p_range_of_m_for_p_and_q_l1324_132411

-- Define the equations for p and q
def p (x y m : ℝ) : Prop := x^2 / (m + 1) + y^2 / (4 - m) = 1
def q (x y m : ℝ) : Prop := x^2 + y^2 - 2*x - 2*m*y + 5 = 0

-- Define what it means for p to be an ellipse with foci on the x-axis
def p_is_ellipse (m : ℝ) : Prop := m + 1 > 0 ∧ 4 - m > 0 ∧ m + 1 ≠ 4 - m

-- Define what it means for q to be a circle
def q_is_circle (m : ℝ) : Prop := m^2 - 4 > 0

-- Theorem 1
theorem range_of_m_for_p (m : ℝ) :
  p_is_ellipse m → m > 3/2 ∧ m < 4 :=
sorry

-- Theorem 2
theorem range_of_m_for_p_and_q (m : ℝ) :
  p_is_ellipse m ∧ q_is_circle m → m > 2 ∧ m < 4 :=
sorry

end range_of_m_for_p_range_of_m_for_p_and_q_l1324_132411


namespace binomial_8_2_l1324_132438

theorem binomial_8_2 : Nat.choose 8 2 = 28 := by
  sorry

end binomial_8_2_l1324_132438


namespace set_A_nonempty_iff_l1324_132458

def A (a : ℝ) : Set ℝ := {x : ℝ | x^2 + 2*x - a = 0}

theorem set_A_nonempty_iff (a : ℝ) : Set.Nonempty (A a) ↔ a ≥ -1 := by
  sorry

end set_A_nonempty_iff_l1324_132458


namespace perpendicular_line_equation_l1324_132430

/-- Given a line L with equation 3x - 6y = 9 and a point P(-2, 3), 
    the line perpendicular to L passing through P has equation y = -2x - 1 -/
theorem perpendicular_line_equation (x y : ℝ) :
  let L : Set (ℝ × ℝ) := {(x, y) | 3 * x - 6 * y = 9}
  let P : ℝ × ℝ := (-2, 3)
  let m : ℝ := 1/2  -- slope of the original line
  let m_perp : ℝ := -1/m  -- slope of the perpendicular line
  let perp_line : Set (ℝ × ℝ) := {(x, y) | y = m_perp * x + (P.2 - m_perp * P.1)}
  perp_line = {(x, y) | y = -2 * x - 1} := by
  sorry


end perpendicular_line_equation_l1324_132430


namespace teacher_weight_l1324_132428

theorem teacher_weight (num_students : ℕ) (avg_weight : ℝ) (weight_increase : ℝ) : 
  num_students = 24 →
  avg_weight = 35 →
  weight_increase = 0.4 →
  (num_students * avg_weight + (avg_weight + weight_increase) * (num_students + 1)) / (num_students + 1) - avg_weight = weight_increase →
  (num_students + 1) * (avg_weight + weight_increase) - num_students * avg_weight = 45 :=
by
  sorry

end teacher_weight_l1324_132428


namespace min_students_per_bench_l1324_132491

theorem min_students_per_bench (male_students : ℕ) (benches : ℕ) : 
  male_students = 29 →
  benches = 29 →
  let female_students := 4 * male_students
  let total_students := male_students + female_students
  (total_students + benches - 1) / benches = 5 := by
  sorry

end min_students_per_bench_l1324_132491


namespace trig_identity_l1324_132403

theorem trig_identity (a b : ℝ) (θ : ℝ) (h : (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + b)) :
  (Real.sin θ)^12 / a^2 + (Real.cos θ)^12 / b^2 = (a^4 + b^4) / (a + b)^6 := by
  sorry

end trig_identity_l1324_132403


namespace convex_quadrilateral_probability_l1324_132474

/-- The number of points on the circle -/
def n : ℕ := 7

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords with n points -/
def total_chords : ℕ := n.choose 2

/-- The probability of four randomly selected chords from n points on a circle forming a convex quadrilateral -/
theorem convex_quadrilateral_probability :
  (n.choose k : ℚ) / (total_chords.choose k : ℚ) = 1 / 171 :=
sorry

end convex_quadrilateral_probability_l1324_132474


namespace gena_hits_target_l1324_132419

/-- Calculates the number of hits given the total shots, initial shots, and additional shots per hit -/
def calculate_hits (total_shots initial_shots additional_shots_per_hit : ℕ) : ℕ :=
  (total_shots - initial_shots) / additional_shots_per_hit

/-- Theorem: Given the shooting range conditions, Gena hit the target 6 times -/
theorem gena_hits_target : 
  let initial_shots : ℕ := 5
  let additional_shots_per_hit : ℕ := 2
  let total_shots : ℕ := 17
  calculate_hits total_shots initial_shots additional_shots_per_hit = 6 := by
sorry

#eval calculate_hits 17 5 2

end gena_hits_target_l1324_132419


namespace area_ratio_ACEG_to_hexadecagon_l1324_132452

/-- Regular hexadecagon with vertices ABCDEFGHIJKLMNOP -/
structure RegularHexadecagon where
  vertices : Fin 16 → ℝ × ℝ
  is_regular : sorry -- Additional properties to ensure it's a regular hexadecagon

/-- Area of a regular hexadecagon -/
def area_hexadecagon (h : RegularHexadecagon) : ℝ := sorry

/-- Quadrilateral ACEG formed by connecting every fourth vertex of the hexadecagon -/
def quadrilateral_ACEG (h : RegularHexadecagon) : Set (ℝ × ℝ) := sorry

/-- Area of quadrilateral ACEG -/
def area_ACEG (h : RegularHexadecagon) : ℝ := sorry

/-- The main theorem: The ratio of the area of ACEG to the area of the hexadecagon is √2/2 -/
theorem area_ratio_ACEG_to_hexadecagon (h : RegularHexadecagon) :
  (area_ACEG h) / (area_hexadecagon h) = Real.sqrt 2 / 2 := by
  sorry

end area_ratio_ACEG_to_hexadecagon_l1324_132452


namespace arithmetic_problem_l1324_132440

theorem arithmetic_problem : 4 * (8 - 3) / 2 - 7 = 3 := by
  sorry

end arithmetic_problem_l1324_132440


namespace max_value_of_f_l1324_132466

theorem max_value_of_f (x : ℝ) (h : 0 < x ∧ x < 2) : 
  ∃ (max_val : ℝ), max_val = 16/3 ∧ ∀ y ∈ Set.Ioo 0 2, x * (8 - 3 * x) ≤ max_val :=
sorry

end max_value_of_f_l1324_132466


namespace third_ball_yarn_amount_l1324_132492

/-- The amount of yarn (in feet) used for each ball -/
structure YarnBalls where
  first : ℝ
  second : ℝ
  third : ℝ

/-- Properties of the yarn balls based on the given conditions -/
def validYarnBalls (y : YarnBalls) : Prop :=
  y.first = y.second / 2 ∧ 
  y.third = 3 * y.first ∧ 
  y.second = 18

/-- Theorem stating that the third ball uses 27 feet of yarn -/
theorem third_ball_yarn_amount (y : YarnBalls) (h : validYarnBalls y) : 
  y.third = 27 := by
  sorry

end third_ball_yarn_amount_l1324_132492


namespace landscape_playground_ratio_l1324_132424

/-- Given a rectangular landscape with specific dimensions and a playground,
    prove the ratio of the playground's area to the total landscape area. -/
theorem landscape_playground_ratio :
  ∀ (length breadth playground_area : ℝ),
    breadth = 8 * length →
    breadth = 480 →
    playground_area = 3200 →
    playground_area / (length * breadth) = 1 / 9 := by
  sorry

end landscape_playground_ratio_l1324_132424


namespace rectangle_perimeter_l1324_132468

theorem rectangle_perimeter (width length : ℝ) (h1 : width = Real.sqrt 3) (h2 : length = Real.sqrt 6) :
  2 * (width + length) = 2 * Real.sqrt 3 + 2 * Real.sqrt 6 := by
  sorry

end rectangle_perimeter_l1324_132468


namespace notebook_cost_example_l1324_132425

/-- The cost of notebooks given the number of notebooks, pages per notebook, and cost per page. -/
def notebook_cost (num_notebooks : ℕ) (pages_per_notebook : ℕ) (cost_per_page : ℚ) : ℚ :=
  (num_notebooks * pages_per_notebook : ℚ) * cost_per_page

/-- Theorem stating that the cost of 2 notebooks with 50 pages each, at 5 cents per page, is $5.00 -/
theorem notebook_cost_example : notebook_cost 2 50 (5 / 100) = 5 := by
  sorry

end notebook_cost_example_l1324_132425


namespace absolute_value_equation_solution_l1324_132465

theorem absolute_value_equation_solution :
  ∀ x : ℝ, (|2 * x + 6| = 3 * x + 9) ↔ (x = -3) :=
by sorry

end absolute_value_equation_solution_l1324_132465


namespace division_problem_l1324_132449

theorem division_problem : (-1) / (-5) / (-1/5) = -1 := by
  sorry

end division_problem_l1324_132449


namespace fraction_equality_l1324_132404

theorem fraction_equality (a b : ℝ) (h : 1/a - 1/b = 4) :
  (a - 2*a*b - b) / (2*a + 7*a*b - 2*b) = -2 := by
  sorry

end fraction_equality_l1324_132404


namespace circles_externally_tangent_l1324_132439

theorem circles_externally_tangent : 
  let circle1 := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let circle2 := {(x, y) : ℝ × ℝ | x^2 + y^2 - 6*y + 5 = 0}
  ∃ (p : ℝ × ℝ), p ∈ circle1 ∧ p ∈ circle2 ∧
  (∀ (q : ℝ × ℝ), q ≠ p → (q ∈ circle1 → q ∉ circle2) ∧ (q ∈ circle2 → q ∉ circle1)) :=
by sorry

end circles_externally_tangent_l1324_132439


namespace max_sum_of_digits_base8_less_than_1800_l1324_132409

/-- Represents the sum of digits in base 8 for a natural number -/
def sumOfDigitsBase8 (n : ℕ) : ℕ := sorry

/-- The greatest possible sum of digits in base 8 for numbers less than 1800 -/
def maxSumOfDigitsBase8LessThan1800 : ℕ := 23

/-- Theorem stating that the maximum sum of digits in base 8 for positive integers less than 1800 is 23 -/
theorem max_sum_of_digits_base8_less_than_1800 :
  ∀ n : ℕ, 0 < n → n < 1800 → sumOfDigitsBase8 n ≤ maxSumOfDigitsBase8LessThan1800 ∧
  ∃ m : ℕ, 0 < m ∧ m < 1800 ∧ sumOfDigitsBase8 m = maxSumOfDigitsBase8LessThan1800 :=
sorry

end max_sum_of_digits_base8_less_than_1800_l1324_132409


namespace english_sample_count_l1324_132479

/-- Represents the number of books for each subject -/
structure BookCount where
  chinese : ℕ
  math : ℕ
  english : ℕ

/-- Represents the ratio of books for each subject -/
structure BookRatio where
  chinese : ℕ
  math : ℕ
  english : ℕ

/-- Given a ratio of Chinese to English books and the number of Chinese books sampled,
    calculate the number of English books that should be sampled using stratified sampling. -/
def stratifiedSample (ratio : BookRatio) (chineseSampled : ℕ) : ℕ :=
  (ratio.english * chineseSampled) / ratio.chinese

/-- Theorem stating that given the specified ratio and number of Chinese books sampled,
    the number of English books to be sampled is 25. -/
theorem english_sample_count (ratio : BookRatio) (h1 : ratio.chinese = 2) (h2 : ratio.english = 5) :
  stratifiedSample ratio 10 = 25 := by
  sorry

#check english_sample_count

end english_sample_count_l1324_132479


namespace lcm_of_24_and_16_l1324_132405

theorem lcm_of_24_and_16 :
  let n : ℕ := 24
  let m : ℕ := 16
  Nat.gcd n m = 8 →
  Nat.lcm n m = 48 := by
sorry

end lcm_of_24_and_16_l1324_132405


namespace candidates_per_state_l1324_132498

theorem candidates_per_state (total_candidates : ℕ) : 
  (total_candidates : ℝ) * 0.06 + 80 = total_candidates * 0.07 → 
  total_candidates = 8000 := by
  sorry

end candidates_per_state_l1324_132498


namespace total_cars_count_l1324_132455

/-- The number of cars owned by Cathy, Lindsey, Carol, and Susan -/
def total_cars (cathy lindsey carol susan : ℕ) : ℕ :=
  cathy + lindsey + carol + susan

/-- Theorem stating the total number of cars owned by all four people -/
theorem total_cars_count :
  ∀ (cathy lindsey carol susan : ℕ),
    cathy = 5 →
    lindsey = cathy + 4 →
    carol = 2 * cathy →
    susan = carol - 2 →
    total_cars cathy lindsey carol susan = 32 := by
  sorry

end total_cars_count_l1324_132455


namespace fraction_product_equals_seven_fifty_fourths_l1324_132416

theorem fraction_product_equals_seven_fifty_fourths : 
  (7 : ℚ) / 4 * 8 / 12 * 14 / 6 * 18 / 30 * 16 / 24 * 35 / 49 * 27 / 54 * 40 / 20 = 7 / 54 := by
  sorry

end fraction_product_equals_seven_fifty_fourths_l1324_132416


namespace intersection_of_lines_l1324_132426

/-- The intersection point of two lines in 2D space -/
def intersection_point (a b c d e f : ℝ) : ℝ × ℝ := sorry

/-- Theorem: The point (-1, -2) is the unique intersection of the given lines -/
theorem intersection_of_lines :
  let line1 : ℝ → ℝ → Prop := λ x y => 2 * x + 3 * y + 8 = 0
  let line2 : ℝ → ℝ → Prop := λ x y => x - y - 1 = 0
  let point := (-1, -2)
  (line1 point.1 point.2 ∧ line2 point.1 point.2) ∧
  (∀ x y, line1 x y ∧ line2 x y → (x, y) = point) := by
sorry

end intersection_of_lines_l1324_132426


namespace multiply_by_number_l1324_132450

theorem multiply_by_number (x : ℝ) (n : ℝ) : x = 5 → x * n = (16 - x) + 4 → n = 3 := by
  sorry

end multiply_by_number_l1324_132450


namespace probability_triangle_or_circle_l1324_132431

theorem probability_triangle_or_circle (total : ℕ) (triangles : ℕ) (circles : ℕ) 
  (h1 : total = 10) 
  (h2 : triangles = 4) 
  (h3 : circles = 4) : 
  (triangles + circles : ℚ) / total = 4 / 5 := by
  sorry

end probability_triangle_or_circle_l1324_132431


namespace no_real_solution_log_equation_l1324_132487

theorem no_real_solution_log_equation :
  ¬ ∃ (x : ℝ), (Real.log (x + 5) + Real.log (x - 3) = Real.log (x^2 - 8*x + 15)) ∧
               (x + 5 > 0) ∧ (x - 3 > 0) ∧ (x^2 - 8*x + 15 > 0) :=
by sorry

end no_real_solution_log_equation_l1324_132487


namespace complex_equation_roots_l1324_132422

theorem complex_equation_roots : ∃ (z₁ z₂ : ℂ), 
  z₁ = 3 - I ∧ z₂ = -2 + I ∧ 
  z₁^2 - z₁ = 5 - 5*I ∧ 
  z₂^2 - z₂ = 5 - 5*I := by
  sorry

end complex_equation_roots_l1324_132422


namespace main_theorem_l1324_132463

/-- The logarithm function with base 2 -/
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

/-- The main function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := log2 (x + a)

/-- The companion function g(x) -/
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (1/2) * log2 (4*x + a)

/-- The difference function F(x) -/
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

theorem main_theorem (a : ℝ) (h : a > 0) :
  (∀ x, f a x < -1 ↔ -a < x ∧ x < 1/2 - a) ∧
  (∀ x ∈ Set.Ioo 0 2, f a x < g a x ↔ 0 < a ∧ a ≤ 1) ∧
  (∃ M, M = 1 - (1/2) * log2 3 ∧ 
    ∀ x ∈ Set.Ioo 0 2, |F 1 x| ≤ M ∧
    ∃ x₀ ∈ Set.Ioo 0 2, |F 1 x₀| = M) :=
by sorry

end main_theorem_l1324_132463


namespace negation_equivalence_l1324_132495

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x ≤ 1 ∨ x^2 > 4) ↔ (∃ x : ℝ, x > 1 ∧ x^2 ≤ 4) := by
  sorry

end negation_equivalence_l1324_132495


namespace remainder_sum_squares_mod_11_l1324_132433

theorem remainder_sum_squares_mod_11 :
  (2 * (88134^2 + 88135^2 + 88136^2 + 88137^2 + 88138^2 + 88139^2)) % 11 = 3 := by
  sorry

end remainder_sum_squares_mod_11_l1324_132433


namespace f_properties_l1324_132481

/-- The function f(x) defined as 2 / (2^x + 1) + m -/
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 / (2^x + 1) + m

/-- f is an odd function -/
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

/-- f is decreasing on ℝ -/
def is_decreasing (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x > f y

theorem f_properties (m : ℝ) :
  (is_odd (f · m) → m = -1) ∧
  is_decreasing (f · m) ∧
  (∀ x ≤ 1, f x m ≥ f 1 m) ∧
  f (-1) m = 4/3 + m :=
sorry

end f_properties_l1324_132481


namespace largest_when_third_digit_changed_l1324_132456

def original_number : ℚ := 0.08765

def change_third_digit : ℚ := 0.08865
def change_fourth_digit : ℚ := 0.08785
def change_fifth_digit : ℚ := 0.08768

theorem largest_when_third_digit_changed :
  change_third_digit > change_fourth_digit ∧
  change_third_digit > change_fifth_digit :=
by sorry

end largest_when_third_digit_changed_l1324_132456


namespace church_capacity_l1324_132402

/-- Calculates the number of usable chairs in a church with three sections -/
def total_usable_chairs : ℕ :=
  let section1_rows : ℕ := 15
  let section1_chairs_per_row : ℕ := 8
  let section1_unusable_per_row : ℕ := 3
  let section2_rows : ℕ := 20
  let section2_chairs_per_row : ℕ := 6
  let section2_unavailable_rows : ℕ := 2
  let section3_rows : ℕ := 25
  let section3_chairs_per_row : ℕ := 10
  let section3_unusable_every_second : ℕ := 5

  let section1_usable := section1_rows * (section1_chairs_per_row - section1_unusable_per_row)
  let section2_usable := (section2_rows - section2_unavailable_rows) * section2_chairs_per_row
  let section3_usable := (section3_rows / 2) * section3_chairs_per_row + 
                         (section3_rows - section3_rows / 2) * (section3_chairs_per_row - section3_unusable_every_second)

  section1_usable + section2_usable + section3_usable

theorem church_capacity : total_usable_chairs = 373 := by
  sorry

end church_capacity_l1324_132402


namespace choir_arrangement_l1324_132414

theorem choir_arrangement (n : ℕ) : n ≥ 32400 ∧ 
  (∃ k : ℕ, n = k^2) ∧ 
  n % 8 = 0 ∧ n % 9 = 0 ∧ n % 10 = 0 →
  n = 32400 :=
sorry

end choir_arrangement_l1324_132414


namespace bookshelf_picking_l1324_132410

theorem bookshelf_picking (english_books math_books : ℕ) 
  (h1 : english_books = 6) 
  (h2 : math_books = 2) : 
  english_books + math_books = 8 := by
  sorry

end bookshelf_picking_l1324_132410


namespace difference_of_squares_255_745_l1324_132478

theorem difference_of_squares_255_745 : 255^2 - 745^2 = -490000 := by
  sorry

end difference_of_squares_255_745_l1324_132478


namespace binomial_coefficient_identity_l1324_132420

theorem binomial_coefficient_identity (r m k : ℕ) (h1 : k ≤ m) (h2 : m ≤ r) :
  (Nat.choose r m) * (Nat.choose m k) = (Nat.choose r k) * (Nat.choose (r - k) (m - k)) := by
  sorry

end binomial_coefficient_identity_l1324_132420


namespace previous_painting_price_l1324_132489

/-- 
Given a painter whose most recent painting sold for $44,000, and this price is $1000 less than 
five times more than his previous painting, prove that the price of the previous painting was $9,000.
-/
theorem previous_painting_price (recent_price previous_price : ℕ) : 
  recent_price = 44000 ∧ 
  recent_price = 5 * previous_price - 1000 →
  previous_price = 9000 := by
sorry

end previous_painting_price_l1324_132489


namespace correct_calculation_l1324_132400

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * x^2 * y = x^2 * y := by
  sorry

end correct_calculation_l1324_132400


namespace female_emu_ratio_is_half_l1324_132413

/-- Represents the emu farm setup and egg production --/
structure EmuFarm where
  num_pens : ℕ
  emus_per_pen : ℕ
  eggs_per_week : ℕ

/-- Calculates the ratio of female emus to total emus --/
def female_emu_ratio (farm : EmuFarm) : ℚ :=
  let total_emus := farm.num_pens * farm.emus_per_pen
  let eggs_per_day := farm.eggs_per_week / 7
  eggs_per_day / total_emus

/-- Theorem stating that the ratio of female emus to total emus is 1/2 --/
theorem female_emu_ratio_is_half (farm : EmuFarm) 
    (h1 : farm.num_pens = 4)
    (h2 : farm.emus_per_pen = 6)
    (h3 : farm.eggs_per_week = 84) : 
  female_emu_ratio farm = 1/2 := by
  sorry

#eval female_emu_ratio ⟨4, 6, 84⟩

end female_emu_ratio_is_half_l1324_132413


namespace equal_area_triangle_octagon_ratio_l1324_132493

/-- The ratio of side lengths of an equilateral triangle and a regular octagon with equal areas -/
theorem equal_area_triangle_octagon_ratio :
  ∀ (s_t s_o : ℝ),
  s_t > 0 → s_o > 0 →
  (s_t^2 * Real.sqrt 3) / 4 = 2 * s_o^2 * (1 + Real.sqrt 2) →
  s_t / s_o = Real.sqrt (8 * Real.sqrt 3 * (1 + Real.sqrt 2) / 3) :=
by sorry


end equal_area_triangle_octagon_ratio_l1324_132493


namespace sin_210_degrees_l1324_132432

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end sin_210_degrees_l1324_132432


namespace choco_given_away_l1324_132447

/-- Represents the number of cookies in a dozen. -/
def dozen : ℕ := 12

/-- Represents the number of dozens of oatmeal raisin cookies baked. -/
def oatmeal_baked : ℚ := 3

/-- Represents the number of dozens of sugar cookies baked. -/
def sugar_baked : ℚ := 2

/-- Represents the number of dozens of chocolate chip cookies baked. -/
def choco_baked : ℚ := 4

/-- Represents the number of dozens of oatmeal raisin cookies given away. -/
def oatmeal_given : ℚ := 2

/-- Represents the number of dozens of sugar cookies given away. -/
def sugar_given : ℚ := 3/2

/-- Represents the total number of cookies Ann keeps. -/
def cookies_kept : ℕ := 36

/-- Theorem stating the number of dozens of chocolate chip cookies given away. -/
theorem choco_given_away : 
  (oatmeal_baked * dozen + sugar_baked * dozen + choco_baked * dozen - 
   oatmeal_given * dozen - sugar_given * dozen - cookies_kept) / dozen = 5/2 := by
  sorry


end choco_given_away_l1324_132447


namespace domain_of_g_l1324_132417

-- Define the function f with domain [-2, 4]
def f : Set ℝ := { x : ℝ | -2 ≤ x ∧ x ≤ 4 }

-- Define the function g as g(x) = f(x) + f(-x)
def g (x : ℝ) : Prop := x ∈ f ∧ (-x) ∈ f

-- Theorem stating that the domain of g is [-2, 2]
theorem domain_of_g : { x : ℝ | g x } = { x : ℝ | -2 ≤ x ∧ x ≤ 2 } := by
  sorry

end domain_of_g_l1324_132417


namespace shells_calculation_l1324_132454

/-- Given an initial amount of shells and an additional amount added, 
    calculate the total amount of shells -/
def total_shells (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that with 5 pounds initial and 23 pounds added, 
    the total is 28 pounds -/
theorem shells_calculation :
  total_shells 5 23 = 28 := by
  sorry

end shells_calculation_l1324_132454


namespace tank_filling_time_l1324_132412

def pipe1_rate : ℚ := 1 / 8
def pipe2_rate : ℚ := 1 / 12

def combined_rate : ℚ := pipe1_rate + pipe2_rate

theorem tank_filling_time : (1 : ℚ) / combined_rate = 24 / 5 := by sorry

end tank_filling_time_l1324_132412


namespace keaton_ladder_climbs_l1324_132496

/-- Proves that Keaton climbed the ladder 20 times given the problem conditions -/
theorem keaton_ladder_climbs : 
  let keaton_ladder_height : ℕ := 30 * 12  -- 30 feet in inches
  let reece_ladder_height : ℕ := (30 - 4) * 12  -- 26 feet in inches
  let reece_climbs : ℕ := 15
  let total_length : ℕ := 11880  -- in inches
  ∃ (keaton_climbs : ℕ), 
    keaton_climbs * keaton_ladder_height + reece_climbs * reece_ladder_height = total_length ∧ 
    keaton_climbs = 20 :=
by
  sorry


end keaton_ladder_climbs_l1324_132496
