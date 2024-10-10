import Mathlib

namespace min_xy_given_otimes_l370_37085

/-- The custom operation ⊗ defined for positive real numbers -/
def otimes (a b : ℝ) : ℝ := a * b - a - b

/-- Theorem stating the minimum value of xy given the conditions -/
theorem min_xy_given_otimes (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : otimes x y = 3) :
  ∀ z w : ℝ, z > 0 → w > 0 → otimes z w = 3 → x * y ≤ z * w :=
sorry

end min_xy_given_otimes_l370_37085


namespace average_of_three_l370_37011

theorem average_of_three (M : ℝ) (h1 : 12 < M) (h2 : M < 25) :
  let avg := (8 + 15 + M) / 3
  (avg = 12 ∨ avg = 15) ∧ avg ≠ 18 ∧ avg ≠ 20 ∧ avg ≠ 23 := by
  sorry

end average_of_three_l370_37011


namespace line_equation_l370_37004

/-- The equation of a line with slope 2 passing through the point (0, 3) is y = 2x + 3 -/
theorem line_equation (l : Set (ℝ × ℝ)) (slope : ℝ) (point : ℝ × ℝ) : 
  slope = 2 → 
  point = (0, 3) → 
  (∀ (x y : ℝ), (x, y) ∈ l ↔ y = 2*x + 3) :=
sorry

end line_equation_l370_37004


namespace farmer_cages_solution_l370_37003

/-- Represents the problem of determining the number of cages a farmer wants to fill -/
def farmer_cages_problem (initial_rabbits : ℕ) (additional_rabbits : ℕ) (total_rabbits : ℕ) : Prop :=
  ∃ (num_cages : ℕ) (rabbits_per_cage : ℕ),
    num_cages > 1 ∧
    initial_rabbits + additional_rabbits = total_rabbits ∧
    num_cages * rabbits_per_cage = total_rabbits

/-- The solution to the farmer's cage problem -/
theorem farmer_cages_solution :
  farmer_cages_problem 164 6 170 → ∃ (num_cages : ℕ), num_cages = 10 :=
by
  sorry

#check farmer_cages_solution

end farmer_cages_solution_l370_37003


namespace first_dog_takes_one_more_than_second_l370_37013

def dog_bone_problem (second_dog_bones : ℕ) : Prop :=
  let first_dog_bones := 3
  let third_dog_bones := 2 * second_dog_bones
  let fourth_dog_bones := 1
  let fifth_dog_bones := 2 * fourth_dog_bones
  first_dog_bones + second_dog_bones + third_dog_bones + fourth_dog_bones + fifth_dog_bones = 12

theorem first_dog_takes_one_more_than_second :
  ∃ (second_dog_bones : ℕ), dog_bone_problem second_dog_bones ∧ 3 = second_dog_bones + 1 := by
  sorry

end first_dog_takes_one_more_than_second_l370_37013


namespace invalid_paper_percentage_l370_37002

theorem invalid_paper_percentage (total_papers : ℕ) (valid_papers : ℕ) 
  (h1 : total_papers = 400)
  (h2 : valid_papers = 240) :
  (total_papers - valid_papers) * 100 / total_papers = 40 := by
  sorry

end invalid_paper_percentage_l370_37002


namespace diophantine_equation_solutions_l370_37043

theorem diophantine_equation_solutions (n : ℕ) :
  let solutions := {(a, b, c, d) : ℕ × ℕ × ℕ × ℕ | a^2 + b^2 + c^2 + d^2 = 7 * 4^n}
  solutions = {(5 * 2^(n-1), 2^(n-1), 2^(n-1), 2^(n-1)),
               (2^(n+1), 2^n, 2^n, 2^n),
               (3 * 2^(n-1), 3 * 2^(n-1), 3 * 2^(n-1), 2^(n-1))} :=
by sorry

end diophantine_equation_solutions_l370_37043


namespace domain_of_sqrt_fraction_l370_37015

theorem domain_of_sqrt_fraction (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 3)) / (Real.sqrt (8 - x))) ↔ x ∈ Set.Ici (-3) ∩ Set.Iio 8 := by
sorry

end domain_of_sqrt_fraction_l370_37015


namespace update_year_is_ninth_l370_37067

def maintenance_cost (n : ℕ) : ℚ :=
  if n ≤ 7 then 2 * n + 2 else 16 * (5/4)^(n-7)

def maintenance_sum (n : ℕ) : ℚ :=
  if n ≤ 7 then n^2 + 3*n else 80 * (5/4)^(n-7) - 10

def average_maintenance_cost (n : ℕ) : ℚ :=
  maintenance_sum n / n

theorem update_year_is_ninth :
  ∀ k, k < 9 → average_maintenance_cost k ≤ 12 ∧
  average_maintenance_cost 9 > 12 :=
sorry

end update_year_is_ninth_l370_37067


namespace page_number_added_thrice_l370_37079

/-- Given a book with n pages, if the sum of all page numbers plus twice a specific page number p equals 2046, then p = 15 -/
theorem page_number_added_thrice (n : ℕ) (p : ℕ) 
  (h : n > 0) 
  (h_sum : n * (n + 1) / 2 + 2 * p = 2046) : 
  p = 15 := by
sorry

end page_number_added_thrice_l370_37079


namespace no_zero_roots_l370_37024

-- Define the equations
def equation1 (x : ℝ) : Prop := 5 * x^2 - 15 = 35
def equation2 (x : ℝ) : Prop := (3*x-2)^2 = (2*x)^2
def equation3 (x : ℝ) : Prop := x^2 + 3*x - 4 = 2*x + 3

-- Theorem statement
theorem no_zero_roots :
  (∀ x : ℝ, equation1 x → x ≠ 0) ∧
  (∀ x : ℝ, equation2 x → x ≠ 0) ∧
  (∀ x : ℝ, equation3 x → x ≠ 0) :=
sorry

end no_zero_roots_l370_37024


namespace smallest_sum_of_three_non_coprime_integers_with_prime_sum_l370_37027

/-- Two natural numbers are not coprime if their greatest common divisor is greater than 1 -/
def not_coprime (a b : ℕ) : Prop := Nat.gcd a b > 1

/-- A natural number is prime if it's greater than 1 and its only divisors are 1 and itself -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_sum_of_three_non_coprime_integers_with_prime_sum :
  ∀ a b c : ℕ,
    a > 0 → b > 0 → c > 0 →
    (not_coprime a b ∨ not_coprime b c ∨ not_coprime a c) →
    is_prime (a + b + c) →
    ∀ x y z : ℕ,
      x > 0 → y > 0 → z > 0 →
      (not_coprime x y ∨ not_coprime y z ∨ not_coprime x z) →
      is_prime (x + y + z) →
      a + b + c ≤ x + y + z →
      a + b + c = 31 :=
sorry

end smallest_sum_of_three_non_coprime_integers_with_prime_sum_l370_37027


namespace school_vote_total_l370_37053

theorem school_vote_total (x : ℝ) : 
  (0.35 * x = 0.65 * x) ∧ 
  (0.45 * (x + 80) = 0.65 * x) →
  x + 80 = 260 := by
sorry

end school_vote_total_l370_37053


namespace problem_solution_l370_37063

def p₁ : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

def p₂ : Prop := ∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - 1 ≥ 0

theorem problem_solution : ¬p₁ ∧ p₂ := by sorry

end problem_solution_l370_37063


namespace gcf_24_72_60_l370_37078

theorem gcf_24_72_60 : Nat.gcd 24 (Nat.gcd 72 60) = 12 := by
  sorry

end gcf_24_72_60_l370_37078


namespace correct_calculation_l370_37042

theorem correct_calculation (x : ℝ) : 5 * x + 4 = 104 → (x + 5) / 4 = 6.25 := by
  sorry

end correct_calculation_l370_37042


namespace sin_75_degrees_l370_37055

theorem sin_75_degrees : 
  let sin75 := Real.sin (75 * Real.pi / 180)
  let sin45 := Real.sin (45 * Real.pi / 180)
  let cos45 := Real.cos (45 * Real.pi / 180)
  let sin30 := Real.sin (30 * Real.pi / 180)
  let cos30 := Real.cos (30 * Real.pi / 180)
  sin75 = (Real.sqrt 6 + Real.sqrt 2) / 4 ∧
  sin45 = Real.sqrt 2 / 2 ∧
  cos45 = Real.sqrt 2 / 2 ∧
  sin30 = 1 / 2 ∧
  cos30 = Real.sqrt 3 / 2 ∧
  sin75 = sin45 * cos30 + cos45 * sin30 :=
by sorry


end sin_75_degrees_l370_37055


namespace original_price_calculation_l370_37012

theorem original_price_calculation (discount_percentage : ℝ) (discounted_price : ℝ) : 
  discount_percentage = 20 ∧ discounted_price = 96 → 
  ∃ (original_price : ℝ), original_price = 120 ∧ discounted_price = original_price * (1 - discount_percentage / 100) :=
by sorry

end original_price_calculation_l370_37012


namespace john_jane_difference_l370_37097

-- Define the street width
def street_width : ℕ := 25

-- Define the block side length
def block_side : ℕ := 500

-- Define Jane's path length (same as block side)
def jane_path : ℕ := block_side

-- Define John's path length (block side + 2 * street width)
def john_path : ℕ := block_side + 2 * street_width

-- Theorem statement
theorem john_jane_difference : 
  4 * john_path - 4 * jane_path = 200 := by
  sorry

end john_jane_difference_l370_37097


namespace fraction_always_positive_l370_37018

theorem fraction_always_positive (x : ℝ) : 3 / (x^2 + 1) > 0 := by
  sorry

end fraction_always_positive_l370_37018


namespace school_girls_count_l370_37016

theorem school_girls_count (total_students sample_size : ℕ) 
  (h1 : total_students = 2000)
  (h2 : sample_size = 200)
  (h3 : ∃ (girls_in_sample : ℕ), 
    girls_in_sample + (girls_in_sample + 10) = sample_size) :
  ∃ (girls_in_school : ℕ), 
    girls_in_school = (950 : ℕ) ∧ 
    (girls_in_school : ℚ) / total_students = 
      ((sample_size / 2 - 5) : ℚ) / sample_size :=
by sorry

end school_girls_count_l370_37016


namespace sqrt_difference_equality_l370_37084

theorem sqrt_difference_equality (x a : ℝ) (m n : ℤ) (h : 0 < a) (h1 : 0 < m) (h2 : 0 < n) 
  (h3 : x + Real.sqrt (x^2 - 1) = a^((m - n : ℝ) / (2 * m * n : ℝ))) :
  x - Real.sqrt (x^2 - 1) = a^((n - m : ℝ) / (2 * m * n : ℝ)) := by
  sorry

end sqrt_difference_equality_l370_37084


namespace arithmetic_operation_proof_l370_37059

theorem arithmetic_operation_proof : 65 + 5 * 12 / (180 / 3) = 66 := by
  sorry

end arithmetic_operation_proof_l370_37059


namespace expression_evaluation_l370_37050

theorem expression_evaluation (a b : ℝ) 
  (h : |a + 1| + (b - 2)^2 = 0) : 
  2 * (3 * a^2 - a * b + 1) - (-a^2 + 2 * a * b + 1) = 16 := by
  sorry

end expression_evaluation_l370_37050


namespace expression_factorization_l370_37014

theorem expression_factorization (a : ℝ) :
  (9 * a^4 + 105 * a^3 - 15 * a^2 + 1) - (-2 * a^4 + 3 * a^3 - 4 * a^2 + 2 * a - 5) =
  (a - 3) * (11 * a^2 * (a + 1) - 2) :=
by sorry

end expression_factorization_l370_37014


namespace trigonometric_identity_l370_37073

theorem trigonometric_identity (α : ℝ) : 
  1 - Real.cos (3 * Real.pi / 2 - 3 * α) - Real.sin (3 * α / 2) ^ 2 + Real.cos (3 * α / 2) ^ 2 = 
  2 * Real.sqrt 2 * Real.cos (3 * α / 2) * Real.sin (3 * α / 2 + Real.pi / 4) := by
  sorry

end trigonometric_identity_l370_37073


namespace single_elimination_tournament_games_l370_37069

/-- The number of games played in a single-elimination tournament. -/
def gamesPlayed (n : ℕ) : ℕ :=
  n - 1

/-- Theorem: A single-elimination tournament with 21 teams requires 20 games. -/
theorem single_elimination_tournament_games :
  gamesPlayed 21 = 20 := by
  sorry

end single_elimination_tournament_games_l370_37069


namespace car_distance_proof_l370_37074

theorem car_distance_proof (initial_time : ℝ) (speed : ℝ) (time_factor : ℝ) : 
  initial_time = 6 →
  speed = 32 →
  time_factor = 3 / 2 →
  speed * (time_factor * initial_time) = 288 := by
  sorry

end car_distance_proof_l370_37074


namespace minimum_shoeing_time_l370_37087

theorem minimum_shoeing_time 
  (blacksmiths : ℕ) 
  (horses : ℕ) 
  (time_per_shoe : ℕ) 
  (h1 : blacksmiths = 48) 
  (h2 : horses = 60) 
  (h3 : time_per_shoe = 5) : 
  (horses * 4 * time_per_shoe) / blacksmiths = 25 := by
  sorry

end minimum_shoeing_time_l370_37087


namespace squats_on_third_day_l370_37082

/-- Calculates the number of squats on a given day, given the initial number and daily increase. -/
def squatsOnDay (initialSquats : ℕ) (dailyIncrease : ℕ) (day : ℕ) : ℕ :=
  initialSquats + (day * dailyIncrease)

/-- Theorem: Given an initial number of 30 squats and a daily increase of 5 squats,
    the number of squats on the third day will be 45. -/
theorem squats_on_third_day :
  squatsOnDay 30 5 2 = 45 := by
  sorry


end squats_on_third_day_l370_37082


namespace cricket_team_average_age_l370_37054

def average_age_of_team : ℝ := 25

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (wicket_keeper_age : ℝ) 
  (remaining_players_average_age : ℝ) :
  team_size = 11 →
  wicket_keeper_age = average_age_of_team + 3 →
  remaining_players_average_age = average_age_of_team - 1 →
  average_age_of_team * team_size = 
    wicket_keeper_age + 
    (team_size - 2) * remaining_players_average_age + 
    (average_age_of_team * team_size - wicket_keeper_age - (team_size - 2) * remaining_players_average_age) →
  average_age_of_team = 25 := by
sorry

end cricket_team_average_age_l370_37054


namespace worker_assignment_proof_l370_37086

/-- The number of shifts -/
def num_shifts : ℕ := 5

/-- The number of workers per shift -/
def workers_per_shift : ℕ := 2

/-- The total number of ways to assign workers -/
def total_assignments : ℕ := 45

/-- The total number of new workers -/
def total_workers : ℕ := 15

/-- Theorem: The number of ways to choose 2 workers from 15 workers is equal to 45 -/
theorem worker_assignment_proof :
  Nat.choose total_workers workers_per_shift = total_assignments :=
by sorry

end worker_assignment_proof_l370_37086


namespace two_digit_integer_problem_l370_37029

theorem two_digit_integer_problem :
  ∃ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧  -- a is a 2-digit positive integer
    10 ≤ b ∧ b < 100 ∧  -- b is a 2-digit positive integer
    a ≠ b ∧             -- a and b are different
    (a + b) / 2 = a + b / 100 ∧  -- average equals the special number
    a < b ∧             -- a is smaller than b
    a = 49 :=           -- a is 49
by sorry

end two_digit_integer_problem_l370_37029


namespace boy_bike_rest_time_l370_37035

theorem boy_bike_rest_time 
  (total_distance : ℝ) 
  (outbound_speed inbound_speed : ℝ) 
  (total_time : ℝ) :
  total_distance = 15 →
  outbound_speed = 5 →
  inbound_speed = 3 →
  total_time = 6 →
  (total_distance / 2) / outbound_speed + 
  (total_distance / 2) / inbound_speed + 
  (total_time - (total_distance / 2) / outbound_speed - (total_distance / 2) / inbound_speed) = 2 :=
by sorry

end boy_bike_rest_time_l370_37035


namespace accuracy_of_150_38_million_l370_37048

/-- Represents a number in millions with two decimal places -/
structure MillionNumber where
  value : ℝ
  isMillions : value ≥ 0
  twoDecimalPlaces : ∃ n : ℕ, value = (n : ℝ) / 100

/-- Represents the accuracy of a number in terms of place value -/
inductive PlaceValue
  | Hundred
  | Thousand
  | TenThousand
  | HundredThousand
  | Million

/-- Given a MillionNumber, returns its accuracy in terms of PlaceValue -/
def getAccuracy (n : MillionNumber) : PlaceValue :=
  PlaceValue.Hundred

/-- Theorem stating that 150.38 million is accurate to the hundred place -/
theorem accuracy_of_150_38_million :
  let n : MillionNumber := ⟨150.38, by norm_num, ⟨15038, by norm_num⟩⟩
  getAccuracy n = PlaceValue.Hundred := by
  sorry

end accuracy_of_150_38_million_l370_37048


namespace intersection_A_B_complement_union_A_B_l370_37007

-- Define the sets A and B
def A : Set ℝ := {x | (x + 3) / (x - 7) < 0}
def B : Set ℝ := {x | |x - 4| ≤ 6}

-- Theorem for part (1)
theorem intersection_A_B : A ∩ B = {x : ℝ | -2 ≤ x ∧ x < 7} := by sorry

-- Theorem for part (2)
theorem complement_union_A_B : (A ∪ B)ᶜ = {x : ℝ | x ≤ -3 ∨ x > 10} := by sorry

end intersection_A_B_complement_union_A_B_l370_37007


namespace product_units_digit_base6_l370_37021

/-- The units digit in base 6 of a number -/
def unitsDigitBase6 (n : ℕ) : ℕ := n % 6

/-- The product of 168 and 59 -/
def product : ℕ := 168 * 59

theorem product_units_digit_base6 :
  unitsDigitBase6 product = 0 := by sorry

end product_units_digit_base6_l370_37021


namespace specific_hexagon_area_l370_37098

/-- A hexagon formed by attaching six isosceles triangles to a central rectangle -/
structure Hexagon where
  /-- The base length of each isosceles triangle -/
  triangle_base : ℝ
  /-- The height of each isosceles triangle -/
  triangle_height : ℝ
  /-- The length of the central rectangle -/
  rectangle_length : ℝ
  /-- The width of the central rectangle -/
  rectangle_width : ℝ

/-- Calculate the area of the hexagon -/
def hexagon_area (h : Hexagon) : ℝ :=
  6 * (0.5 * h.triangle_base * h.triangle_height) + h.rectangle_length * h.rectangle_width

/-- Theorem stating that the area of the specific hexagon is 20 square units -/
theorem specific_hexagon_area :
  let h : Hexagon := {
    triangle_base := 2,
    triangle_height := 2,
    rectangle_length := 4,
    rectangle_width := 2
  }
  hexagon_area h = 20 := by sorry

end specific_hexagon_area_l370_37098


namespace correct_repetitions_per_bracelet_l370_37005

/-- The number of pattern repetitions per bracelet -/
def repetitions_per_bracelet : ℕ := 3

/-- The number of green beads in one pattern -/
def green_beads : ℕ := 3

/-- The number of purple beads in one pattern -/
def purple_beads : ℕ := 5

/-- The number of red beads in one pattern -/
def red_beads : ℕ := 6

/-- The number of beads in one pattern -/
def beads_per_pattern : ℕ := green_beads + purple_beads + red_beads

/-- The number of pattern repetitions per necklace -/
def repetitions_per_necklace : ℕ := 5

/-- The number of necklaces -/
def number_of_necklaces : ℕ := 10

/-- The total number of beads for 1 bracelet and 10 necklaces -/
def total_beads : ℕ := 742

theorem correct_repetitions_per_bracelet :
  repetitions_per_bracelet * beads_per_pattern +
  number_of_necklaces * repetitions_per_necklace * beads_per_pattern = total_beads :=
by sorry

end correct_repetitions_per_bracelet_l370_37005


namespace parabola_properties_l370_37095

/-- A parabola with the given properties -/
structure Parabola where
  a : ℝ
  b : ℝ
  intersectsXAxis : (a * (-1)^2 + b * (-1) + 2 = 0) ∧ (a ≠ 0)
  distanceAB : ∃ x, x ≠ -1 ∧ a * x^2 + b * x + 2 = 0 ∧ |x - (-1)| = 3
  increasingAfterA : ∀ x > -1, ∀ y > -1, 
    a * x^2 + b * x + 2 > a * y^2 + b * y + 2 → x > y

/-- The axis of symmetry and point P for the parabola -/
theorem parabola_properties (p : Parabola) :
  (∃ x, x = -(p.a + 2) / (2 * p.a) ∧ 
    ∀ y, p.a * (x + y)^2 + p.b * (x + y) + 2 = p.a * (x - y)^2 + p.b * (x - y) + 2) ∧
  (∃ x y, (x = -3 ∨ x = -2) ∧ y = -1 ∧ 
    p.a * x^2 + p.b * x + 2 = y ∧ 
    y < 0 ∧
    ∃ xB yC, p.a * xB^2 + p.b * xB + 2 = 0 ∧ yC = 2 ∧
    2 * (yC - y) = xB - x) := by
  sorry

end parabola_properties_l370_37095


namespace cosine_midline_l370_37094

/-- Given a cosine function y = a cos(bx + c) + d where a, b, c, and d are positive constants,
    if the graph oscillates between 5 and 1, then d = 3 -/
theorem cosine_midline (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) →
  d = 3 := by
  sorry

end cosine_midline_l370_37094


namespace ravi_overall_profit_l370_37093

/-- Calculates the overall profit or loss for Ravi's sales -/
theorem ravi_overall_profit (refrigerator_cost mobile_cost : ℕ)
  (refrigerator_loss_percent mobile_profit_percent : ℚ) :
  refrigerator_cost = 15000 →
  mobile_cost = 8000 →
  refrigerator_loss_percent = 4 / 100 →
  mobile_profit_percent = 10 / 100 →
  (refrigerator_cost * (1 - refrigerator_loss_percent) +
   mobile_cost * (1 + mobile_profit_percent) -
   (refrigerator_cost + mobile_cost) : ℚ) = 200 := by
  sorry

end ravi_overall_profit_l370_37093


namespace largest_number_l370_37033

theorem largest_number (a b c d e : ℝ) : 
  a = 0.9891 → b = 0.9799 → c = 0.989 → d = 0.978 → e = 0.979 →
  (a ≥ b ∧ a ≥ c ∧ a ≥ d ∧ a ≥ e) := by
  sorry

end largest_number_l370_37033


namespace average_after_removal_l370_37017

theorem average_after_removal (numbers : Finset ℝ) (sum : ℝ) :
  Finset.card numbers = 12 →
  sum / 12 = 72 →
  60 ∈ numbers →
  80 ∈ numbers →
  ((sum - 60 - 80) / 10 : ℝ) = 72.4 := by
  sorry

end average_after_removal_l370_37017


namespace triangle_side_calculation_l370_37036

theorem triangle_side_calculation (A B C : Real) (a b c : Real) :
  A = 45 * π / 180 →
  B = 60 * π / 180 →
  a = Real.sqrt 2 →
  b = Real.sqrt 3 :=
by
  sorry

end triangle_side_calculation_l370_37036


namespace snail_return_time_is_integer_l370_37088

/-- Represents the snail's position on the plane -/
structure SnailPosition :=
  (x : ℝ) (y : ℝ)

/-- Represents the snail's movement parameters -/
structure SnailMovement :=
  (speed : ℝ)
  (turnAngle : ℝ)
  (turnInterval : ℝ)

/-- Calculates the snail's position after a given time -/
def snailPositionAfterTime (initialPos : SnailPosition) (movement : SnailMovement) (time : ℝ) : SnailPosition :=
  sorry

/-- Checks if the snail has returned to the origin -/
def hasReturnedToOrigin (pos : SnailPosition) : Prop :=
  pos.x = 0 ∧ pos.y = 0

/-- Theorem: The snail can only return to the origin after an integer number of hours -/
theorem snail_return_time_is_integer 
  (movement : SnailMovement) 
  (h1 : movement.speed > 0)
  (h2 : movement.turnAngle = π / 3)
  (h3 : movement.turnInterval = 1 / 2) :
  ∀ t : ℝ, hasReturnedToOrigin (snailPositionAfterTime ⟨0, 0⟩ movement t) → ∃ n : ℕ, t = n :=
sorry

end snail_return_time_is_integer_l370_37088


namespace daily_wage_c_value_l370_37041

/-- The daily wage of worker c given the conditions of the problem -/
def daily_wage_c (days_a days_b days_c : ℕ) 
                 (wage_ratio_a wage_ratio_b wage_ratio_c : ℕ) 
                 (total_earning : ℚ) : ℚ :=
  let wage_a := total_earning * wage_ratio_a / 
    (days_a * wage_ratio_a + days_b * wage_ratio_b + days_c * wage_ratio_c)
  wage_a * wage_ratio_c / wage_ratio_a

theorem daily_wage_c_value : 
  daily_wage_c 6 9 4 3 4 5 1480 = 100 / 3 := by
  sorry

#eval daily_wage_c 6 9 4 3 4 5 1480

end daily_wage_c_value_l370_37041


namespace smallest_n_for_terminating_decimal_l370_37049

/-- A fraction a/b is a terminating decimal if and only if b can be written as 2^m * 5^n for some non-negative integers m and n. -/
def is_terminating_decimal (a b : ℕ) : Prop :=
  ∃ (m n : ℕ), b = 2^m * 5^n

/-- 50 is the smallest positive integer n such that n/(n+150) is a terminating decimal. -/
theorem smallest_n_for_terminating_decimal :
  (∀ k : ℕ, 0 < k → k < 50 → ¬ is_terminating_decimal k (k + 150)) ∧
  is_terminating_decimal 50 200 := by
sorry

end smallest_n_for_terminating_decimal_l370_37049


namespace weight_change_result_l370_37070

/-- Calculate the final weight after weight loss and gain -/
def final_weight (initial_weight : ℝ) (loss_percentage : ℝ) (weight_gain : ℝ) : ℝ :=
  initial_weight - (initial_weight * loss_percentage) + weight_gain

/-- Theorem stating that the given weight changes result in a final weight of 200 pounds -/
theorem weight_change_result : 
  final_weight 220 0.1 2 = 200 := by
  sorry

end weight_change_result_l370_37070


namespace f_properties_l370_37066

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2 * |x|

-- Theorem for the properties of f
theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 1 < x → x < y → f x < f y) ∧
  ({a : ℝ | f (|a| + 3/2) > 0} = {a : ℝ | a > 1/2 ∨ a < -1/2}) :=
by sorry

end f_properties_l370_37066


namespace circle_center_and_radius_l370_37081

theorem circle_center_and_radius :
  let eq := fun (x y : ℝ) => x^2 - 6*x + y^2 + 2*y - 9 = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = (3, -1) ∧ 
    radius = Real.sqrt 19 ∧
    ∀ (x y : ℝ), eq x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by sorry

end circle_center_and_radius_l370_37081


namespace trig_identity_l370_37006

theorem trig_identity (θ : Real) (h1 : π/2 < θ ∧ θ < π) (h2 : Real.tan (θ + π/3) = 1/2) : 
  Real.sin θ + Real.sqrt 3 * Real.cos θ = -2 * Real.sqrt 5 / 5 := by
  sorry

end trig_identity_l370_37006


namespace complex_fraction_sum_l370_37045

theorem complex_fraction_sum (a b : ℝ) (h : (2 : ℂ) / (1 - I) = a + b * I) : a + b = 2 := by
  sorry

end complex_fraction_sum_l370_37045


namespace quadratic_solution_l370_37022

theorem quadratic_solution : ∃ x : ℝ, x^2 - 5*x + 6 = 0 ↔ x = 2 ∨ x = 3 := by sorry

end quadratic_solution_l370_37022


namespace inverse_g_84_l370_37068

def g (x : ℝ) : ℝ := 3 * x^3 + 3

theorem inverse_g_84 : g⁻¹ 84 = 3 := by
  sorry

end inverse_g_84_l370_37068


namespace cooler_capacity_l370_37038

theorem cooler_capacity (c1 c2 c3 : ℝ) : 
  c1 = 100 → 
  c2 = c1 * 1.5 → 
  c3 = c2 / 2 → 
  c1 + c2 + c3 = 325 := by
sorry

end cooler_capacity_l370_37038


namespace price_is_four_l370_37096

/-- The price per bag of leaves for Bob and Johnny's leaf raking business -/
def price_per_bag (monday_bags : ℕ) (tuesday_bags : ℕ) (wednesday_bags : ℕ) (total_earnings : ℕ) : ℚ :=
  total_earnings / (monday_bags + tuesday_bags + wednesday_bags)

/-- Theorem stating that the price per bag is $4 given the conditions -/
theorem price_is_four :
  price_per_bag 5 3 9 68 = 4 := by
  sorry

end price_is_four_l370_37096


namespace three_digit_number_problem_l370_37044

-- Define the structure of a three-digit number
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  h_hundreds : hundreds < 10
  h_tens : tens < 10
  h_units : units < 10

def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.units

def ThreeDigitNumber.reverse (n : ThreeDigitNumber) : Nat :=
  100 * n.units + 10 * n.tens + n.hundreds

def ThreeDigitNumber.sumOfDigits (n : ThreeDigitNumber) : Nat :=
  n.hundreds + n.tens + n.units

theorem three_digit_number_problem (n : ThreeDigitNumber) : 
  n.toNat = 253 → 
  n.sumOfDigits = 10 ∧ 
  n.tens = n.hundreds + n.units ∧ 
  n.reverse = n.toNat + 99 := by
  sorry

#eval ThreeDigitNumber.toNat ⟨2, 5, 3, by norm_num, by norm_num, by norm_num⟩

end three_digit_number_problem_l370_37044


namespace intersection_and_parallel_line_l370_37090

-- Define the lines
def line1 (x y : ℝ) : Prop := 2 * x - y - 3 = 0
def line2 (x y : ℝ) : Prop := 4 * x - 3 * y - 5 = 0
def line3 (x y : ℝ) : Prop := 2 * x + 3 * y + 5 = 0

-- Define the intersection point
def intersection_point (x y : ℝ) : Prop := line1 x y ∧ line2 x y

-- Define parallel lines
def parallel_lines (a b c d e f : ℝ) : Prop := a * e = b * d

-- Theorem statement
theorem intersection_and_parallel_line :
  ∃ (k : ℝ), ∀ (x y : ℝ),
    intersection_point x y →
    parallel_lines 2 3 k 2 3 5 →
    2 * x + 3 * y + k = 0 →
    k = -7 :=
sorry

end intersection_and_parallel_line_l370_37090


namespace toy_store_revenue_ratio_l370_37057

/-- 
Given a toy store's revenue in three months (November, December, and January), 
prove that the ratio of January's revenue to November's revenue is 1/3.
-/
theorem toy_store_revenue_ratio 
  (revenue_nov revenue_dec revenue_jan : ℝ)
  (h1 : revenue_nov = (3/5) * revenue_dec)
  (h2 : revenue_dec = (5/2) * ((revenue_nov + revenue_jan) / 2)) :
  revenue_jan / revenue_nov = 1/3 :=
by sorry

end toy_store_revenue_ratio_l370_37057


namespace same_number_probability_l370_37071

def max_number : ℕ := 250
def billy_multiple : ℕ := 20
def bobbi_multiple : ℕ := 30

theorem same_number_probability :
  let billy_choices := (max_number - 1) / billy_multiple
  let bobbi_choices := (max_number - 1) / bobbi_multiple
  let common_choices := (max_number - 1) / (lcm billy_multiple bobbi_multiple)
  (common_choices : ℚ) / (billy_choices * bobbi_choices) = 1 / 24 := by
sorry

end same_number_probability_l370_37071


namespace sam_walking_time_l370_37077

/-- Given that Sam walks 0.75 miles in 15 minutes at a constant rate, 
    prove that it takes him 40 minutes to walk 2 miles. -/
theorem sam_walking_time (initial_distance : ℝ) (initial_time : ℝ) (target_distance : ℝ) :
  initial_distance = 0.75 ∧ 
  initial_time = 15 ∧ 
  target_distance = 2 →
  (target_distance / initial_distance) * initial_time = 40 := by
sorry

end sam_walking_time_l370_37077


namespace fifth_element_row_20_l370_37040

/-- Definition of binomial coefficient -/
def binomial (n k : ℕ) : ℕ := Nat.choose n k

/-- Pascal's triangle element at row n, position k -/
def pascal_triangle_element (n k : ℕ) : ℕ := binomial n (k - 1)

/-- The fifth element in Row 20 of Pascal's triangle is 4845 -/
theorem fifth_element_row_20 : pascal_triangle_element 20 5 = 4845 := by
  sorry

end fifth_element_row_20_l370_37040


namespace banana_orange_equivalence_l370_37037

theorem banana_orange_equivalence : 
  ∀ (banana_value orange_value : ℚ),
  (3/4 : ℚ) * 12 * banana_value = 9 * orange_value →
  (2/3 : ℚ) * 6 * banana_value = 4 * orange_value :=
by
  sorry

end banana_orange_equivalence_l370_37037


namespace total_buttons_eq_1600_l370_37083

/-- The number of 3-button shirts ordered -/
def shirts_3_button : ℕ := 200

/-- The number of 5-button shirts ordered -/
def shirts_5_button : ℕ := 200

/-- The number of buttons on a 3-button shirt -/
def buttons_per_3_button_shirt : ℕ := 3

/-- The number of buttons on a 5-button shirt -/
def buttons_per_5_button_shirt : ℕ := 5

/-- The total number of buttons used for the order -/
def total_buttons : ℕ := shirts_3_button * buttons_per_3_button_shirt + shirts_5_button * buttons_per_5_button_shirt

theorem total_buttons_eq_1600 : total_buttons = 1600 := by
  sorry

end total_buttons_eq_1600_l370_37083


namespace temperature_comparison_l370_37052

theorem temperature_comparison : -3 < -0.3 := by
  sorry

end temperature_comparison_l370_37052


namespace positive_x_squared_1024_l370_37031

theorem positive_x_squared_1024 (x : ℝ) (h1 : x > 0) (h2 : 4 * x^2 = 1024) : x = 16 := by
  sorry

end positive_x_squared_1024_l370_37031


namespace inequality_proof_l370_37008

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x^2 + y^2 + z^2 = 1) : 
  x*y/z + y*z/x + x*z/y ≥ Real.sqrt 3 ∧ 
  (x*y/z + y*z/x + x*z/y = Real.sqrt 3 ↔ x = y ∧ y = z ∧ z = Real.sqrt 3 / 3) :=
by sorry

end inequality_proof_l370_37008


namespace total_gumballs_l370_37064

/-- Represents the number of gumballs in a small package -/
def small_package : ℕ := 5

/-- Represents the number of gumballs in a medium package -/
def medium_package : ℕ := 12

/-- Represents the number of gumballs in a large package -/
def large_package : ℕ := 20

/-- Represents the number of small packages Nathan bought -/
def small_quantity : ℕ := 4

/-- Represents the number of medium packages Nathan bought -/
def medium_quantity : ℕ := 3

/-- Represents the number of large packages Nathan bought -/
def large_quantity : ℕ := 2

/-- Theorem stating the total number of gumballs Nathan ate -/
theorem total_gumballs : 
  small_quantity * small_package + 
  medium_quantity * medium_package + 
  large_quantity * large_package = 96 := by
  sorry

end total_gumballs_l370_37064


namespace trig_sum_equality_l370_37047

open Real

theorem trig_sum_equality (θ φ : ℝ) :
  (cos θ ^ 6 / cos φ ^ 2) + (sin θ ^ 6 / sin φ ^ 2) = 2 →
  (sin φ ^ 6 / sin θ ^ 2) + (cos φ ^ 6 / cos θ ^ 2) = 1 :=
by sorry

end trig_sum_equality_l370_37047


namespace shaded_area_rectangle_with_circles_l370_37089

/-- Given a rectangle with width 30 inches and length 60 inches, and four identical circles
    each tangent to two adjacent sides of the rectangle and its neighboring circles,
    the total shaded area when the circles are excluded is 1800 - 225π square inches. -/
theorem shaded_area_rectangle_with_circles :
  let rectangle_width : ℝ := 30
  let rectangle_length : ℝ := 60
  let circle_radius : ℝ := rectangle_width / 4
  let rectangle_area : ℝ := rectangle_width * rectangle_length
  let circle_area : ℝ := π * circle_radius^2
  let total_circle_area : ℝ := 4 * circle_area
  let shaded_area : ℝ := rectangle_area - total_circle_area
  shaded_area = 1800 - 225 * π :=
by sorry

end shaded_area_rectangle_with_circles_l370_37089


namespace exp_inequality_equivalence_l370_37061

theorem exp_inequality_equivalence (x : ℝ) : 1 < Real.exp x ∧ Real.exp x < 2 ↔ 0 < x ∧ x < Real.log 2 := by
  sorry

end exp_inequality_equivalence_l370_37061


namespace subjectB_least_hours_subjectB_total_hours_l370_37072

/-- Represents the study hours for each subject over a 15-week semester. -/
structure StudyHours where
  subjectA : ℕ
  subjectB : ℕ
  subjectC : ℕ
  subjectD : ℕ

/-- Calculates the total study hours for Subject A over 15 weeks. -/
def calculateSubjectA : ℕ := 3 * 5 * 15

/-- Calculates the total study hours for Subject B over 15 weeks. -/
def calculateSubjectB : ℕ := 2 * 3 * 15

/-- Calculates the total study hours for Subject C over 15 weeks. -/
def calculateSubjectC : ℕ := (4 + 3 + 3) * 15

/-- Calculates the total study hours for Subject D over 15 weeks. -/
def calculateSubjectD : ℕ := (1 * 5 + 5) * 15

/-- Creates a StudyHours structure with the calculated hours for each subject. -/
def parisStudyHours : StudyHours :=
  { subjectA := calculateSubjectA
  , subjectB := calculateSubjectB
  , subjectC := calculateSubjectC
  , subjectD := calculateSubjectD }

/-- Theorem: Subject B has the least study hours among all subjects. -/
theorem subjectB_least_hours (h : StudyHours) (h_eq : h = parisStudyHours) :
  h.subjectB ≤ h.subjectA ∧ h.subjectB ≤ h.subjectC ∧ h.subjectB ≤ h.subjectD :=
by sorry

/-- Theorem: The total study hours for Subject B is 90. -/
theorem subjectB_total_hours : parisStudyHours.subjectB = 90 :=
by sorry

end subjectB_least_hours_subjectB_total_hours_l370_37072


namespace dexter_card_boxes_l370_37010

theorem dexter_card_boxes (x : ℕ) : 
  (15 * x + 20 * (x - 3) = 255) → x = 9 := by
  sorry

end dexter_card_boxes_l370_37010


namespace at_most_one_equal_area_point_l370_37062

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A convex quadrilateral in a 2D plane -/
structure ConvexQuadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D
  convex : Bool  -- Assumption that the quadrilateral is convex

/-- Calculate the area of a triangle given its three vertices -/
def triangleArea (p1 p2 p3 : Point2D) : ℝ :=
  sorry

/-- Check if four triangles have equal areas -/
def equalAreaTriangles (p : Point2D) (quad : ConvexQuadrilateral) : Prop :=
  let areaABP := triangleArea quad.A quad.B p
  let areaBCP := triangleArea quad.B quad.C p
  let areaCDP := triangleArea quad.C quad.D p
  let areaDPA := triangleArea quad.D quad.A p
  areaABP = areaBCP ∧ areaBCP = areaCDP ∧ areaCDP = areaDPA

/-- Main theorem: There exists at most one point P that satisfies the equal area condition -/
theorem at_most_one_equal_area_point (quad : ConvexQuadrilateral) :
  ∃! p : Point2D, equalAreaTriangles p quad :=
sorry

end at_most_one_equal_area_point_l370_37062


namespace roof_length_width_difference_roof_area_is_720_length_is_5_times_width_l370_37099

/-- Represents the dimensions of a rectangular roof -/
structure RoofDimensions where
  width : ℝ
  length : ℝ

/-- The roof of an apartment building -/
def apartmentRoof : RoofDimensions where
  width := (720 / 5).sqrt
  length := 5 * (720 / 5).sqrt

theorem roof_length_width_difference : 
  apartmentRoof.length - apartmentRoof.width = 48 := by
  sorry

/-- The area of the roof -/
def roofArea (roof : RoofDimensions) : ℝ :=
  roof.length * roof.width

theorem roof_area_is_720 : roofArea apartmentRoof = 720 := by
  sorry

theorem length_is_5_times_width : 
  apartmentRoof.length = 5 * apartmentRoof.width := by
  sorry

end roof_length_width_difference_roof_area_is_720_length_is_5_times_width_l370_37099


namespace sum_of_powers_l370_37091

theorem sum_of_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 := by
  sorry

end sum_of_powers_l370_37091


namespace base7_subtraction_l370_37023

/-- Converts a base-7 number represented as a list of digits to its decimal equivalent -/
def toDecimal (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => d + 7 * acc) 0

/-- Converts a decimal number to its base-7 representation as a list of digits -/
def toBase7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- The first number in base 7 -/
def num1 : List Nat := [2, 4, 5, 6]

/-- The second number in base 7 -/
def num2 : List Nat := [1, 2, 3, 4]

/-- The expected difference in base 7 -/
def expected_diff : List Nat := [1, 2, 2, 2]

theorem base7_subtraction :
  toBase7 (toDecimal num1 - toDecimal num2) = expected_diff := by
  sorry

end base7_subtraction_l370_37023


namespace lara_age_proof_l370_37030

/-- Lara's age 7 years ago -/
def lara_age_7_years_ago : ℕ := 9

/-- Years since Lara was 9 -/
def years_since_9 : ℕ := 7

/-- Years until future age -/
def years_to_future : ℕ := 10

/-- Lara's future age -/
def lara_future_age : ℕ := lara_age_7_years_ago + years_since_9 + years_to_future

theorem lara_age_proof : lara_future_age = 26 := by
  sorry

end lara_age_proof_l370_37030


namespace semipro_max_salary_l370_37025

/-- Represents the structure of a baseball team with salary constraints -/
structure BaseballTeam where
  players : ℕ
  minSalary : ℕ
  salaryCap : ℕ

/-- Calculates the maximum possible salary for a single player in a baseball team -/
def maxPlayerSalary (team : BaseballTeam) : ℕ :=
  team.salaryCap - (team.players - 1) * team.minSalary

/-- Theorem stating the maximum possible salary for a single player
    given the specific constraints of the semipro baseball league -/
theorem semipro_max_salary :
  let team : BaseballTeam := ⟨25, 15000, 875000⟩
  maxPlayerSalary team = 515000 := by
  sorry


end semipro_max_salary_l370_37025


namespace equation_solution_l370_37080

theorem equation_solution : 
  ∃ (y₁ y₂ : ℝ), y₁ = 10/3 ∧ y₂ = -10 ∧ 
  (∀ y : ℝ, (10 - y)^2 = 4*y^2 ↔ (y = y₁ ∨ y = y₂)) := by
  sorry

end equation_solution_l370_37080


namespace vector_decomposition_l370_37051

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![0, -8, 9]
def p : Fin 3 → ℝ := ![0, -2, 1]
def q : Fin 3 → ℝ := ![3, 1, -1]
def r : Fin 3 → ℝ := ![4, 0, 1]

/-- Theorem stating that x can be expressed as a linear combination of p, q, and r -/
theorem vector_decomposition :
  x = (2 : ℝ) • p + (-4 : ℝ) • q + (3 : ℝ) • r :=
by sorry

end vector_decomposition_l370_37051


namespace black_balls_count_l370_37065

theorem black_balls_count (total : ℕ) (red : ℕ) (white_prob : ℚ) 
  (h_total : total = 100)
  (h_red : red = 30)
  (h_white_prob : white_prob = 47/100)
  (h_sum : red + (white_prob * total).floor + (total - red - (white_prob * total).floor) = total) :
  total - red - (white_prob * total).floor = 23 := by
  sorry

end black_balls_count_l370_37065


namespace softball_players_count_l370_37009

theorem softball_players_count (total : ℕ) (cricket : ℕ) (hockey : ℕ) (football : ℕ) 
  (h1 : total = 50)
  (h2 : cricket = 12)
  (h3 : hockey = 17)
  (h4 : football = 11) :
  total - (cricket + hockey + football) = 10 := by
  sorry

end softball_players_count_l370_37009


namespace convention_handshakes_specific_l370_37032

/-- The number of handshakes in a convention with multiple companies --/
def convention_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem stating the number of handshakes for the specific convention described --/
theorem convention_handshakes_specific : convention_handshakes 5 3 = 90 := by
  sorry

end convention_handshakes_specific_l370_37032


namespace train_speed_l370_37056

/-- The speed of a train given its length and time to cross a fixed point -/
theorem train_speed (length : ℝ) (time : ℝ) (h1 : length = 350) (h2 : time = 7) :
  length / time = 50 := by
  sorry

end train_speed_l370_37056


namespace income_comparison_l370_37019

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = juan * 0.8)
  (h2 : mary = tim * 1.6) : 
  mary = juan * 1.28 := by sorry

end income_comparison_l370_37019


namespace circle_area_approximation_l370_37028

/-- The area of a circle with radius 0.6 meters is 1.08 square meters when pi is approximated as 3 -/
theorem circle_area_approximation (r : ℝ) (π : ℝ) (A : ℝ) : 
  r = 0.6 → π = 3 → A = π * r^2 → A = 1.08 := by
  sorry

end circle_area_approximation_l370_37028


namespace orange_harvest_theorem_l370_37046

/-- Calculates the number of sacks of oranges after a harvest period -/
def sacks_after_harvest (sacks_harvested_per_day : ℕ) (sacks_discarded_per_day : ℕ) (harvest_days : ℕ) : ℕ :=
  (sacks_harvested_per_day - sacks_discarded_per_day) * harvest_days

/-- Proves that the number of sacks of oranges after 51 days of harvest is 153 -/
theorem orange_harvest_theorem :
  sacks_after_harvest 74 71 51 = 153 := by
  sorry

end orange_harvest_theorem_l370_37046


namespace triangle_similarity_condition_l370_37060

/-- Two triangles with side lengths a, b, c and a₁, b₁, c₁ are similar if and only if
    √(a·a₁) + √(b·b₁) + √(c·c₁) = √((a+b+c)·(a₁+b₁+c₁)) -/
theorem triangle_similarity_condition 
  (a b c a₁ b₁ c₁ : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (ha₁ : a₁ > 0) (hb₁ : b₁ > 0) (hc₁ : c₁ > 0) :
  (∃ (k : ℝ), k > 0 ∧ a₁ = k * a ∧ b₁ = k * b ∧ c₁ = k * c) ↔ 
  Real.sqrt (a * a₁) + Real.sqrt (b * b₁) + Real.sqrt (c * c₁) = 
  Real.sqrt ((a + b + c) * (a₁ + b₁ + c₁)) :=
by sorry

end triangle_similarity_condition_l370_37060


namespace intersection_M_N_l370_37001

-- Define set M
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set N
def N : Set ℝ := {x | Real.log x / Real.log 2 < 0}

-- Theorem statement
theorem intersection_M_N : M ∩ N = Set.Ioo 0 1 := by sorry

end intersection_M_N_l370_37001


namespace fibonacci_geometric_sequence_sum_l370_37092

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

def isGeometricSequence (a b c : ℕ) : Prop :=
  (fib b) ^ 2 = (fib a) * (fib c)

theorem fibonacci_geometric_sequence_sum (a b c : ℕ) :
  isGeometricSequence a b c ∧ 
  fib a ≤ fib b ∧ 
  fib b ≤ fib c ∧ 
  a + b + c = 1500 → 
  a = 499 := by
  sorry

end fibonacci_geometric_sequence_sum_l370_37092


namespace max_product_sum_l370_37076

theorem max_product_sum (A M C : ℕ) (h : A + M + C = 12) :
  (∀ A' M' C' : ℕ, A' + M' + C' = 12 → 
    A'*M'*C' + A'*M' + M'*C' + C'*A' ≤ A*M*C + A*M + M*C + C*A) →
  A*M*C + A*M + M*C + C*A = 112 :=
by sorry

end max_product_sum_l370_37076


namespace train_distance_theorem_l370_37026

/-- The distance a train can travel given its fuel efficiency and remaining coal -/
def train_distance (miles_per_coal : ℚ) (remaining_coal : ℚ) : ℚ :=
  miles_per_coal * remaining_coal

/-- Theorem: A train traveling 5 miles for every 2 pounds of coal with 160 pounds remaining can travel 400 miles -/
theorem train_distance_theorem :
  let miles_per_coal : ℚ := 5 / 2
  let remaining_coal : ℚ := 160
  train_distance miles_per_coal remaining_coal = 400 := by
sorry

end train_distance_theorem_l370_37026


namespace six_distinct_objects_arrangements_l370_37058

theorem six_distinct_objects_arrangements : Nat.factorial 6 = 720 := by
  sorry

end six_distinct_objects_arrangements_l370_37058


namespace second_business_owner_donation_l370_37075

/-- Given the fundraising conditions, prove the second business owner's donation per slice --/
theorem second_business_owner_donation
  (total_cakes : ℕ)
  (slices_per_cake : ℕ)
  (price_per_slice : ℚ)
  (first_donation_per_slice : ℚ)
  (total_raised : ℚ)
  (h1 : total_cakes = 10)
  (h2 : slices_per_cake = 8)
  (h3 : price_per_slice = 1)
  (h4 : first_donation_per_slice = 1/2)
  (h5 : total_raised = 140) :
  (total_raised - (total_cakes * slices_per_cake * (price_per_slice + first_donation_per_slice))) / (total_cakes * slices_per_cake) = 1/4 := by
  sorry

end second_business_owner_donation_l370_37075


namespace fraction_meaningful_l370_37039

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 1 / x) ↔ x ≠ 0 := by sorry

end fraction_meaningful_l370_37039


namespace point_transformation_l370_37000

/-- Rotates a point (x, y) by 180° counterclockwise around (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

/-- Reflects a point (x, y) about the line y = x -/
def reflectAboutYEqualsX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

/-- The main theorem -/
theorem point_transformation (a b : ℝ) :
  let p := (a, b)
  let rotated := rotate180 a b 2 3
  let final := reflectAboutYEqualsX rotated.1 rotated.2
  final = (5, -1) → b - a = -4 := by
sorry

end point_transformation_l370_37000


namespace sons_age_l370_37020

theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 27 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 25 := by
sorry

end sons_age_l370_37020


namespace first_play_duration_is_20_l370_37034

/-- Represents the duration of a soccer game in minutes -/
def game_duration : ℕ := 90

/-- Represents the duration of the second part of play in minutes -/
def second_play_duration : ℕ := 35

/-- Represents the duration of sideline time in minutes -/
def sideline_duration : ℕ := 35

/-- Calculates the duration of the first part of play given the total game duration,
    second part play duration, and sideline duration -/
def first_play_duration (total : ℕ) (second : ℕ) (sideline : ℕ) : ℕ :=
  total - second - sideline

theorem first_play_duration_is_20 :
  first_play_duration game_duration second_play_duration sideline_duration = 20 := by
  sorry

end first_play_duration_is_20_l370_37034
