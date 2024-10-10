import Mathlib

namespace prob_two_sunny_days_l940_94015

/-- The probability of exactly 2 sunny days in a 5-day period with 75% chance of rain each day -/
theorem prob_two_sunny_days : 
  let n : ℕ := 5  -- Total number of days
  let p : ℚ := 3/4  -- Probability of rain each day
  let k : ℕ := 2  -- Number of sunny days we want
  Nat.choose n k * (1 - p)^k * p^(n - k) = 135/512 :=
by sorry

end prob_two_sunny_days_l940_94015


namespace max_roses_for_680_l940_94008

/-- Represents the pricing structure for roses -/
structure RosePricing where
  individual_price : ℚ
  dozen_price : ℚ
  two_dozen_price : ℚ
  five_dozen_price : ℚ
  discount_rate : ℚ
  discount_threshold : ℕ

/-- Calculates the maximum number of roses that can be purchased given a budget -/
def max_roses_purchased (pricing : RosePricing) (budget : ℚ) : ℕ :=
  sorry

/-- The specific pricing structure given in the problem -/
def problem_pricing : RosePricing :=
  { individual_price := 9/2,
    dozen_price := 36,
    two_dozen_price := 50,
    five_dozen_price := 110,
    discount_rate := 1/10,
    discount_threshold := 36 }

theorem max_roses_for_680 :
  max_roses_purchased problem_pricing 680 = 364 :=
sorry

end max_roses_for_680_l940_94008


namespace penny_splitting_game_result_l940_94032

/-- Represents the result of the penny splitting game. -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins

/-- The penny splitting game. -/
def pennySplittingGame (n : ℕ) : GameResult :=
  sorry

/-- Theorem stating the conditions for each player's victory. -/
theorem penny_splitting_game_result (n : ℕ) (h : n ≥ 3) :
  pennySplittingGame n = 
    if n = 3 ∨ n % 2 = 0 then
      GameResult.FirstPlayerWins
    else
      GameResult.SecondPlayerWins :=
  sorry

end penny_splitting_game_result_l940_94032


namespace distance_between_5th_and_30th_red_light_l940_94040

/-- Represents the color of a light in the sequence -/
inductive LightColor
  | Red
  | Green

/-- Calculates the position of a light in the sequence given its number and color -/
def lightPosition (n : Nat) (color : LightColor) : Nat :=
  match color with
  | LightColor.Red => (n - 1) / 3 * 7 + (n - 1) % 3 + 1
  | LightColor.Green => (n - 1) / 4 * 7 + (n - 1) % 4 + 4

/-- The spacing between lights in inches -/
def lightSpacing : Nat := 8

/-- The number of inches in a foot -/
def inchesPerFoot : Nat := 12

/-- Calculates the distance in feet between two lights given their positions -/
def distanceBetweenLights (pos1 pos2 : Nat) : Nat :=
  ((pos2 - pos1) * lightSpacing) / inchesPerFoot

theorem distance_between_5th_and_30th_red_light :
  distanceBetweenLights (lightPosition 5 LightColor.Red) (lightPosition 30 LightColor.Red) = 41 := by
  sorry


end distance_between_5th_and_30th_red_light_l940_94040


namespace square_six_z_minus_five_l940_94060

theorem square_six_z_minus_five (z : ℝ) (hz : 3 * z^2 + 2 * z = 5 * z + 11) : 
  (6 * z - 5)^2 = 141 := by
  sorry

end square_six_z_minus_five_l940_94060


namespace distance_n_n_l940_94059

/-- The distance function for a point (a,b) on the polygonal path -/
def distance (a b : ℕ) : ℕ := sorry

/-- The theorem stating that the distance of (n,n) is n^2 + n -/
theorem distance_n_n (n : ℕ) : distance n n = n^2 + n := by sorry

end distance_n_n_l940_94059


namespace point_movement_theorem_point_M_movement_l940_94031

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Moves a point on the number line -/
def move (p : Point) (distance : ℝ) : Point :=
  ⟨p.value + distance⟩

theorem point_movement_theorem (M N : Point) :
  (M.value = 9) →
  (move (move N (-4)) 6 = M) →
  N.value = 7 :=
sorry

theorem point_M_movement (M : Point) :
  (M.value = 9) →
  (∃ (new_M : Point), (move M 4 = new_M ∨ move M (-4) = new_M) ∧ (new_M.value = 5 ∨ new_M.value = 13)) :=
sorry

end point_movement_theorem_point_M_movement_l940_94031


namespace heart_then_king_probability_l940_94025

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of hearts in a standard deck -/
def num_hearts : ℕ := 13

/-- The number of kings in a standard deck -/
def num_kings : ℕ := 4

/-- The probability of drawing a heart first and a king second from a standard deck -/
theorem heart_then_king_probability :
  (num_hearts / deck_size) * ((num_kings - 1) / (deck_size - 1)) +
  ((num_hearts - 1) / deck_size) * (num_kings / (deck_size - 1)) =
  1 / deck_size :=
sorry

end heart_then_king_probability_l940_94025


namespace cylinder_volume_equality_l940_94096

theorem cylinder_volume_equality (x : ℝ) (hx : x > 0) : 
  (π * (7 + x)^2 * 5 = π * 7^2 * (5 + 2*x)) → x = 28/5 := by
  sorry

end cylinder_volume_equality_l940_94096


namespace total_marbles_l940_94077

/-- Given a bag of marbles with red, blue, green, and yellow marbles in the ratio 3:4:2:5,
    and 30 yellow marbles, prove that the total number of marbles is 84. -/
theorem total_marbles (red blue green yellow total : ℕ) 
  (h_ratio : red + blue + green + yellow = total)
  (h_proportion : 3 * yellow = 5 * red ∧ 4 * yellow = 5 * blue ∧ 2 * yellow = 5 * green)
  (h_yellow : yellow = 30) : total = 84 := by
  sorry

end total_marbles_l940_94077


namespace line_plane_zero_angle_l940_94024

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- The angle between a line and a plane -/
def angle_line_plane (l : Line3D) (p : Plane3D) : ℝ :=
  sorry

/-- A line is parallel to a plane -/
def is_parallel (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- A line lies within a plane -/
def lies_within (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- If the angle between a line and a plane is 0°, then the line is either parallel to the plane or lies within it -/
theorem line_plane_zero_angle (l : Line3D) (p : Plane3D) :
  angle_line_plane l p = 0 → is_parallel l p ∨ lies_within l p :=
sorry

end line_plane_zero_angle_l940_94024


namespace sara_lunch_cost_l940_94052

/-- The cost of Sara's lunch -/
def lunch_cost (hotdog_price salad_price : ℚ) : ℚ :=
  hotdog_price + salad_price

/-- Theorem: Sara's lunch cost $10.46 -/
theorem sara_lunch_cost :
  lunch_cost 5.36 5.10 = 10.46 := by
  sorry

end sara_lunch_cost_l940_94052


namespace grocery_store_inventory_l940_94076

theorem grocery_store_inventory (regular_soda diet_soda apples : ℕ) : 
  regular_soda = 79 → 
  diet_soda = 53 → 
  regular_soda - diet_soda = 26 → 
  ¬∃ f : ℕ → ℕ → ℕ, f regular_soda diet_soda = apples :=
by sorry

end grocery_store_inventory_l940_94076


namespace school_dinner_theatre_attendance_l940_94004

theorem school_dinner_theatre_attendance
  (child_ticket_price : ℕ)
  (adult_ticket_price : ℕ)
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (h1 : child_ticket_price = 6)
  (h2 : adult_ticket_price = 9)
  (h3 : total_tickets = 225)
  (h4 : total_revenue = 1875) :
  ∃ (child_tickets adult_tickets : ℕ),
    child_tickets + adult_tickets = total_tickets ∧
    child_tickets * child_ticket_price + adult_tickets * adult_ticket_price = total_revenue ∧
    adult_tickets = 175 :=
sorry

end school_dinner_theatre_attendance_l940_94004


namespace angel_letters_count_l940_94090

theorem angel_letters_count :
  ∀ (large_envelopes small_letters letters_per_large : ℕ),
    large_envelopes = 30 →
    letters_per_large = 2 →
    small_letters = 20 →
    large_envelopes * letters_per_large + small_letters = 80 :=
by
  sorry

end angel_letters_count_l940_94090


namespace sophomore_latin_probability_l940_94029

/-- Represents the percentage of students in each class -/
structure ClassDistribution :=
  (freshmen : ℚ)
  (sophomores : ℚ)
  (juniors : ℚ)
  (seniors : ℚ)

/-- Represents the percentage of students taking Latin in each class -/
structure LatinRates :=
  (freshmen : ℚ)
  (sophomores : ℚ)
  (juniors : ℚ)
  (seniors : ℚ)

/-- The probability that a randomly chosen Latin student is a sophomore -/
def sophomoreProbability (dist : ClassDistribution) (rates : LatinRates) : ℚ :=
  (dist.sophomores * rates.sophomores) /
  (dist.freshmen * rates.freshmen + dist.sophomores * rates.sophomores +
   dist.juniors * rates.juniors + dist.seniors * rates.seniors)

theorem sophomore_latin_probability :
  let dist : ClassDistribution := {
    freshmen := 2/5, sophomores := 3/10, juniors := 1/5, seniors := 1/10
  }
  let rates : LatinRates := {
    freshmen := 1, sophomores := 4/5, juniors := 1/2, seniors := 1/5
  }
  sophomoreProbability dist rates = 6/19 := by sorry

end sophomore_latin_probability_l940_94029


namespace total_selling_price_l940_94005

/-- Calculate the total selling price of three items given their cost prices and profit/loss percentages -/
theorem total_selling_price
  (cost_A cost_B cost_C : ℝ)
  (loss_A gain_B loss_C : ℝ)
  (h_cost_A : cost_A = 1400)
  (h_cost_B : cost_B = 2500)
  (h_cost_C : cost_C = 3200)
  (h_loss_A : loss_A = 0.15)
  (h_gain_B : gain_B = 0.10)
  (h_loss_C : loss_C = 0.05) :
  cost_A * (1 - loss_A) + cost_B * (1 + gain_B) + cost_C * (1 - loss_C) = 6980 :=
by sorry

end total_selling_price_l940_94005


namespace stock_certificate_tearing_l940_94006

theorem stock_certificate_tearing : ¬ ∃ k : ℕ, 1 + 7 * k = 2002 := by
  sorry

end stock_certificate_tearing_l940_94006


namespace tax_discount_commute_l940_94066

theorem tax_discount_commute (p t d : ℝ) (h1 : 0 ≤ p) (h2 : 0 ≤ t) (h3 : 0 ≤ d) (h4 : d ≤ 1) :
  p * (1 + t) * (1 - d) = p * (1 - d) * (1 + t) :=
by sorry

#check tax_discount_commute

end tax_discount_commute_l940_94066


namespace solution_set_for_a_2_find_a_value_l940_94080

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part 1
theorem solution_set_for_a_2 :
  {x : ℝ | |x - 2| ≥ 4 - |x - 4|} = {x : ℝ | x ≥ 5 ∨ x ≤ 1} :=
sorry

-- Part 2
theorem find_a_value (a : ℝ) (h : a > 1) :
  ({x : ℝ | |f a (2*x + a) - 2*(f a x)| ≤ 2} = {x : ℝ | 1 ≤ x ∧ x ≤ 2}) →
  a = 3 :=
sorry

end solution_set_for_a_2_find_a_value_l940_94080


namespace second_grade_survey_count_l940_94030

/-- Calculates the number of students to be surveyed from the second grade in a stratified sampling method. -/
theorem second_grade_survey_count
  (total_students : ℕ)
  (grade_ratio : Fin 3 → ℕ)
  (total_surveyed : ℕ)
  (h1 : total_students = 1500)
  (h2 : grade_ratio 0 = 4 ∧ grade_ratio 1 = 5 ∧ grade_ratio 2 = 6)
  (h3 : total_surveyed = 150) :
  (total_surveyed * grade_ratio 1) / (grade_ratio 0 + grade_ratio 1 + grade_ratio 2) = 50 :=
by sorry

end second_grade_survey_count_l940_94030


namespace walking_time_proof_l940_94019

/-- Proves that walking 1.5 km at 5 km/h takes 18 minutes -/
theorem walking_time_proof (speed : ℝ) (distance : ℝ) (time_minutes : ℝ) : 
  speed = 5 → distance = 1.5 → time_minutes = (distance / speed) * 60 → time_minutes = 18 := by
  sorry

end walking_time_proof_l940_94019


namespace dance_attendance_l940_94075

theorem dance_attendance (girls : ℕ) (boys : ℕ) : 
  boys = 2 * girls ∧ 
  boys = (girls - 1) + 8 → 
  boys = 14 := by
sorry

end dance_attendance_l940_94075


namespace regular_nonagon_diagonal_relation_l940_94054

/-- Regular nonagon -/
structure RegularNonagon where
  /-- Length of a side -/
  a : ℝ
  /-- Length of the shortest diagonal -/
  b : ℝ
  /-- Length of the longest diagonal -/
  d : ℝ
  /-- a is positive -/
  a_pos : a > 0

/-- Theorem: In a regular nonagon, d^2 = a^2 + ab + b^2 -/
theorem regular_nonagon_diagonal_relation (N : RegularNonagon) : N.d^2 = N.a^2 + N.a*N.b + N.b^2 := by
  sorry

end regular_nonagon_diagonal_relation_l940_94054


namespace intersection_of_A_and_complement_of_B_l940_94085

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 3}

-- Define set B
def B : Set ℝ := {y : ℝ | y ≥ 1/2}

-- State the theorem
theorem intersection_of_A_and_complement_of_B :
  A ∩ (U \ B) = {x : ℝ | -2 ≤ x ∧ x < 1/2} := by sorry

end intersection_of_A_and_complement_of_B_l940_94085


namespace tim_water_consumption_l940_94097

/-- The number of ounces in a quart -/
def ounces_per_quart : ℕ := 32

/-- The number of quarts in each bottle Tim drinks -/
def quarts_per_bottle : ℚ := 3/2

/-- The number of bottles Tim drinks per day -/
def bottles_per_day : ℕ := 2

/-- The additional ounces Tim drinks per day -/
def additional_ounces_per_day : ℕ := 20

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The total amount of water Tim drinks in a week, in ounces -/
def water_per_week : ℕ := 812

theorem tim_water_consumption :
  (bottles_per_day * (quarts_per_bottle * ounces_per_quart).floor + additional_ounces_per_day) * days_per_week = water_per_week :=
sorry

end tim_water_consumption_l940_94097


namespace power_zero_eq_one_l940_94091

theorem power_zero_eq_one (x : ℝ) (h : x ≠ 2023) : (2023 - x)^0 = 1 := by
  sorry

end power_zero_eq_one_l940_94091


namespace brick_height_l940_94058

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * l * w + 2 * l * h + 2 * w * h

/-- Theorem: The height of a rectangular prism with given dimensions -/
theorem brick_height (l w sa : ℝ) (hl : l = 8) (hw : w = 4) (hsa : sa = 112) :
  ∃ h : ℝ, surface_area l w h = sa ∧ h = 2 := by
sorry

end brick_height_l940_94058


namespace smallest_n_terminating_with_3_l940_94087

def is_terminating_decimal (n : ℕ+) : Prop :=
  ∃ (a b : ℕ), n = 2^a * 5^b

def contains_digit_3 (n : ℕ) : Prop :=
  ∃ (d : ℕ), d ∈ n.digits 10 ∧ d = 3

theorem smallest_n_terminating_with_3 :
  ∀ n : ℕ+, n < 32 →
    ¬(is_terminating_decimal n ∧ contains_digit_3 n) ∧
    (is_terminating_decimal 32 ∧ contains_digit_3 32) :=
sorry

end smallest_n_terminating_with_3_l940_94087


namespace exponential_property_l940_94041

theorem exponential_property (a : ℝ) :
  (∀ x > 0, a^x > 1) → a > 1 := by sorry

end exponential_property_l940_94041


namespace largest_prime_divisor_check_l940_94012

theorem largest_prime_divisor_check (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) :
  Nat.Prime n → ∀ p, Nat.Prime p ∧ p ≤ 31 → ¬(p ∣ n) := by
  sorry

end largest_prime_divisor_check_l940_94012


namespace arithmetic_sequence_sum_l940_94017

/-- An arithmetic sequence with 20 terms -/
structure ArithmeticSequence :=
  (a : ℚ)  -- First term
  (d : ℚ)  -- Common difference

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n / 2 * (2 * seq.a + (n - 1) * seq.d)

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) : 
  sum_n seq 3 = 15 ∧ 
  sum_n seq 3 - 3 * seq.a - 51 * seq.d = 12 → 
  sum_n seq 20 = 90 := by
sorry

end arithmetic_sequence_sum_l940_94017


namespace angle_inequality_equivalence_l940_94074

theorem angle_inequality_equivalence (θ : Real) : 
  (0 < θ ∧ θ < Real.pi / 2) ↔ 
  (∀ x : Real, 0 ≤ x ∧ x ≤ 1 → 
    x^2 * Real.cos θ - x * (1 - x) + 2 * (1 - x)^2 * Real.sin θ > 0) :=
by sorry

end angle_inequality_equivalence_l940_94074


namespace box_volume_l940_94072

/-- The volume of a rectangular box formed from a cardboard sheet -/
theorem box_volume (initial_length initial_width corner_side : ℝ) 
  (h1 : initial_length = 13)
  (h2 : initial_width = 9)
  (h3 : corner_side = 2) : 
  (initial_length - 2 * corner_side) * (initial_width - 2 * corner_side) * corner_side = 90 := by
  sorry

end box_volume_l940_94072


namespace white_washing_cost_l940_94095

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the area of two opposite walls of a room -/
def wall_area (d : Dimensions) : ℝ := 2 * d.length * d.height

/-- Calculates the total area of four walls of a room -/
def total_wall_area (d : Dimensions) : ℝ := wall_area d + wall_area { d with length := d.width }

/-- Calculates the area of a rectangular object -/
def area (d : Dimensions) : ℝ := d.length * d.width

theorem white_washing_cost 
  (room : Dimensions) 
  (door : Dimensions)
  (window : Dimensions)
  (num_windows : ℕ)
  (cost_per_sqft : ℝ)
  (h_room : room = { length := 25, width := 15, height := 12 })
  (h_door : door = { length := 6, width := 3, height := 0 })
  (h_window : window = { length := 4, width := 3, height := 0 })
  (h_num_windows : num_windows = 3)
  (h_cost : cost_per_sqft = 8) :
  (total_wall_area room - (area door + num_windows * area window)) * cost_per_sqft = 7248 := by
  sorry

end white_washing_cost_l940_94095


namespace average_marks_of_all_students_l940_94081

theorem average_marks_of_all_students
  (batch1_size : ℕ) (batch2_size : ℕ) (batch3_size : ℕ)
  (batch1_avg : ℝ) (batch2_avg : ℝ) (batch3_avg : ℝ)
  (h1 : batch1_size = 40)
  (h2 : batch2_size = 50)
  (h3 : batch3_size = 60)
  (h4 : batch1_avg = 45)
  (h5 : batch2_avg = 55)
  (h6 : batch3_avg = 65) :
  let total_students := batch1_size + batch2_size + batch3_size
  let total_marks := batch1_size * batch1_avg + batch2_size * batch2_avg + batch3_size * batch3_avg
  total_marks / total_students = 56.33 := by
sorry

end average_marks_of_all_students_l940_94081


namespace total_bad_vegetables_l940_94050

/-- Calculate the total number of bad vegetables picked by Carol and her mom -/
theorem total_bad_vegetables (carol_carrots carol_cucumbers carol_tomatoes : ℕ)
  (mom_carrots mom_cucumbers mom_tomatoes : ℕ)
  (carol_good_carrot_percent carol_good_cucumber_percent carol_good_tomato_percent : ℚ)
  (mom_good_carrot_percent mom_good_cucumber_percent mom_good_tomato_percent : ℚ)
  (h1 : carol_carrots = 29)
  (h2 : carol_cucumbers = 15)
  (h3 : carol_tomatoes = 10)
  (h4 : mom_carrots = 16)
  (h5 : mom_cucumbers = 12)
  (h6 : mom_tomatoes = 14)
  (h7 : carol_good_carrot_percent = 80/100)
  (h8 : carol_good_cucumber_percent = 95/100)
  (h9 : carol_good_tomato_percent = 90/100)
  (h10 : mom_good_carrot_percent = 85/100)
  (h11 : mom_good_cucumber_percent = 70/100)
  (h12 : mom_good_tomato_percent = 75/100) :
  (carol_carrots - ⌊carol_carrots * carol_good_carrot_percent⌋) +
  (carol_cucumbers - ⌊carol_cucumbers * carol_good_cucumber_percent⌋) +
  (carol_tomatoes - ⌊carol_tomatoes * carol_good_tomato_percent⌋) +
  (mom_carrots - ⌊mom_carrots * mom_good_carrot_percent⌋) +
  (mom_cucumbers - ⌊mom_cucumbers * mom_good_cucumber_percent⌋) +
  (mom_tomatoes - ⌊mom_tomatoes * mom_good_tomato_percent⌋) = 19 := by
  sorry


end total_bad_vegetables_l940_94050


namespace nine_point_chords_l940_94070

/-- The number of chords that can be drawn from n points on a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem: The number of chords from 9 points on a circle is 36 -/
theorem nine_point_chords : num_chords 9 = 36 := by
  sorry

end nine_point_chords_l940_94070


namespace matrix_power_2023_l940_94082

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end matrix_power_2023_l940_94082


namespace roy_julia_multiple_l940_94048

-- Define variables for current ages
variable (R J K : ℕ)

-- Define the multiple
variable (M : ℕ)

-- Roy is 6 years older than Julia
def roy_julia_diff : Prop := R = J + 6

-- Roy is half of 6 years older than Kelly
def roy_kelly_diff : Prop := R = K + 3

-- In 4 years, Roy will be some multiple of Julia's age
def future_age_multiple : Prop := R + 4 = M * (J + 4)

-- In 4 years, Roy's age multiplied by Kelly's age would be 108
def future_age_product : Prop := (R + 4) * (K + 4) = 108

theorem roy_julia_multiple
  (h1 : roy_julia_diff R J)
  (h2 : roy_kelly_diff R K)
  (h3 : future_age_multiple R J M)
  (h4 : future_age_product R K) :
  M = 2 := by sorry

end roy_julia_multiple_l940_94048


namespace frog_climb_time_l940_94043

/-- Represents the frog's climbing problem -/
structure FrogClimb where
  well_depth : ℕ
  climb_distance : ℕ
  slip_distance : ℕ
  climb_time : ℚ
  slip_time : ℚ
  intermediate_time : ℕ
  intermediate_distance : ℕ

/-- Theorem stating the time taken for the frog to climb out of the well -/
theorem frog_climb_time (f : FrogClimb)
  (h1 : f.well_depth = 12)
  (h2 : f.climb_distance = 3)
  (h3 : f.slip_distance = 1)
  (h4 : f.slip_time = f.climb_time / 3)
  (h5 : f.intermediate_time = 17)
  (h6 : f.intermediate_distance = f.well_depth - 3)
  (h7 : f.climb_time = 1) :
  ∃ (total_time : ℕ), total_time = 22 ∧ 
  (∃ (cycles : ℕ), 
    cycles * (f.climb_distance - f.slip_distance) + 
    (total_time - cycles * (f.climb_time + f.slip_time)) * f.climb_distance / f.climb_time = f.well_depth) :=
sorry

end frog_climb_time_l940_94043


namespace blue_glass_ball_probability_l940_94021

/-- The probability of drawing a blue glass ball given that a glass ball is drawn -/
theorem blue_glass_ball_probability :
  let total_balls : ℕ := 5 + 11
  let red_glass_balls : ℕ := 2
  let blue_glass_balls : ℕ := 4
  let total_glass_balls : ℕ := red_glass_balls + blue_glass_balls
  (blue_glass_balls : ℚ) / total_glass_balls = 2 / 3 :=
by sorry

end blue_glass_ball_probability_l940_94021


namespace isosceles_triangle_vertex_angle_l940_94044

/-- An isosceles triangle with two angles in the ratio 1:4 has a vertex angle of either 20 or 120 degrees. -/
theorem isosceles_triangle_vertex_angle (a b c : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Angles are positive
  a + b + c = 180 →  -- Sum of angles in a triangle
  a = b →  -- Isosceles triangle condition
  (c = a ∧ b = 4 * a) ∨ (a = 4 * c ∧ b = 4 * c) →  -- Ratio condition
  c = 20 ∨ c = 120 := by
sorry


end isosceles_triangle_vertex_angle_l940_94044


namespace tim_watched_24_hours_l940_94035

/-- Calculates the total hours of TV watched given the number of episodes and duration per episode for two shows. -/
def total_hours_watched (short_episodes : ℕ) (short_duration : ℚ) (long_episodes : ℕ) (long_duration : ℚ) : ℚ :=
  short_episodes * short_duration + long_episodes * long_duration

/-- Proves that Tim watched 24 hours of TV given the specified conditions. -/
theorem tim_watched_24_hours :
  let short_episodes : ℕ := 24
  let short_duration : ℚ := 1/2
  let long_episodes : ℕ := 12
  let long_duration : ℚ := 1
  total_hours_watched short_episodes short_duration long_episodes long_duration = 24 := by
  sorry

#eval total_hours_watched 24 (1/2) 12 1

end tim_watched_24_hours_l940_94035


namespace harmonic_quadrilateral_properties_l940_94063

-- Define a structure for a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a quadrilateral as a collection of four points
structure Quadrilateral :=
  (A B C D : Point)

-- Define the property of a harmonic quadrilateral
def is_harmonic (q : Quadrilateral) : Prop :=
  ∃ (AB CD AC BD AD BC : ℝ),
    AB * CD = AC * BD ∧ AB * CD = AD * BC

-- Define the concyclic property for four points
def are_concyclic (A B C D : Point) : Prop :=
  ∃ (center : Point) (radius : ℝ),
    (A.x - center.x)^2 + (A.y - center.y)^2 = radius^2 ∧
    (B.x - center.x)^2 + (B.y - center.y)^2 = radius^2 ∧
    (C.x - center.x)^2 + (C.y - center.y)^2 = radius^2 ∧
    (D.x - center.x)^2 + (D.y - center.y)^2 = radius^2

-- State the theorem
theorem harmonic_quadrilateral_properties
  (ABCD : Quadrilateral)
  (A1 B1 C1 D1 : Point)
  (h1 : is_harmonic ABCD)
  (h2 : is_harmonic ⟨A1, ABCD.B, ABCD.C, ABCD.D⟩)
  (h3 : is_harmonic ⟨ABCD.A, B1, ABCD.C, ABCD.D⟩)
  (h4 : is_harmonic ⟨ABCD.A, ABCD.B, C1, ABCD.D⟩)
  (h5 : is_harmonic ⟨ABCD.A, ABCD.B, ABCD.C, D1⟩) :
  are_concyclic ABCD.A ABCD.B C1 D1 ∧ is_harmonic ⟨A1, B1, C1, D1⟩ :=
sorry

end harmonic_quadrilateral_properties_l940_94063


namespace polynomial_identity_solution_l940_94036

theorem polynomial_identity_solution :
  ∀ (a b c : ℝ),
    (∀ x : ℝ, x^3 - a*x^2 + b*x - c = (x-a)*(x-b)*(x-c))
    ↔ 
    (a = -1 ∧ b = -1 ∧ c = 1) :=
by sorry

end polynomial_identity_solution_l940_94036


namespace vector_dot_product_l940_94065

theorem vector_dot_product (α : ℝ) (b : Fin 2 → ℝ) :
  let a : Fin 2 → ℝ := ![Real.cos α, Real.sin α]
  (a • b = -1) →
  (a • (2 • a - b) = 3) := by
  sorry

end vector_dot_product_l940_94065


namespace parabola_equation_l940_94089

/-- A parabola with vertex at the origin and focus on the x-axis -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop
  vertex_origin : equation 0 0
  focus_x_axis : ∃ (f : ℝ), equation f 0 ∧ f ≠ 0

/-- The line y = 2x + 1 -/
def line (x y : ℝ) : Prop := y = 2 * x + 1

/-- The chord created by intersecting the parabola with the line -/
def chord (p : Parabola) (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  p.equation x₁ y₁ ∧ p.equation x₂ y₂ ∧ line x₁ y₁ ∧ line x₂ y₂ ∧ (x₁ ≠ x₂ ∨ y₁ ≠ y₂)

theorem parabola_equation (p : Parabola) :
  (∃ (x₁ y₁ x₂ y₂ : ℝ), chord p x₁ y₁ x₂ y₂ ∧ (x₁ - x₂)^2 + (y₁ - y₂)^2 = 15) →
  p.equation = λ x y => y^2 = 12 * x :=
sorry

end parabola_equation_l940_94089


namespace inequality_proof_l940_94018

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a + b + c = 1/a + 1/b + 1/c) : a + b + c ≥ 3/(a*b*c) := by
  sorry

end inequality_proof_l940_94018


namespace sandy_comic_books_l940_94022

theorem sandy_comic_books (x : ℕ) : 
  (x / 2 : ℚ) - 3 + 6 = 13 → x = 20 := by
  sorry

end sandy_comic_books_l940_94022


namespace girls_in_circle_l940_94038

/-- The number of girls in a circle of children, given specific conditions. -/
def number_of_girls (total : ℕ) (holding_boys_hand : ℕ) (holding_girls_hand : ℕ) : ℕ :=
  (2 * holding_girls_hand + holding_boys_hand - total) / 2

/-- Theorem stating that the number of girls in the circle is 24. -/
theorem girls_in_circle : number_of_girls 40 22 30 = 24 := by
  sorry

#eval number_of_girls 40 22 30

end girls_in_circle_l940_94038


namespace root_sum_reciprocal_l940_94083

theorem root_sum_reciprocal (α β γ : ℝ) : 
  (60 * α^3 - 80 * α^2 + 24 * α - 2 = 0) →
  (60 * β^3 - 80 * β^2 + 24 * β - 2 = 0) →
  (60 * γ^3 - 80 * γ^2 + 24 * γ - 2 = 0) →
  (α ≠ β) → (β ≠ γ) → (α ≠ γ) →
  (0 < α) → (α < 1) →
  (0 < β) → (β < 1) →
  (0 < γ) → (γ < 1) →
  (1 / (1 - α) + 1 / (1 - β) + 1 / (1 - γ) = 22) :=
by sorry

end root_sum_reciprocal_l940_94083


namespace middle_is_four_l940_94027

/-- Represents a trio of integers -/
structure Trio :=
  (left : ℕ)
  (middle : ℕ)
  (right : ℕ)

/-- Checks if a trio satisfies the given conditions -/
def validTrio (t : Trio) : Prop :=
  t.left < t.middle ∧ t.middle < t.right ∧ t.left + t.middle + t.right = 15

/-- Casey cannot determine the other two numbers -/
def caseyUncertain (t : Trio) : Prop :=
  ∃ t' : Trio, t' ≠ t ∧ validTrio t' ∧ t'.left = t.left

/-- Tracy cannot determine the other two numbers -/
def tracyUncertain (t : Trio) : Prop :=
  ∃ t' : Trio, t' ≠ t ∧ validTrio t' ∧ t'.right = t.right

/-- Stacy cannot determine the other two numbers -/
def stacyUncertain (t : Trio) : Prop :=
  ∃ t' : Trio, t' ≠ t ∧ validTrio t' ∧ t'.middle = t.middle

/-- The main theorem stating that the middle number must be 4 -/
theorem middle_is_four :
  ∀ t : Trio, validTrio t →
    caseyUncertain t → tracyUncertain t → stacyUncertain t →
    t.middle = 4 :=
by sorry

end middle_is_four_l940_94027


namespace central_cell_is_two_l940_94071

/-- Represents a 3x3 grid with numbers from 0 to 8 -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Checks if two cells are neighbors -/
def is_neighbor (a b : Fin 3 × Fin 3) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

/-- Checks if the grid satisfies the consecutive number condition -/
def consecutive_condition (g : Grid) : Prop :=
  ∀ i j k l : Fin 3, is_neighbor (i, j) (k, l) →
    (g i j = g k l + 1 ∨ g i j + 1 = g k l)

/-- Calculates the sum of corner cells -/
def corner_sum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- Theorem: In a valid 3x3 grid where the sum of corner cells is 18,
    the number in the central cell must be 2 -/
theorem central_cell_is_two (g : Grid) 
  (h1 : consecutive_condition g) 
  (h2 : corner_sum g = 18) : 
  g 1 1 = 2 := by
  sorry

end central_cell_is_two_l940_94071


namespace luke_laundry_loads_l940_94067

def total_clothing : ℕ := 47
def first_load : ℕ := 17
def pieces_per_load : ℕ := 6

theorem luke_laundry_loads : 
  (total_clothing - first_load) / pieces_per_load = 5 :=
by sorry

end luke_laundry_loads_l940_94067


namespace perpendicular_line_equation_l940_94009

/-- The general form equation of a line perpendicular to 2x+y-5=0 and passing through (1,2) is x-2y+3=0 -/
theorem perpendicular_line_equation :
  ∀ (x y : ℝ),
  (∃ (m b : ℝ), y = m * x + b ∧ m * 2 = -1) →  -- perpendicular line condition
  (1 : ℝ) - 2 * (2 : ℝ) + 3 = 0 →              -- point (1,2) satisfies the equation
  x - 2 * y + 3 = 0                            -- the equation we want to prove
  := by sorry

end perpendicular_line_equation_l940_94009


namespace square_difference_existence_l940_94014

theorem square_difference_existence (n : ℤ) : 
  (∃ a b : ℤ, n + a^2 = b^2) ↔ n % 4 ≠ 2 := by sorry

end square_difference_existence_l940_94014


namespace ordering_of_powers_l940_94045

theorem ordering_of_powers : 6^8 < 3^15 ∧ 3^15 < 8^10 := by
  sorry

end ordering_of_powers_l940_94045


namespace alice_speed_l940_94028

theorem alice_speed (total_distance : ℝ) (abel_speed : ℝ) (time_difference : ℝ) (alice_delay : ℝ) :
  total_distance = 1000 →
  abel_speed = 50 →
  time_difference = 6 →
  alice_delay = 1 →
  (total_distance / abel_speed + alice_delay) - (total_distance / abel_speed) = time_difference →
  total_distance / ((total_distance / abel_speed) + time_difference) = 200 / 3 := by
sorry

end alice_speed_l940_94028


namespace floor_sqrt_10_l940_94069

theorem floor_sqrt_10 : ⌊Real.sqrt 10⌋ = 3 := by
  sorry

end floor_sqrt_10_l940_94069


namespace function_property_l940_94010

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x + f (1 - x) = 10)
  (h2 : ∀ x : ℝ, f (1 + x) = 3 + f x) :
  ∀ x : ℝ, f x + f (-x) = 7 := by
sorry

end function_property_l940_94010


namespace betty_age_l940_94001

/-- Given the relationships between Albert's, Mary's, and Betty's ages, prove Betty's age. -/
theorem betty_age (albert mary betty : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 8) :
  betty = 4 := by
  sorry

end betty_age_l940_94001


namespace parallelogram_angle_c_l940_94093

-- Define a parallelogram structure
structure Parallelogram :=
  (A B C D : ℝ × ℝ)

-- Define angle measure in degrees
def angle_measure (p : Parallelogram) (vertex : Char) : ℝ := sorry

-- State the theorem
theorem parallelogram_angle_c (p : Parallelogram) :
  angle_measure p 'A' + 40 = angle_measure p 'B' →
  angle_measure p 'C' = 70 := by sorry

end parallelogram_angle_c_l940_94093


namespace probability_theorem_l940_94078

/- Define the number of white and black balls -/
def white_balls : ℕ := 2
def black_balls : ℕ := 3
def total_balls : ℕ := white_balls + black_balls

/- Define the probability of drawing a white ball and a black ball -/
def prob_white : ℚ := white_balls / total_balls
def prob_black : ℚ := black_balls / total_balls

/- Part I: Sampling with replacement -/
def prob_different_colors : ℚ := prob_white * prob_black * 2

/- Part II: Sampling without replacement -/
def prob_zero_white : ℚ := (black_balls / total_balls) * ((black_balls - 1) / (total_balls - 1))
def prob_one_white : ℚ := (black_balls / total_balls) * (white_balls / (total_balls - 1)) + 
                          (white_balls / total_balls) * (black_balls / (total_balls - 1))
def prob_two_white : ℚ := (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))

def expectation : ℚ := 0 * prob_zero_white + 1 * prob_one_white + 2 * prob_two_white
def variance : ℚ := (0 - expectation)^2 * prob_zero_white + 
                    (1 - expectation)^2 * prob_one_white + 
                    (2 - expectation)^2 * prob_two_white

theorem probability_theorem :
  prob_different_colors = 12/25 ∧
  prob_zero_white = 3/10 ∧
  prob_one_white = 3/5 ∧
  prob_two_white = 1/10 ∧
  expectation = 4/5 ∧
  variance = 9/25 := by sorry

end probability_theorem_l940_94078


namespace ellipse_and_line_properties_l940_94016

/-- Ellipse C with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Given conditions for the problem -/
structure ProblemConditions (C : Ellipse) where
  eccentricity : ℝ
  focusDistance : ℝ
  h1 : eccentricity = 1/2
  h2 : focusDistance = 2 * Real.sqrt 2

/-- The equation of line l -/
def line_equation (x y : ℝ) : Prop :=
  Real.sqrt 3 * x + y - 2 * Real.sqrt 3 = 0

/-- Main theorem statement -/
theorem ellipse_and_line_properties
  (C : Ellipse)
  (cond : ProblemConditions C)
  (T_y_coord : ℝ)
  (h_T_y : T_y_coord = 6 * Real.sqrt 3) :
  (∀ x y, x^2 / 16 + y^2 / 12 = 1 ↔ x^2 / C.a^2 + y^2 / C.b^2 = 1) ∧
  (∀ x y, line_equation x y) :=
sorry

end ellipse_and_line_properties_l940_94016


namespace multiply_twelve_problem_l940_94073

theorem multiply_twelve_problem (x : ℚ) : 
  (12 * x * 2 = 7899665 - 7899593) → x = 3 := by
sorry

end multiply_twelve_problem_l940_94073


namespace square_difference_65_35_l940_94053

theorem square_difference_65_35 : 65^2 - 35^2 = 3000 := by
  sorry

end square_difference_65_35_l940_94053


namespace modulo_five_power_difference_l940_94046

theorem modulo_five_power_difference : (27^1235 - 19^1235) % 5 = 2 := by sorry

end modulo_five_power_difference_l940_94046


namespace speed_ratio_l940_94061

/-- The speed of Person A -/
def speed_A : ℝ := sorry

/-- The speed of Person B -/
def speed_B : ℝ := sorry

/-- The distance covered by Person A in a given time -/
def distance_A : ℝ := sorry

/-- The distance covered by Person B in the same time -/
def distance_B : ℝ := sorry

/-- The time taken for both persons to cover their respective distances -/
def time : ℝ := sorry

theorem speed_ratio :
  (speed_A / speed_B = 3 / 2) ∧
  (distance_A = 3) ∧
  (distance_B = 2) ∧
  (speed_A = distance_A / time) ∧
  (speed_B = distance_B / time) :=
by sorry

end speed_ratio_l940_94061


namespace simplify_and_sum_exponents_l940_94037

variables (a b c : ℝ)

theorem simplify_and_sum_exponents :
  ∃ (x y z : ℕ) (w : ℝ),
    (40 * a^7 * b^9 * c^14)^(1/3) = 2 * a^x * b^y * c^z * w^(1/3) ∧
    w = 5 * a * c^2 ∧
    x + y + z = 9 := by sorry

end simplify_and_sum_exponents_l940_94037


namespace vector_subtraction_l940_94099

/-- Given two vectors AB and AC in a plane, prove that vector BC is their difference. -/
theorem vector_subtraction (AB AC : ℝ × ℝ) (h1 : AB = (3, 4)) (h2 : AC = (1, 3)) :
  AC - AB = (-2, -1) := by
  sorry

end vector_subtraction_l940_94099


namespace probability_two_red_balls_l940_94000

def total_balls : ℕ := 10
def red_balls : ℕ := 4
def white_balls : ℕ := 6
def drawn_balls : ℕ := 5
def red_balls_drawn : ℕ := 2

theorem probability_two_red_balls :
  (Nat.choose red_balls red_balls_drawn * Nat.choose white_balls (drawn_balls - red_balls_drawn)) /
  Nat.choose total_balls drawn_balls = 10 / 21 := by
  sorry

end probability_two_red_balls_l940_94000


namespace nell_card_count_l940_94079

/-- The number of cards Nell has after receiving cards from Jeff -/
def total_cards (initial_cards given_cards : ℝ) : ℝ :=
  initial_cards + given_cards

/-- Theorem stating that Nell's total cards equal the sum of her initial cards and those given by Jeff -/
theorem nell_card_count (initial_cards given_cards : ℝ) :
  total_cards initial_cards given_cards = initial_cards + given_cards :=
by sorry

end nell_card_count_l940_94079


namespace point_on_line_m_value_l940_94020

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) = (p3.x - p1.x) * (p2.y - p1.y)

theorem point_on_line_m_value :
  ∀ m : ℝ,
  let P : Point := ⟨3, m⟩
  let M : Point := ⟨2, -1⟩
  let N : Point := ⟨-3, 4⟩
  collinear P M N → m = -2 := by
  sorry

end point_on_line_m_value_l940_94020


namespace equipment_production_l940_94011

theorem equipment_production (total : ℕ) (sample_size : ℕ) (sample_a : ℕ) 
  (h_total : total = 4800)
  (h_sample_size : sample_size = 80)
  (h_sample_a : sample_a = 50) :
  total - (total * sample_a / sample_size) = 1800 := by
sorry

end equipment_production_l940_94011


namespace liters_to_pints_conversion_l940_94007

/-- Given that 0.33 liters is approximately 0.7 pints, prove that one liter is approximately 2.1 pints. -/
theorem liters_to_pints_conversion (ε : ℝ) (h_ε : ε > 0) :
  ∃ (δ : ℝ), δ > 0 ∧ 
  ∀ (x y : ℝ), 
    (abs (x - 0.33) < δ ∧ abs (y - 0.7) < δ) → 
    abs ((1 / x * y) - 2.1) < ε :=
sorry

end liters_to_pints_conversion_l940_94007


namespace kabadi_player_count_l940_94003

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := 15

/-- The total number of players -/
def total_players : ℕ := 50

/-- The number of players who play kho kho only -/
def kho_kho_only : ℕ := 40

/-- The number of players who play both games -/
def both_games : ℕ := 5

theorem kabadi_player_count :
  kabadi_players = total_players - kho_kho_only + both_games :=
by sorry

end kabadi_player_count_l940_94003


namespace vector_equation_solution_l940_94034

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)
variable (x y : ℝ)

theorem vector_equation_solution (h_not_collinear : ¬ ∃ (k : ℝ), b = k • a) 
  (h_eq : (2*x - y) • a + 4 • b = 5 • a + (x - 2*y) • b) : 
  x + y = 1 := by sorry

end vector_equation_solution_l940_94034


namespace bridge_renovation_problem_l940_94042

theorem bridge_renovation_problem (bridge_length : ℝ) (efficiency_increase : ℝ) (days_ahead : ℝ) 
  (h1 : bridge_length = 36)
  (h2 : efficiency_increase = 0.5)
  (h3 : days_ahead = 2) :
  ∃ x : ℝ, x = 6 ∧ 
    bridge_length / x = bridge_length / ((1 + efficiency_increase) * x) + days_ahead :=
by sorry

end bridge_renovation_problem_l940_94042


namespace gcd_459_357_l940_94086

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l940_94086


namespace fourth_term_binomial_expansion_l940_94033

theorem fourth_term_binomial_expansion 
  (a x : ℝ) (ha : a ≠ 0) (hx : x > 0) :
  let binomial := (2*a/Real.sqrt x - Real.sqrt x/(2*a^2))^8
  let fourth_term := (Nat.choose 8 3) * (2*a/Real.sqrt x)^5 * (-Real.sqrt x/(2*a^2))^3
  fourth_term = -4/(a*x) :=
by sorry

end fourth_term_binomial_expansion_l940_94033


namespace angle_GDA_measure_l940_94057

-- Define the points
variable (A B C D E F G : Point)

-- Define the shapes
def is_regular_pentagon (C D E : Point) : Prop := sorry

def is_square (A B C D : Point) : Prop := sorry

-- Define the angle measure
def angle_measure (G D A : Point) : ℝ := sorry

-- State the theorem
theorem angle_GDA_measure 
  (h1 : is_regular_pentagon C D E)
  (h2 : is_square A B C D)
  (h3 : is_square D E F G) :
  angle_measure G D A = 72 := by sorry

end angle_GDA_measure_l940_94057


namespace goods_train_length_l940_94013

/-- The length of a goods train passing a man in an opposite-moving train --/
theorem goods_train_length (man_speed goods_speed : ℝ) (pass_time : ℝ) : 
  man_speed = 64 →
  goods_speed = 20 →
  pass_time = 18 →
  ∃ (length : ℝ), abs (length - (man_speed + goods_speed) * 1000 / 3600 * pass_time) < 1 :=
by sorry

end goods_train_length_l940_94013


namespace original_cost_of_mixed_nuts_l940_94098

/-- Calculates the original cost of a bag of mixed nuts -/
theorem original_cost_of_mixed_nuts
  (bag_size : ℕ)
  (serving_size : ℕ)
  (cost_per_serving_after_coupon : ℚ)
  (coupon_value : ℚ)
  (h1 : bag_size = 40)
  (h2 : serving_size = 1)
  (h3 : cost_per_serving_after_coupon = 1/2)
  (h4 : coupon_value = 5) :
  bag_size * cost_per_serving_after_coupon + coupon_value = 25 :=
sorry

end original_cost_of_mixed_nuts_l940_94098


namespace collinear_points_k_value_l940_94051

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem collinear_points_k_value :
  ∀ k : ℝ,
  let A : Point := ⟨3, 1⟩
  let B : Point := ⟨-2, k⟩
  let C : Point := ⟨8, 11⟩
  collinear A B C → k = -9 := by
  sorry

end collinear_points_k_value_l940_94051


namespace homework_time_reduction_l940_94068

theorem homework_time_reduction (x : ℝ) : 
  (∀ t₀ t₂ : ℝ, t₀ > 0 ∧ t₂ > 0 ∧ t₀ > t₂ →
    (∃ t₁ : ℝ, t₁ = t₀ * (1 - x) ∧ t₂ = t₁ * (1 - x)) ↔
    t₀ * (1 - x)^2 = t₂) →
  100 * (1 - x)^2 = 70 :=
by sorry

end homework_time_reduction_l940_94068


namespace det_max_value_l940_94062

open Real Matrix

theorem det_max_value (θ : ℝ) :
  let A : Matrix (Fin 3) (Fin 3) ℝ := !![1, 1, 1; 1 + tan θ, 1, 1; 1, 1, 1 + cos θ]
  ∀ φ : ℝ, det A ≤ det (!![1, 1, 1; 1 + tan φ, 1, 1; 1, 1, 1 + cos φ]) :=
by sorry

end det_max_value_l940_94062


namespace correct_statement_l940_94026

/-- Proposition p: There exists an x₀ ∈ ℝ such that x₀² + x₀ + 1 ≤ 0 -/
def p : Prop := ∃ x₀ : ℝ, x₀^2 + x₀ + 1 ≤ 0

/-- Proposition q: The function f(x) = x^(1/3) is an increasing function -/
def q : Prop := ∀ x y : ℝ, x < y → Real.rpow x (1/3) < Real.rpow y (1/3)

/-- The correct statement is (¬p) ∨ q -/
theorem correct_statement : (¬p) ∨ q := by sorry

end correct_statement_l940_94026


namespace complex_cube_root_identity_l940_94039

theorem complex_cube_root_identity (z : ℂ) (h1 : z^3 + 1 = 0) (h2 : z ≠ -1) :
  (z / (z - 1))^2018 + (1 / (z - 1))^2018 = -1 := by
  sorry

end complex_cube_root_identity_l940_94039


namespace polygon_sides_l940_94094

theorem polygon_sides (n : ℕ) (sum_angles : ℝ) : sum_angles = 1800 → (n - 2) * 180 = sum_angles → n = 12 := by
  sorry

end polygon_sides_l940_94094


namespace smallest_six_digit_multiple_of_1379_l940_94023

theorem smallest_six_digit_multiple_of_1379 : 
  ∀ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 1379 = 0 → n ≥ 100657 :=
by
  sorry

end smallest_six_digit_multiple_of_1379_l940_94023


namespace cubic_function_root_product_l940_94049

/-- A cubic function with specific properties -/
structure CubicFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  x₁ : ℝ
  x₂ : ℝ
  root_zero : d = 0
  root_x₁ : a * x₁^3 + b * x₁^2 + c * x₁ + d = 0
  root_x₂ : a * x₂^3 + b * x₂^2 + c * x₂ + d = 0
  extreme_value_1 : (3 * a * ((3 - Real.sqrt 3) / 3)^2 + 2 * b * ((3 - Real.sqrt 3) / 3) + c) = 0
  extreme_value_2 : (3 * a * ((3 + Real.sqrt 3) / 3)^2 + 2 * b * ((3 + Real.sqrt 3) / 3) + c) = 0
  a_nonzero : a ≠ 0

/-- The product of non-zero roots of the cubic function is 2 -/
theorem cubic_function_root_product (f : CubicFunction) : f.x₁ * f.x₂ = 2 := by
  sorry

end cubic_function_root_product_l940_94049


namespace geometric_sequence_property_l940_94055

/-- A sequence is geometric if the ratio of successive terms is constant. -/
def IsGeometric (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a_n where a_3 * a_5 * a_7 = (-√3)^3, prove a_2 * a_8 = 3 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geometric : IsGeometric a) 
    (h_product : a 3 * a 5 * a 7 = (-Real.sqrt 3)^3) : 
  a 2 * a 8 = 3 := by
  sorry

end geometric_sequence_property_l940_94055


namespace number_minus_division_equals_l940_94056

theorem number_minus_division_equals (x : ℝ) : x - (502 / 100.4) = 5015 → x = 5020 := by
  sorry

end number_minus_division_equals_l940_94056


namespace sin_300_degrees_l940_94064

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_degrees_l940_94064


namespace roots_equal_magnitude_implies_real_ratio_l940_94092

theorem roots_equal_magnitude_implies_real_ratio 
  (p q : ℂ) 
  (h_q_nonzero : q ≠ 0) 
  (h_roots_equal_magnitude : ∀ z₁ z₂ : ℂ, z₁^2 + p*z₁ + q^2 = 0 → z₂^2 + p*z₂ + q^2 = 0 → Complex.abs z₁ = Complex.abs z₂) :
  ∃ r : ℝ, p / q = r := by sorry

end roots_equal_magnitude_implies_real_ratio_l940_94092


namespace simplify_fraction_l940_94047

theorem simplify_fraction : (270 / 5400) * 30 = 3 / 2 := by
  sorry

end simplify_fraction_l940_94047


namespace school_travel_time_difference_l940_94084

/-- The problem of calculating the late arrival time of a boy traveling to school. -/
theorem school_travel_time_difference (distance : ℝ) (speed1 speed2 : ℝ) (early_time : ℝ) : 
  distance = 2.5 →
  speed1 = 5 →
  speed2 = 10 →
  early_time = 8 / 60 →
  (distance / speed1 - (distance / speed2 + early_time)) * 60 = 7 := by
  sorry

end school_travel_time_difference_l940_94084


namespace intersection_equals_half_open_interval_l940_94002

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | x ≥ 0}
def N : Set ℝ := {x : ℝ | x^2 < 1}

-- Define the intersection of M and N
def M_intersect_N : Set ℝ := M ∩ N

-- State the theorem
theorem intersection_equals_half_open_interval :
  M_intersect_N = Set.Icc 0 1 := by sorry

end intersection_equals_half_open_interval_l940_94002


namespace jake_debt_l940_94088

/-- The amount Jake originally owed given his payment and work details --/
def original_debt (prepaid_amount : ℕ) (hourly_rate : ℕ) (hours_worked : ℕ) : ℕ :=
  prepaid_amount + hourly_rate * hours_worked

/-- Theorem stating that Jake's original debt was $100 --/
theorem jake_debt : original_debt 40 15 4 = 100 := by
  sorry

end jake_debt_l940_94088
