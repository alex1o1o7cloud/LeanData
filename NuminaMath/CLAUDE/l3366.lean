import Mathlib

namespace NUMINAMATH_CALUDE_remainder_problem_l3366_336658

theorem remainder_problem (n : ℤ) : n % 5 = 3 → (4 * n - 5) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3366_336658


namespace NUMINAMATH_CALUDE_sale_price_markdown_l3366_336603

theorem sale_price_markdown (original_price : ℝ) (h1 : original_price > 0) : 
  let sale_price := 0.8 * original_price
  let final_price := 0.64 * original_price
  let markdown_percentage := (sale_price - final_price) / sale_price * 100
  markdown_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_sale_price_markdown_l3366_336603


namespace NUMINAMATH_CALUDE_triangle_area_not_integer_l3366_336615

theorem triangle_area_not_integer (a b c : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) 
  (htriangle : a + b > c ∧ b + c > a ∧ c + a > b) : 
  ¬ ∃ (S : ℕ), (S : ℝ)^2 * 16 = (a + b + c) * ((a + b + c) - 2*a) * ((a + b + c) - 2*b) * ((a + b + c) - 2*c) :=
sorry


end NUMINAMATH_CALUDE_triangle_area_not_integer_l3366_336615


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3366_336649

-- Problem 1
theorem problem_1 : (-12) + 13 + (-18) + 16 = -1 := by sorry

-- Problem 2
theorem problem_2 : 19.5 + (-6.9) + (-3.1) + (-9.5) = 0 := by sorry

-- Problem 3
theorem problem_3 : (6/5 : ℚ) * (-1/3 - 1/2) / (5/4 : ℚ) = -4/5 := by sorry

-- Problem 4
theorem problem_4 : 18 + 32 * (-1/2)^5 - (1/2)^4 * (-2)^5 = 19 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l3366_336649


namespace NUMINAMATH_CALUDE_consecutive_integers_median_l3366_336644

theorem consecutive_integers_median (n : ℕ) (sum : ℕ) (h1 : n = 25) (h2 : sum = 3125) :
  (sum : ℚ) / n = 125 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_median_l3366_336644


namespace NUMINAMATH_CALUDE_rectangle_area_l3366_336691

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 226) : L * B = 3060 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3366_336691


namespace NUMINAMATH_CALUDE_max_value_expression_l3366_336633

theorem max_value_expression (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) : 
  (a + 1) * (b + 1) * (c + 1) / (a * b * c + 1) ≤ 16 / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_expression_l3366_336633


namespace NUMINAMATH_CALUDE_percentage_difference_l3366_336659

theorem percentage_difference : (65 / 100 * 40) - (4 / 5 * 25) = 6 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3366_336659


namespace NUMINAMATH_CALUDE_buffer_saline_volume_l3366_336692

theorem buffer_saline_volume 
  (total_buffer : ℚ) 
  (solution_b_volume : ℚ) 
  (saline_volume : ℚ) 
  (initial_mixture_volume : ℚ) :
  total_buffer = 3/2 →
  solution_b_volume = 1/4 →
  saline_volume = 1/6 →
  initial_mixture_volume = 5/12 →
  solution_b_volume + saline_volume = initial_mixture_volume →
  (total_buffer * (saline_volume / initial_mixture_volume) : ℚ) = 3/5 :=
by sorry

end NUMINAMATH_CALUDE_buffer_saline_volume_l3366_336692


namespace NUMINAMATH_CALUDE_root_ratio_to_power_l3366_336681

theorem root_ratio_to_power (x : ℝ) (h : x > 0) :
  (x^(1/3)) / (x^(1/5)) = x^(2/15) :=
by sorry

end NUMINAMATH_CALUDE_root_ratio_to_power_l3366_336681


namespace NUMINAMATH_CALUDE_boat_speed_is_18_l3366_336602

/-- The speed of the boat in still water -/
def boat_speed : ℝ := 18

/-- The speed of the stream -/
def stream_speed : ℝ := 6

/-- Theorem stating that the boat speed in still water is 18 kmph -/
theorem boat_speed_is_18 :
  (∀ t : ℝ, t > 0 → 1 / (boat_speed - stream_speed) = 2 / (boat_speed + stream_speed)) →
  boat_speed = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_boat_speed_is_18_l3366_336602


namespace NUMINAMATH_CALUDE_total_votes_l3366_336635

theorem total_votes (jerry_votes : ℕ) (vote_difference : ℕ) : 
  jerry_votes = 108375 →
  vote_difference = 20196 →
  jerry_votes + (jerry_votes - vote_difference) = 196554 :=
by
  sorry

end NUMINAMATH_CALUDE_total_votes_l3366_336635


namespace NUMINAMATH_CALUDE_tenth_number_value_l3366_336629

def known_numbers : List ℕ := [744, 745, 747, 748, 749, 752, 752, 753, 755]

theorem tenth_number_value (x : ℕ) :
  (known_numbers.sum + x) / 10 = 750 →
  x = 1555 := by
  sorry

end NUMINAMATH_CALUDE_tenth_number_value_l3366_336629


namespace NUMINAMATH_CALUDE_coefficient_x3y3_l3366_336647

/-- The coefficient of x³y³ in the expansion of (x+2y)(x+y)⁵ is 30 -/
theorem coefficient_x3y3 : Int :=
  30

#check coefficient_x3y3

end NUMINAMATH_CALUDE_coefficient_x3y3_l3366_336647


namespace NUMINAMATH_CALUDE_park_area_l3366_336645

/-- Represents a rectangular park with sides in ratio 3:2 -/
structure Park where
  x : ℝ
  length : ℝ := 3 * x
  width : ℝ := 2 * x

/-- The perimeter of the park -/
def Park.perimeter (p : Park) : ℝ := 2 * (p.length + p.width)

/-- The area of the park -/
def Park.area (p : Park) : ℝ := p.length * p.width

/-- The cost of fencing per meter in rupees -/
def fencing_cost_per_meter : ℝ := 0.50

/-- The total cost of fencing the park in rupees -/
def total_fencing_cost : ℝ := 175

theorem park_area (p : Park) : 
  p.perimeter * fencing_cost_per_meter = total_fencing_cost → 
  p.area = 7350 := by
  sorry

#check park_area

end NUMINAMATH_CALUDE_park_area_l3366_336645


namespace NUMINAMATH_CALUDE_equal_digit_prob_is_three_eighths_l3366_336626

/-- Represents a die with a given number of sides -/
structure Die :=
  (sides : ℕ)

/-- Probability of rolling a one-digit number on a given die -/
def prob_one_digit (d : Die) : ℚ :=
  if d.sides ≤ 9 then 1 else (9 : ℚ) / d.sides

/-- Probability of rolling a two-digit number on a given die -/
def prob_two_digit (d : Die) : ℚ :=
  1 - prob_one_digit d

/-- The set of dice used in the game -/
def game_dice : List Die :=
  [⟨6⟩, ⟨6⟩, ⟨6⟩, ⟨12⟩, ⟨12⟩]

/-- The probability of having an equal number of dice showing two-digit and one-digit numbers -/
def equal_digit_prob : ℚ :=
  2 * (prob_two_digit ⟨12⟩ * prob_one_digit ⟨12⟩)

theorem equal_digit_prob_is_three_eighths :
  equal_digit_prob = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_equal_digit_prob_is_three_eighths_l3366_336626


namespace NUMINAMATH_CALUDE_right_triangle_angle_bisector_segments_l3366_336687

/-- Given a right triangle where an acute angle bisector divides the adjacent leg into segments m and n,
    prove the lengths of the other leg and hypotenuse. -/
theorem right_triangle_angle_bisector_segments (m n : ℝ) (h : m > n) :
  ∃ (other_leg hypotenuse : ℝ),
    other_leg = n * Real.sqrt ((m + n) / (m - n)) ∧
    hypotenuse = m * Real.sqrt ((m + n) / (m - n)) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_angle_bisector_segments_l3366_336687


namespace NUMINAMATH_CALUDE_movie_ticket_ratio_l3366_336636

def monday_cost : ℚ := 5
def wednesday_cost : ℚ := 2 * monday_cost

theorem movie_ticket_ratio :
  ∃ (saturday_cost : ℚ),
    wednesday_cost + saturday_cost = 35 ∧
    saturday_cost / monday_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_ratio_l3366_336636


namespace NUMINAMATH_CALUDE_root_product_l3366_336608

theorem root_product (f g : ℝ → ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ) :
  (∀ x, f x = x^5 - x^3 + 1) →
  (∀ x, g x = x^2 - 2) →
  f x₁ = 0 ∧ f x₂ = 0 ∧ f x₃ = 0 ∧ f x₄ = 0 ∧ f x₅ = 0 →
  g x₁ * g x₂ * g x₃ * g x₄ * g x₅ = -7 := by
  sorry

end NUMINAMATH_CALUDE_root_product_l3366_336608


namespace NUMINAMATH_CALUDE_carpool_gas_expense_l3366_336653

/-- Calculates the monthly gas expense per person in a carpool scenario -/
theorem carpool_gas_expense
  (one_way_commute : ℝ)
  (gas_cost_per_gallon : ℝ)
  (car_efficiency : ℝ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (num_people : ℕ)
  (h1 : one_way_commute = 21)
  (h2 : gas_cost_per_gallon = 2.5)
  (h3 : car_efficiency = 30)
  (h4 : days_per_week = 5)
  (h5 : weeks_per_month = 4)
  (h6 : num_people = 5) :
  (2 * one_way_commute * days_per_week * weeks_per_month / car_efficiency * gas_cost_per_gallon) / num_people = 14 := by
  sorry


end NUMINAMATH_CALUDE_carpool_gas_expense_l3366_336653


namespace NUMINAMATH_CALUDE_afternoon_emails_l3366_336623

/-- The number of emails Jack received in the morning -/
def morning_emails : ℕ := 9

/-- The number of emails Jack received in the evening -/
def evening_emails : ℕ := 7

/-- The difference between morning and evening emails -/
def email_difference : ℕ := 2

/-- Theorem stating that Jack received 7 emails in the afternoon -/
theorem afternoon_emails : ℕ := by
  sorry

end NUMINAMATH_CALUDE_afternoon_emails_l3366_336623


namespace NUMINAMATH_CALUDE_range_of_a_l3366_336604

open Set

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Icc 1 2, x^2 - a ≥ 0) ∨ 
  (∃ x₀ : ℝ, ∀ x : ℝ, x + (a - 1) * x₀ + 1 < 0) ∧
  ¬((∀ x ∈ Icc 1 2, x^2 - a ≥ 0) ∧ 
    (∃ x₀ : ℝ, ∀ x : ℝ, x + (a - 1) * x₀ + 1 < 0)) →
  a > 3 ∨ a ∈ Icc (-1) 1 :=
by sorry

#check range_of_a

end NUMINAMATH_CALUDE_range_of_a_l3366_336604


namespace NUMINAMATH_CALUDE_largest_number_l3366_336656

theorem largest_number (a b c d e f : ℝ) 
  (ha : a = 0.986) 
  (hb : b = 0.9859) 
  (hc : c = 0.98609) 
  (hd : d = 0.896) 
  (he : e = 0.8979) 
  (hf : f = 0.987) : 
  f = max a (max b (max c (max d (max e f)))) :=
sorry

end NUMINAMATH_CALUDE_largest_number_l3366_336656


namespace NUMINAMATH_CALUDE_delta_value_l3366_336617

theorem delta_value : ∀ Δ : ℤ, 5 * (-3) = Δ - 3 → Δ = -12 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l3366_336617


namespace NUMINAMATH_CALUDE_evaluate_expression_l3366_336694

theorem evaluate_expression : 
  2011 * 20122012 * 201320132013 - 2013 * 20112011 * 201220122012 = 
  -2 * 2012 * 2013 * 10001 * 100010001 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3366_336694


namespace NUMINAMATH_CALUDE_intersection_of_N_and_complement_of_M_l3366_336619

open Set

theorem intersection_of_N_and_complement_of_M : 
  let M : Set ℝ := {x | x > 2}
  let N : Set ℝ := {x | 1 < x ∧ x < 3}
  (N ∩ (univ \ M)) = {x : ℝ | 1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_N_and_complement_of_M_l3366_336619


namespace NUMINAMATH_CALUDE_probability_is_one_seventh_l3366_336680

/-- The number of students in the group -/
def total_students : ℕ := 7

/-- The number of students who can speak a foreign language -/
def foreign_language_speakers : ℕ := 3

/-- The number of students being selected -/
def selected_students : ℕ := 2

/-- The probability of selecting two students who both speak a foreign language -/
def probability_both_speak_foreign : ℚ :=
  (foreign_language_speakers.choose selected_students) / (total_students.choose selected_students)

theorem probability_is_one_seventh :
  probability_both_speak_foreign = 1 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_one_seventh_l3366_336680


namespace NUMINAMATH_CALUDE_remainder_theorem_l3366_336638

theorem remainder_theorem (r : ℝ) : 
  (r^14 - r + 5) % (r - 1) = 5 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3366_336638


namespace NUMINAMATH_CALUDE_previous_largest_spider_weight_l3366_336606

/-- Proves the weight of the previous largest spider given the characteristics of a giant spider. -/
theorem previous_largest_spider_weight
  (weight_ratio : ℝ)
  (leg_count : ℕ)
  (leg_area : ℝ)
  (leg_pressure : ℝ)
  (h1 : weight_ratio = 2.5)
  (h2 : leg_count = 8)
  (h3 : leg_area = 0.5)
  (h4 : leg_pressure = 4) :
  let giant_spider_weight := leg_count * leg_area * leg_pressure
  giant_spider_weight / weight_ratio = 6.4 := by
sorry

end NUMINAMATH_CALUDE_previous_largest_spider_weight_l3366_336606


namespace NUMINAMATH_CALUDE_banana_arrangements_l3366_336689

def banana_length : ℕ := 6
def num_a : ℕ := 3
def num_n : ℕ := 2

theorem banana_arrangements : 
  (banana_length.factorial) / (num_a.factorial * num_n.factorial) = 60 := by
  sorry

end NUMINAMATH_CALUDE_banana_arrangements_l3366_336689


namespace NUMINAMATH_CALUDE_largest_number_with_sum_19_l3366_336651

def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc
    else aux (m / 10) ((m % 10) :: acc)
  aux n []

def sum_digits (n : ℕ) : ℕ :=
  (digits n).sum

def all_digits_different (n : ℕ) : Prop :=
  (digits n).Nodup

theorem largest_number_with_sum_19 :
  ∀ n : ℕ, 
    sum_digits n = 19 → 
    all_digits_different n → 
    n ≤ 65431 :=
by sorry

end NUMINAMATH_CALUDE_largest_number_with_sum_19_l3366_336651


namespace NUMINAMATH_CALUDE_probability_two_black_balls_l3366_336601

/-- The probability of drawing two black balls from a box containing white and black balls. -/
theorem probability_two_black_balls (white_balls black_balls : ℕ) 
  (h_white : white_balls = 7) (h_black : black_balls = 8) : 
  (black_balls.choose 2 : ℚ) / ((white_balls + black_balls).choose 2) = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_black_balls_l3366_336601


namespace NUMINAMATH_CALUDE_probability_same_color_adjacent_l3366_336639

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents an arrangement of balls -/
def Arrangement := List Color

/-- The set of all possible arrangements of 2 red and 2 white balls -/
def allArrangements : Finset Arrangement := sorry

/-- Checks if an arrangement has balls of the same color adjacent -/
def hasSameColorAdjacent (arr : Arrangement) : Bool := sorry

/-- Counts the number of arrangements with balls of the same color adjacent -/
def countSameColorAdjacent : Nat := sorry

/-- The total number of possible arrangements -/
def totalArrangements : Nat := sorry

theorem probability_same_color_adjacent :
  (countSameColorAdjacent : ℚ) / totalArrangements = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_adjacent_l3366_336639


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3366_336695

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of specific terms in the sequence equals 120 -/
def SumCondition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : ArithmeticSequence a) (h2 : SumCondition a) : 
  3 * a 9 - a 11 = 48 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3366_336695


namespace NUMINAMATH_CALUDE_eating_contest_l3366_336677

/-- Eating contest problem -/
theorem eating_contest (hot_dog_weight burger_weight pie_weight : ℕ)
  (noah_burgers jacob_pies mason_hotdog_weight : ℕ) :
  hot_dog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  jacob_pies = noah_burgers - 3 →
  noah_burgers = 8 →
  mason_hotdog_weight = 30 →
  mason_hotdog_weight / hot_dog_weight = 15 :=
by sorry

end NUMINAMATH_CALUDE_eating_contest_l3366_336677


namespace NUMINAMATH_CALUDE_toothpick_20th_stage_l3366_336678

def toothpick_sequence (n : ℕ) : ℕ :=
  3 + 3 * (n - 1)

theorem toothpick_20th_stage :
  toothpick_sequence 20 = 60 := by
  sorry

end NUMINAMATH_CALUDE_toothpick_20th_stage_l3366_336678


namespace NUMINAMATH_CALUDE_pizza_toppings_l3366_336662

theorem pizza_toppings (total_slices ham_slices olive_slices : ℕ) 
  (h_total : total_slices = 16)
  (h_ham : ham_slices = 8)
  (h_olive : olive_slices = 12)
  (h_at_least_one : ∀ slice, slice ≤ total_slices → (slice ≤ ham_slices ∨ slice ≤ olive_slices)) :
  ∃ both : ℕ, both = ham_slices + olive_slices - total_slices :=
by sorry

end NUMINAMATH_CALUDE_pizza_toppings_l3366_336662


namespace NUMINAMATH_CALUDE_base3_to_decimal_10101_l3366_336697

/-- Converts a base 3 number represented as a list of digits to its decimal (base 10) equivalent -/
def base3ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

/-- The base 3 representation of the number we want to convert -/
def base3Number : List Nat := [1, 0, 1, 0, 1]

/-- Theorem stating that the base 3 number 10101 is equal to 91 in base 10 -/
theorem base3_to_decimal_10101 :
  base3ToDecimal base3Number = 91 := by
  sorry

#eval base3ToDecimal base3Number -- This should output 91

end NUMINAMATH_CALUDE_base3_to_decimal_10101_l3366_336697


namespace NUMINAMATH_CALUDE_firefly_count_l3366_336670

/-- The number of fireflies remaining after a series of events --/
def remaining_fireflies (initial : ℕ) (joined : ℕ) (left : ℕ) : ℕ :=
  initial + joined - left

/-- Theorem stating the number of remaining fireflies in the given scenario --/
theorem firefly_count : remaining_fireflies 3 8 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_firefly_count_l3366_336670


namespace NUMINAMATH_CALUDE_smallest_n_for_candy_l3366_336663

theorem smallest_n_for_candy (n : ℕ+) : 
  (∃ m : ℕ+, 16 ∣ m ∧ 18 ∣ m ∧ 20 ∣ m ∧ m = 30 * n) →
  (∀ k : ℕ+, k < n → ¬∃ m : ℕ+, 16 ∣ m ∧ 18 ∣ m ∧ 20 ∣ m ∧ m = 30 * k) →
  n = 24 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_for_candy_l3366_336663


namespace NUMINAMATH_CALUDE_m_is_smallest_m_cube_root_n_positive_r_bounds_n_equals_six_l3366_336673

/-- The smallest integer whose cube root is of the form n + r, where n is a positive integer and r is a positive real number less than 1/100 -/
def m : ℕ := sorry

/-- A positive integer n such that the cube root of m is of the form n + r -/
def n : ℕ := sorry

/-- A positive real number r less than 1/100 such that the cube root of m is of the form n + r -/
def r : ℝ := sorry

/-- m is the smallest integer satisfying the given conditions -/
theorem m_is_smallest (k : ℕ) (h : k < m) : 
  ∀ (a : ℕ) (b : ℝ), (a : ℝ) + b = k^(1/3 : ℝ) → ¬(a > 0 ∧ b > 0 ∧ b < 1/100) := sorry

/-- The cube root of m is of the form n + r -/
theorem m_cube_root : (m : ℝ)^(1/3 : ℝ) = n + r := sorry

/-- n is a positive integer -/
theorem n_positive : n > 0 := sorry

/-- r is a positive real number less than 1/100 -/
theorem r_bounds : 0 < r ∧ r < 1/100 := sorry

/-- The main theorem: n equals 6 -/
theorem n_equals_six : n = 6 := sorry

end NUMINAMATH_CALUDE_m_is_smallest_m_cube_root_n_positive_r_bounds_n_equals_six_l3366_336673


namespace NUMINAMATH_CALUDE_cylinder_radius_determination_l3366_336627

/-- Given a cylinder with height 4 units, if increasing its radius by 3 units
    and increasing its height by 3 units both result in the same volume increase,
    then the original radius of the cylinder is 12 units. -/
theorem cylinder_radius_determination (r : ℝ) (y : ℝ) : 
  (4 * π * ((r + 3)^2 - r^2) = y) →
  (3 * π * r^2 = y) →
  r = 12 := by sorry

end NUMINAMATH_CALUDE_cylinder_radius_determination_l3366_336627


namespace NUMINAMATH_CALUDE_circle_equation_radius_five_l3366_336630

/-- A circle equation in the form x^2 + 8x + y^2 + 4y - k = 0 -/
def CircleEquation (x y k : ℝ) : Prop :=
  x^2 + 8*x + y^2 + 4*y - k = 0

/-- The standard form of a circle equation with center (h, j) and radius r -/
def StandardCircleEquation (x y h j r : ℝ) : Prop :=
  (x - h)^2 + (y - j)^2 = r^2

theorem circle_equation_radius_five (k : ℝ) :
  (∀ x y, CircleEquation x y k ↔ StandardCircleEquation x y (-4) (-2) 5) ↔ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_radius_five_l3366_336630


namespace NUMINAMATH_CALUDE_tangent_line_implies_a_equals_one_l3366_336665

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a

theorem tangent_line_implies_a_equals_one (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ = x₀ ∧ (deriv (f a)) x₀ = 1) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_implies_a_equals_one_l3366_336665


namespace NUMINAMATH_CALUDE_equation_solution_l3366_336643

theorem equation_solution :
  ∃ x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 ∧ x = -13/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3366_336643


namespace NUMINAMATH_CALUDE_coefficient_x4_equals_180_l3366_336624

/-- The coefficient of x^4 in the expansion of (2 + √x - 1/x^2016)^10 -/
def coefficient_x4 (x : ℝ) : ℕ :=
  -- We define this as a natural number since coefficients in polynomial expansions are typically integers
  -- The actual computation is not implemented here
  sorry

/-- The main theorem stating that the coefficient of x^4 is 180 -/
theorem coefficient_x4_equals_180 :
  ∀ x : ℝ, coefficient_x4 x = 180 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x4_equals_180_l3366_336624


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3366_336618

-- Define the equation
def equation (k : ℝ) (x : ℝ) : Prop :=
  (x - 3) / (k * x + 2) = x

-- Define the condition for exactly one solution
def has_exactly_one_solution (k : ℝ) : Prop :=
  ∃! x : ℝ, equation k x

-- Theorem statement
theorem unique_solution_condition :
  ∀ k : ℝ, has_exactly_one_solution k ↔ k = -1/12 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3366_336618


namespace NUMINAMATH_CALUDE_largest_triangle_area_21_points_l3366_336693

/-- A configuration of points where every three adjacent points form an equilateral triangle --/
structure TriangleConfiguration where
  num_points : ℕ
  small_triangle_area : ℝ

/-- The area of the largest triangle formed by the configuration --/
def largest_triangle_area (config : TriangleConfiguration) : ℝ :=
  sorry

/-- Theorem stating that for a configuration of 21 points with unit area small triangles,
    the largest triangle has an area of 13 --/
theorem largest_triangle_area_21_points :
  let config : TriangleConfiguration := { num_points := 21, small_triangle_area := 1 }
  largest_triangle_area config = 13 := by
  sorry

end NUMINAMATH_CALUDE_largest_triangle_area_21_points_l3366_336693


namespace NUMINAMATH_CALUDE_work_completion_time_l3366_336696

theorem work_completion_time (a_total_days b_remaining_days : ℚ) 
  (h1 : a_total_days = 15)
  (h2 : b_remaining_days = 10) : 
  let a_work_days : ℚ := 5
  let a_work_fraction : ℚ := a_work_days / a_total_days
  let b_work_fraction : ℚ := 1 - a_work_fraction
  b_remaining_days / b_work_fraction = 15 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3366_336696


namespace NUMINAMATH_CALUDE_quartic_roots_sum_product_l3366_336683

theorem quartic_roots_sum_product (p q : ℝ) : 
  (p^4 - 6*p - 1 = 0) → 
  (q^4 - 6*q - 1 = 0) → 
  (p ≠ q) →
  (∀ x : ℝ, x^4 - 6*x - 1 = 0 → x = p ∨ x = q) →
  p*q + p + q = 1 := by
sorry

end NUMINAMATH_CALUDE_quartic_roots_sum_product_l3366_336683


namespace NUMINAMATH_CALUDE_distance_between_points_l3366_336676

/-- The distance between two points A(4,-3) and B(4,5) is 8. -/
theorem distance_between_points : Real.sqrt ((4 - 4)^2 + (5 - (-3))^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l3366_336676


namespace NUMINAMATH_CALUDE_pascal_triangle_101_row_third_number_l3366_336620

/-- The number of elements in a row of Pascal's triangle -/
def row_elements (n : ℕ) : ℕ := n + 1

/-- The third number in a row of Pascal's triangle -/
def third_number (n : ℕ) : ℕ := n.choose 2

theorem pascal_triangle_101_row_third_number :
  ∃ (n : ℕ), row_elements n = 101 ∧ third_number n = 4950 :=
by sorry

end NUMINAMATH_CALUDE_pascal_triangle_101_row_third_number_l3366_336620


namespace NUMINAMATH_CALUDE_square_difference_ratio_l3366_336634

theorem square_difference_ratio : 
  (1630^2 - 1623^2) / (1640^2 - 1613^2) = 7/27 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_ratio_l3366_336634


namespace NUMINAMATH_CALUDE_total_collection_l3366_336690

def friend_payment : ℕ := 5
def brother_payment : ℕ := 8
def cousin_payment : ℕ := 4
def num_days : ℕ := 7

theorem total_collection :
  (friend_payment * num_days) + (brother_payment * num_days) + (cousin_payment * num_days) = 119 := by
  sorry

end NUMINAMATH_CALUDE_total_collection_l3366_336690


namespace NUMINAMATH_CALUDE_tank_capacity_l3366_336614

/-- Represents a water tank with a certain capacity -/
structure WaterTank where
  capacity : ℝ
  emptyWeight : ℝ
  waterWeight : ℝ
  filledWeight : ℝ
  filledPercentage : ℝ

/-- Theorem stating that a tank with the given properties has a capacity of 200 gallons -/
theorem tank_capacity (tank : WaterTank) 
  (h1 : tank.emptyWeight = 80)
  (h2 : tank.waterWeight = 8)
  (h3 : tank.filledWeight = 1360)
  (h4 : tank.filledPercentage = 0.8) :
  tank.capacity = 200 := by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l3366_336614


namespace NUMINAMATH_CALUDE_ball_probability_after_swap_l3366_336612

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  red : ℕ
  yellow : ℕ
  blue : ℕ

/-- Calculates the probability of drawing a ball of a specific color -/
def probability (bag : BagContents) (color : ℕ) : ℚ :=
  color / (bag.red + bag.yellow + bag.blue)

/-- The initial contents of the bag -/
def initialBag : BagContents :=
  { red := 10, yellow := 2, blue := 8 }

/-- The contents of the bag after removing red balls and adding yellow balls -/
def finalBag (n : ℕ) : BagContents :=
  { red := initialBag.red - n, yellow := initialBag.yellow + n, blue := initialBag.blue }

theorem ball_probability_after_swap :
  probability (finalBag 6) (finalBag 6).yellow = 2/5 :=
sorry

end NUMINAMATH_CALUDE_ball_probability_after_swap_l3366_336612


namespace NUMINAMATH_CALUDE_equation_solutions_l3366_336628

theorem equation_solutions : 
  {x : ℝ | (x - 1) * (x - 3) * (x - 5) * (x - 6) * (x - 3) * (x - 1) / 
           ((x - 3) * (x - 6) * (x - 3)) = 2 ∧ 
           x ≠ 3 ∧ x ≠ 6} = 
  {2 + Real.sqrt 2, 2 - Real.sqrt 2} := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l3366_336628


namespace NUMINAMATH_CALUDE_cone_volume_divided_by_pi_l3366_336657

/-- The volume of a cone formed from a 270-degree sector of a circle with radius 20 units, divided by π, is equal to 1125√7. -/
theorem cone_volume_divided_by_pi : 
  ∀ (r h : ℝ) (V : ℝ),
  -- Conditions
  (2 * π * r = 30 * π) →  -- Arc length becomes circumference of cone's base
  (20^2 = r^2 + h^2) →    -- Pythagorean theorem relating slant height to radius and height
  (V = (1/3) * π * r^2 * h) →  -- Volume formula for a cone
  -- Conclusion
  (V / π = 1125 * Real.sqrt 7) :=
by
  sorry

end NUMINAMATH_CALUDE_cone_volume_divided_by_pi_l3366_336657


namespace NUMINAMATH_CALUDE_empty_proper_subset_singleton_zero_l3366_336672

theorem empty_proper_subset_singleton_zero :
  ∅ ⊂ ({0} : Set ℕ) :=
sorry

end NUMINAMATH_CALUDE_empty_proper_subset_singleton_zero_l3366_336672


namespace NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l3366_336631

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (a b c d : Point3D) : ℝ := sorry

/-- Calculates the height of a tetrahedron from a vertex to the opposite face -/
def tetrahedronHeight (a b c d : Point3D) : ℝ := sorry

theorem tetrahedron_volume_and_height :
  let a₁ : Point3D := ⟨1, -1, 2⟩
  let a₂ : Point3D := ⟨2, 1, 2⟩
  let a₃ : Point3D := ⟨1, 1, 4⟩
  let a₄ : Point3D := ⟨6, -3, 8⟩
  (tetrahedronVolume a₁ a₂ a₃ a₄ = 6) ∧
  (tetrahedronHeight a₄ a₁ a₂ a₃ = 3 * Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_and_height_l3366_336631


namespace NUMINAMATH_CALUDE_pen_profit_percentage_pen_profit_percentage_result_l3366_336650

/-- Given a purchase of pens with specific pricing and discount conditions, 
    calculate the profit percentage. -/
theorem pen_profit_percentage 
  (num_pens : ℕ) 
  (marked_price_ratio : ℚ) 
  (discount_percent : ℚ) : ℚ :=
  let cost_price := marked_price_ratio
  let selling_price := num_pens * (1 - discount_percent / 100)
  let profit := selling_price - cost_price
  let profit_percent := (profit / cost_price) * 100
  by
    -- Assuming num_pens = 50, marked_price_ratio = 46/50, discount_percent = 1
    sorry

/-- The profit percentage for the given pen sale scenario is 7.61%. -/
theorem pen_profit_percentage_result : 
  pen_profit_percentage 50 (46/50) 1 = 761/100 :=
by sorry

end NUMINAMATH_CALUDE_pen_profit_percentage_pen_profit_percentage_result_l3366_336650


namespace NUMINAMATH_CALUDE_min_value_expression_l3366_336664

theorem min_value_expression (x y : ℝ) : 5 * x^2 + 4 * y^2 - 8 * x * y + 2 * x + 4 ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3366_336664


namespace NUMINAMATH_CALUDE_min_cans_required_l3366_336640

def can_capacity : ℕ := 10
def tank_capacity : ℕ := 140

theorem min_cans_required : 
  ∃ n : ℕ, n * can_capacity ≥ tank_capacity ∧ 
  ∀ m : ℕ, m * can_capacity ≥ tank_capacity → m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_min_cans_required_l3366_336640


namespace NUMINAMATH_CALUDE_flour_to_add_l3366_336616

theorem flour_to_add (recipe_amount : ℕ) (already_added : ℕ) (h1 : recipe_amount = 8) (h2 : already_added = 2) :
  recipe_amount - already_added = 6 := by
  sorry

end NUMINAMATH_CALUDE_flour_to_add_l3366_336616


namespace NUMINAMATH_CALUDE_count_divisors_with_specific_remainder_l3366_336660

theorem count_divisors_with_specific_remainder :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, n > 17 ∧ 2017 % n = 17) ∧
    (∀ n : ℕ, n > 17 ∧ 2017 % n = 17 → n ∈ S) ∧
    S.card = 13 :=
by sorry

end NUMINAMATH_CALUDE_count_divisors_with_specific_remainder_l3366_336660


namespace NUMINAMATH_CALUDE_multiply_by_99999_l3366_336666

theorem multiply_by_99999 (x : ℝ) : x * 99999 = 58293485180 → x = 582.935 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_99999_l3366_336666


namespace NUMINAMATH_CALUDE_Q_roots_nature_l3366_336686

def Q (x : ℝ) : ℝ := x^7 - 2*x^6 - 6*x^4 - 4*x + 16

theorem Q_roots_nature :
  (∀ x < 0, Q x > 0) ∧ 
  (∃ x > 0, Q x < 0) ∧ 
  (∃ x > 0, Q x > 0) :=
by sorry

end NUMINAMATH_CALUDE_Q_roots_nature_l3366_336686


namespace NUMINAMATH_CALUDE_f_composed_with_g_l3366_336621

def f (x : ℝ) : ℝ := 3 * x - 4

def g (x : ℝ) : ℝ := x + 2

theorem f_composed_with_g : f (2 + g 3) = 17 := by
  sorry

end NUMINAMATH_CALUDE_f_composed_with_g_l3366_336621


namespace NUMINAMATH_CALUDE_base_4_arithmetic_l3366_336622

/-- Converts a base 4 number to base 10 --/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 --/
def to_base_4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem base_4_arithmetic :
  to_base_4 (to_base_10 [0, 3, 2] - to_base_10 [1, 0, 1] + to_base_10 [2, 2, 3]) = [1, 1, 1, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_4_arithmetic_l3366_336622


namespace NUMINAMATH_CALUDE_total_snake_owners_is_75_l3366_336625

/-- The number of people in the neighborhood who own pets -/
def total_population : ℕ := 200

/-- The number of people who own only dogs -/
def only_dogs : ℕ := 30

/-- The number of people who own only cats -/
def only_cats : ℕ := 25

/-- The number of people who own only birds -/
def only_birds : ℕ := 10

/-- The number of people who own only snakes -/
def only_snakes : ℕ := 7

/-- The number of people who own only fish -/
def only_fish : ℕ := 12

/-- The number of people who own both cats and dogs -/
def cats_and_dogs : ℕ := 15

/-- The number of people who own both birds and dogs -/
def birds_and_dogs : ℕ := 12

/-- The number of people who own both birds and cats -/
def birds_and_cats : ℕ := 8

/-- The number of people who own both snakes and dogs -/
def snakes_and_dogs : ℕ := 3

/-- The number of people who own both snakes and cats -/
def snakes_and_cats : ℕ := 4

/-- The number of people who own both snakes and birds -/
def snakes_and_birds : ℕ := 2

/-- The number of people who own both fish and dogs -/
def fish_and_dogs : ℕ := 9

/-- The number of people who own both fish and cats -/
def fish_and_cats : ℕ := 6

/-- The number of people who own both fish and birds -/
def fish_and_birds : ℕ := 14

/-- The number of people who own both fish and snakes -/
def fish_and_snakes : ℕ := 11

/-- The number of people who own cats, dogs, and snakes -/
def cats_dogs_snakes : ℕ := 5

/-- The number of people who own cats, dogs, and birds -/
def cats_dogs_birds : ℕ := 4

/-- The number of people who own cats, birds, and snakes -/
def cats_birds_snakes : ℕ := 6

/-- The number of people who own dogs, birds, and snakes -/
def dogs_birds_snakes : ℕ := 9

/-- The number of people who own cats, fish, and dogs -/
def cats_fish_dogs : ℕ := 7

/-- The number of people who own birds, fish, and dogs -/
def birds_fish_dogs : ℕ := 5

/-- The number of people who own birds, fish, and cats -/
def birds_fish_cats : ℕ := 3

/-- The number of people who own snakes, fish, and dogs -/
def snakes_fish_dogs : ℕ := 8

/-- The number of people who own snakes, fish, and cats -/
def snakes_fish_cats : ℕ := 4

/-- The number of people who own snakes, fish, and birds -/
def snakes_fish_birds : ℕ := 6

/-- The number of people who own all five pets -/
def all_five_pets : ℕ := 10

/-- The total number of snake owners in the neighborhood -/
def total_snake_owners : ℕ := 
  only_snakes + snakes_and_dogs + snakes_and_cats + snakes_and_birds + 
  fish_and_snakes + cats_dogs_snakes + cats_birds_snakes + dogs_birds_snakes + 
  snakes_fish_dogs + snakes_fish_cats + snakes_fish_birds + all_five_pets

theorem total_snake_owners_is_75 : total_snake_owners = 75 := by
  sorry

end NUMINAMATH_CALUDE_total_snake_owners_is_75_l3366_336625


namespace NUMINAMATH_CALUDE_prism_length_l3366_336648

/-- A regular rectangular prism with given edge sum and proportions -/
structure RegularPrism where
  width : ℝ
  length : ℝ
  height : ℝ
  edge_sum : ℝ
  length_prop : length = 4 * width
  height_prop : height = 3 * width
  sum_prop : 4 * length + 4 * width + 4 * height = edge_sum

/-- The length of a regular rectangular prism with edge sum 256 cm is 32 cm -/
theorem prism_length (p : RegularPrism) (h : p.edge_sum = 256) : p.length = 32 := by
  sorry

end NUMINAMATH_CALUDE_prism_length_l3366_336648


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l3366_336698

/-- Given a set of 50 numbers with arithmetic mean 38, prove that removing 45 and 55
    results in a new set with arithmetic mean 37.5 -/
theorem arithmetic_mean_after_removal (S : Finset ℝ) (sum_S : ℝ) : 
  S.card = 50 →
  sum_S = S.sum id →
  sum_S / 50 = 38 →
  45 ∈ S →
  55 ∈ S →
  let S' := S.erase 45 |>.erase 55
  let sum_S' := sum_S - 45 - 55
  sum_S' / S'.card = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l3366_336698


namespace NUMINAMATH_CALUDE_lobachevsky_angle_existence_l3366_336611

theorem lobachevsky_angle_existence (A B C : Real) 
  (hB : 0 < B ∧ B < Real.pi / 2) 
  (hC : 0 < C ∧ C < Real.pi / 2) : 
  ∃ X, Real.sin X = (Real.sin B * Real.sin C) / (1 - Real.cos A * Real.cos B * Real.cos C) := by
  sorry

end NUMINAMATH_CALUDE_lobachevsky_angle_existence_l3366_336611


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3366_336609

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (a-1)*x + 1 ≤ 0) → a ∈ Set.Ioo (-1 : ℝ) 3 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3366_336609


namespace NUMINAMATH_CALUDE_exponent_division_thirteen_eleven_div_thirteen_four_l3366_336684

theorem exponent_division (a : ℕ) (m n : ℕ) (h : a > 0) : a^m / a^n = a^(m - n) := by sorry

theorem thirteen_eleven_div_thirteen_four :
  (13 : ℕ)^11 / (13 : ℕ)^4 = (13 : ℕ)^7 := by sorry

end NUMINAMATH_CALUDE_exponent_division_thirteen_eleven_div_thirteen_four_l3366_336684


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_positive_no_solution_exists_l3366_336674

theorem quadratic_inequality_always_positive (m : ℝ) (h : m > 1) :
  ∀ x : ℝ, x^2 - 2*x + m > 0 :=
by sorry

theorem no_solution_exists (m : ℝ) (h : m > 1) :
  ¬ ∃ x : ℝ, x^2 - 2*x + m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_positive_no_solution_exists_l3366_336674


namespace NUMINAMATH_CALUDE_boat_current_speed_l3366_336668

/-- Proves that given a boat with a speed of 18 km/hr in still water, traveling downstream
    for 14 minutes and covering a distance of 5.133333333333334 km, the rate of the current
    is 4 km/hr. -/
theorem boat_current_speed
  (boat_speed : ℝ)
  (travel_time : ℝ)
  (distance : ℝ)
  (h1 : boat_speed = 18)
  (h2 : travel_time = 14 / 60)
  (h3 : distance = 5.133333333333334) :
  let current_speed := (distance / travel_time) - boat_speed
  current_speed = 4 := by
sorry

end NUMINAMATH_CALUDE_boat_current_speed_l3366_336668


namespace NUMINAMATH_CALUDE_another_beast_holds_all_candy_l3366_336671

/-- Represents the state of candy distribution among beasts -/
inductive CandyDistribution
  | initial (n : ℕ)  -- Initial distribution with Grogg having n candies
  | distribute (d : List ℕ)  -- List representing candy counts for each beast

/-- Represents a single step in the candy distribution process -/
def distributeStep (d : CandyDistribution) : CandyDistribution :=
  match d with
  | CandyDistribution.initial n => CandyDistribution.distribute (List.replicate n 1)
  | CandyDistribution.distribute (k :: rest) => 
      CandyDistribution.distribute (List.map (· + 1) (List.take k rest) ++ List.drop k rest)
  | _ => d

/-- Checks if all candy is held by a single beast (except Grogg) -/
def allCandyHeldBySingleBeast (d : CandyDistribution) : Bool :=
  match d with
  | CandyDistribution.distribute [n] => true
  | _ => false

/-- Main theorem: Another beast holds all candy iff n = 1 or n = 2 -/
theorem another_beast_holds_all_candy (n : ℕ) (h : n ≥ 1) :
  (∃ d : CandyDistribution, d = distributeStep (CandyDistribution.initial n) ∧ 
    allCandyHeldBySingleBeast d) ↔ n = 1 ∨ n = 2 :=
  sorry

end NUMINAMATH_CALUDE_another_beast_holds_all_candy_l3366_336671


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l3366_336632

theorem smallest_divisible_by_1_to_10 : ∃ n : ℕ, n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧ n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l3366_336632


namespace NUMINAMATH_CALUDE_calculate_expression_l3366_336600

theorem calculate_expression : (81 : ℝ) ^ (1/4) * (81 : ℝ) ^ (1/5) * 2 = 20.09 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3366_336600


namespace NUMINAMATH_CALUDE_quadratic_one_solution_l3366_336655

theorem quadratic_one_solution (m : ℚ) : 
  (∃! x, 3 * x^2 - 7 * x + m = 0) ↔ m = 49/12 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_l3366_336655


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l3366_336613

theorem trigonometric_simplification (α : ℝ) :
  (2 * Real.sin (π - α) + Real.sin (2 * α)) / (2 * (Real.cos (α / 2))^2) = 2 * Real.sin α :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l3366_336613


namespace NUMINAMATH_CALUDE_linear_function_decreasing_l3366_336685

theorem linear_function_decreasing (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = -3 * x₁ - 7 →
  y₂ = -3 * x₂ - 7 →
  x₁ > x₂ →
  y₁ < y₂ := by
sorry

end NUMINAMATH_CALUDE_linear_function_decreasing_l3366_336685


namespace NUMINAMATH_CALUDE_log_expression_equals_zero_l3366_336699

-- Define base 10 logarithm
noncomputable def log10 (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_expression_equals_zero :
  (1/2) * log10 4 + log10 5 - (π + 1)^0 = 0 := by sorry

end NUMINAMATH_CALUDE_log_expression_equals_zero_l3366_336699


namespace NUMINAMATH_CALUDE_rectangle_area_perimeter_inequality_l3366_336607

theorem rectangle_area_perimeter_inequality (a b : ℕ+) : (a + 2) * (b + 2) - 8 ≠ 100 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_perimeter_inequality_l3366_336607


namespace NUMINAMATH_CALUDE_fathers_cookies_l3366_336669

theorem fathers_cookies (total cookies_charlie cookies_mother : ℕ) 
  (h1 : total = 30)
  (h2 : cookies_charlie = 15)
  (h3 : cookies_mother = 5) :
  total - cookies_charlie - cookies_mother = 10 := by
sorry

end NUMINAMATH_CALUDE_fathers_cookies_l3366_336669


namespace NUMINAMATH_CALUDE_k_h_negative_three_equals_sixteen_l3366_336682

-- Define the function h
def h (x : ℝ) : ℝ := 4 * x^2 - 8

-- Define a variable k as a function from ℝ to ℝ
variable (k : ℝ → ℝ)

-- State the theorem
theorem k_h_negative_three_equals_sixteen 
  (h_def : ∀ x, h x = 4 * x^2 - 8)
  (k_h_three : k (h 3) = 16) :
  k (h (-3)) = 16 := by
  sorry

end NUMINAMATH_CALUDE_k_h_negative_three_equals_sixteen_l3366_336682


namespace NUMINAMATH_CALUDE_water_usage_calculation_l3366_336642

/-- Water pricing policy and usage calculation -/
theorem water_usage_calculation (m : ℝ) (usage : ℝ) (payment : ℝ) : 
  (m > 0) →
  (usage > 0) →
  (payment = if usage ≤ 10 then m * usage else 10 * m + 2 * m * (usage - 10)) →
  (payment = 16 * m) →
  (usage = 13) :=
by sorry

end NUMINAMATH_CALUDE_water_usage_calculation_l3366_336642


namespace NUMINAMATH_CALUDE_monotonically_decreasing_implies_a_leq_1_l3366_336667

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x

-- State the theorem
theorem monotonically_decreasing_implies_a_leq_1 :
  ∀ a : ℝ, (∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f a x > f a y) → a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_monotonically_decreasing_implies_a_leq_1_l3366_336667


namespace NUMINAMATH_CALUDE_tomato_production_l3366_336675

/-- The number of tomatoes produced by the first plant -/
def plant1_tomatoes : ℕ := 24

/-- The number of tomatoes produced by the second plant -/
def plant2_tomatoes : ℕ := plant1_tomatoes / 2 + 5

/-- The number of tomatoes produced by the third plant -/
def plant3_tomatoes : ℕ := plant2_tomatoes + 2

/-- The total number of tomatoes produced by all three plants -/
def total_tomatoes : ℕ := plant1_tomatoes + plant2_tomatoes + plant3_tomatoes

theorem tomato_production : total_tomatoes = 60 := by
  sorry

end NUMINAMATH_CALUDE_tomato_production_l3366_336675


namespace NUMINAMATH_CALUDE_distance_is_sqrt_1501_div_17_l3366_336637

/-- The distance from a point to a line in 3D space -/
def distance_point_to_line (point : ℝ × ℝ × ℝ) (line_point : ℝ × ℝ × ℝ) (line_direction : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- The given point -/
def given_point : ℝ × ℝ × ℝ := (2, 3, 4)

/-- A point on the given line -/
def line_point : ℝ × ℝ × ℝ := (5, 6, 8)

/-- The direction vector of the given line -/
def line_direction : ℝ × ℝ × ℝ := (4, 3, -3)

/-- Theorem stating that the distance from the given point to the line is √1501 / 17 -/
theorem distance_is_sqrt_1501_div_17 : 
  distance_point_to_line given_point line_point line_direction = Real.sqrt 1501 / 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_sqrt_1501_div_17_l3366_336637


namespace NUMINAMATH_CALUDE_equation_solutions_l3366_336661

theorem equation_solutions :
  (∀ x : ℝ, 4 * x^2 - 81 = 0 ↔ x = 9/2 ∨ x = -9/2) ∧
  (∀ x : ℝ, (x - 1)^3 = -8 ↔ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3366_336661


namespace NUMINAMATH_CALUDE_delivery_pay_difference_l3366_336605

/-- Calculate the difference in pay between two workers --/
theorem delivery_pay_difference (deliveries_worker1 : ℕ) 
  (pay_per_delivery : ℕ) : 
  deliveries_worker1 = 96 →
  pay_per_delivery = 100 →
  (deliveries_worker1 * pay_per_delivery : ℕ) - 
  ((deliveries_worker1 * 3 / 4) * pay_per_delivery : ℕ) = 2400 := by
sorry

end NUMINAMATH_CALUDE_delivery_pay_difference_l3366_336605


namespace NUMINAMATH_CALUDE_number_of_zeros_equal_l3366_336652

/-- f(n) denotes the number of 0's in the binary representation of a positive integer n -/
def f (n : ℕ+) : ℕ := sorry

/-- Theorem stating that the number of 0's in the binary representation of 8n + 7 
    is equal to the number of 0's in the binary representation of 4n + 3 -/
theorem number_of_zeros_equal (n : ℕ+) : f (8 * n + 7) = f (4 * n + 3) := by
  sorry

end NUMINAMATH_CALUDE_number_of_zeros_equal_l3366_336652


namespace NUMINAMATH_CALUDE_arrangements_count_is_correct_l3366_336641

/-- The number of ways to divide 2 teachers and 4 students into two groups,
    each containing 1 teacher and 2 students, and then assign these groups to two locations -/
def arrangementsCount : ℕ := 12

/-- The number of ways to choose 2 students from 4 students -/
def waysToChooseStudents : ℕ := Nat.choose 4 2

/-- The number of ways to choose 2 students from 2 students (always 1) -/
def waysToChooseRemainingStudents : ℕ := Nat.choose 2 2

/-- The number of ways to assign 2 groups to 2 locations -/
def waysToAssignGroups : ℕ := 2

theorem arrangements_count_is_correct :
  arrangementsCount = waysToChooseStudents * waysToChooseRemainingStudents * waysToAssignGroups :=
by sorry

end NUMINAMATH_CALUDE_arrangements_count_is_correct_l3366_336641


namespace NUMINAMATH_CALUDE_specific_rectangle_burning_time_l3366_336654

/-- Represents a rectangular structure made of toothpicks -/
structure ToothpickRectangle where
  rows : ℕ
  columns : ℕ
  toothpicks : ℕ

/-- Represents the burning properties of toothpicks -/
structure BurningProperties where
  burn_time_per_toothpick : ℕ
  start_corners : ℕ

/-- Calculates the total burning time for a toothpick rectangle -/
def total_burning_time (rect : ToothpickRectangle) (props : BurningProperties) : ℕ :=
  sorry

/-- Theorem statement for the burning time of the specific rectangle -/
theorem specific_rectangle_burning_time :
  let rect := ToothpickRectangle.mk 3 5 38
  let props := BurningProperties.mk 10 2
  total_burning_time rect props = 65 :=
by sorry

end NUMINAMATH_CALUDE_specific_rectangle_burning_time_l3366_336654


namespace NUMINAMATH_CALUDE_volume_of_one_gram_volume_of_one_gram_substance_l3366_336679

-- Define the constants from the problem
def mass_per_cubic_meter : ℝ := 300
def grams_per_kilogram : ℝ := 1000
def cubic_cm_per_cubic_meter : ℝ := 1000000

-- Define the theorem
theorem volume_of_one_gram (mass_per_cubic_meter : ℝ) (grams_per_kilogram : ℝ) (cubic_cm_per_cubic_meter : ℝ) :
  mass_per_cubic_meter * grams_per_kilogram > 0 →
  cubic_cm_per_cubic_meter / (mass_per_cubic_meter * grams_per_kilogram) = 10 / 3 := by
  sorry

-- Apply the theorem to our specific values
theorem volume_of_one_gram_substance :
  cubic_cm_per_cubic_meter / (mass_per_cubic_meter * grams_per_kilogram) = 10 / 3 := by
  apply volume_of_one_gram mass_per_cubic_meter grams_per_kilogram cubic_cm_per_cubic_meter
  -- Prove that mass_per_cubic_meter * grams_per_kilogram > 0
  sorry

end NUMINAMATH_CALUDE_volume_of_one_gram_volume_of_one_gram_substance_l3366_336679


namespace NUMINAMATH_CALUDE_min_value_of_function_l3366_336610

open Real

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  let f := fun (x : ℝ) => x - 1 - (log x) / x
  (∀ y > 0, f y ≥ 0) ∧ (∃ z > 0, f z = 0) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l3366_336610


namespace NUMINAMATH_CALUDE_candy_count_proof_l3366_336688

/-- Calculates the total number of candy pieces given the number of packages and pieces per package -/
def total_candy_pieces (packages : ℕ) (pieces_per_package : ℕ) : ℕ :=
  packages * pieces_per_package

/-- Proves that 45 packages of candy with 9 pieces each results in 405 total pieces -/
theorem candy_count_proof :
  total_candy_pieces 45 9 = 405 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_proof_l3366_336688


namespace NUMINAMATH_CALUDE_sum_of_valid_a_values_l3366_336646

theorem sum_of_valid_a_values : ∃ (S : Finset ℤ), 
  (∀ a ∈ S, (∀ y : ℝ, ¬(y - 1 ≥ (2*y - 1)/3 ∧ -1/2*(y - a) > 0)) ∧ 
              (∃ x : ℝ, x < 0 ∧ a/(x + 1) + 1 = (x + a)/(x - 1))) ∧
  (∀ a : ℤ, (∀ y : ℝ, ¬(y - 1 ≥ (2*y - 1)/3 ∧ -1/2*(y - a) > 0)) ∧ 
             (∃ x : ℝ, x < 0 ∧ a/(x + 1) + 1 = (x + a)/(x - 1)) → a ∈ S) ∧
  (Finset.sum S (λ a => a) = 3) := by
sorry

end NUMINAMATH_CALUDE_sum_of_valid_a_values_l3366_336646
