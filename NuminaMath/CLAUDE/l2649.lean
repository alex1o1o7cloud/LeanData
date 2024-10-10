import Mathlib

namespace toy_truck_cost_l2649_264995

/-- The amount spent on a toy truck given initial amount, pencil case cost, and remaining amount -/
theorem toy_truck_cost (initial : ℝ) (pencil_case : ℝ) (remaining : ℝ) :
  initial = 10 → pencil_case = 2 → remaining = 5 →
  initial - pencil_case - remaining = 3 := by
  sorry

end toy_truck_cost_l2649_264995


namespace decimal_124_to_base_5_has_three_consecutive_digits_l2649_264912

/-- Convert a decimal number to base 5 --/
def to_base_5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Check if a list of digits has three consecutive identical digits --/
def has_three_consecutive_digits (digits : List ℕ) : Prop :=
  ∃ i, i + 2 < digits.length ∧
       digits[i]! = digits[i+1]! ∧
       digits[i+1]! = digits[i+2]!

/-- The main theorem --/
theorem decimal_124_to_base_5_has_three_consecutive_digits :
  has_three_consecutive_digits (to_base_5 124) :=
sorry

end decimal_124_to_base_5_has_three_consecutive_digits_l2649_264912


namespace center_top_second_row_value_l2649_264921

/-- Represents a 4x4 grid of real numbers -/
def Grid := Fin 4 → Fin 4 → ℝ

/-- Checks if a sequence of 4 real numbers is arithmetic -/
def IsArithmeticSequence (s : Fin 4 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Fin 3, s (i + 1) - s i = d

/-- The property that each row and column of the grid is an arithmetic sequence -/
def GridProperty (g : Grid) : Prop :=
  (∀ i : Fin 4, IsArithmeticSequence (λ j ↦ g i j)) ∧
  (∀ j : Fin 4, IsArithmeticSequence (λ i ↦ g i j))

theorem center_top_second_row_value
  (g : Grid)
  (h_grid : GridProperty g)
  (h_first_row : g 0 0 = 4 ∧ g 0 3 = 16)
  (h_last_row : g 3 0 = 10 ∧ g 3 3 = 40) :
  g 1 1 = 12 := by
  sorry

end center_top_second_row_value_l2649_264921


namespace wax_calculation_l2649_264933

theorem wax_calculation (total_wax : ℕ) (additional_wax : ℕ) (possessed_wax : ℕ) : 
  total_wax = 353 → additional_wax = 22 → possessed_wax = total_wax - additional_wax → possessed_wax = 331 := by
  sorry

end wax_calculation_l2649_264933


namespace negative_one_to_zero_power_l2649_264966

theorem negative_one_to_zero_power : (-1 : ℝ) ^ (0 : ℕ) = 1 := by sorry

end negative_one_to_zero_power_l2649_264966


namespace elvin_internet_charge_l2649_264947

/-- Represents Elvin's monthly telephone bill structure -/
structure MonthlyBill where
  callCharge : ℝ
  internetCharge : ℝ

/-- Calculates the total bill amount -/
def totalBill (bill : MonthlyBill) : ℝ :=
  bill.callCharge + bill.internetCharge

theorem elvin_internet_charge :
  ∀ (jan_bill feb_bill : MonthlyBill),
    totalBill jan_bill = 46 →
    totalBill feb_bill = 76 →
    feb_bill.callCharge = 2 * jan_bill.callCharge →
    jan_bill.internetCharge = feb_bill.internetCharge →
    jan_bill.internetCharge = 16 := by
  sorry

end elvin_internet_charge_l2649_264947


namespace triangle_side_length_l2649_264954

theorem triangle_side_length (a b c : ℝ) (C : ℝ) :
  a = 5 → b = 3 → C = 2 * π / 3 → c = 7 := by sorry

end triangle_side_length_l2649_264954


namespace even_sum_and_sum_greater_20_count_l2649_264906

def IntSet := Finset (Nat)

def range_1_to_20 : IntSet := Finset.range 20

def even_sum_pairs (s : IntSet) : Nat :=
  (s.filter (λ x => x ≤ 20)).card

def sum_greater_20_pairs (s : IntSet) : Nat :=
  (s.filter (λ x => x ≤ 20)).card

theorem even_sum_and_sum_greater_20_count :
  (even_sum_pairs range_1_to_20 = 90) ∧
  (sum_greater_20_pairs range_1_to_20 = 100) := by
  sorry

end even_sum_and_sum_greater_20_count_l2649_264906


namespace smallest_divisible_by_72_l2649_264952

theorem smallest_divisible_by_72 (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬(72 ∣ m * 40)) ∧ 
  (72 ∣ n * 40) ∧ 
  (n ≥ 5) ∧
  (∃ k : ℕ, n * 40 = 72 * k) →
  n = 5 :=
sorry

end smallest_divisible_by_72_l2649_264952


namespace divided_triangle_perimeter_l2649_264997

/-- Represents a triangle divided into smaller triangles -/
structure DividedTriangle where
  large_perimeter : ℝ
  num_small_triangles : ℕ
  small_perimeter : ℝ

/-- Theorem stating the relationship between the perimeters of the large and small triangles -/
theorem divided_triangle_perimeter
  (t : DividedTriangle)
  (h1 : t.large_perimeter = 120)
  (h2 : t.num_small_triangles = 9)
  (h3 : t.small_perimeter * 3 = t.large_perimeter) :
  t.small_perimeter = 40 :=
sorry

end divided_triangle_perimeter_l2649_264997


namespace davids_physics_marks_l2649_264928

def english_marks : ℕ := 36
def math_marks : ℕ := 35
def chemistry_marks : ℕ := 57
def biology_marks : ℕ := 55
def average_marks : ℕ := 45
def num_subjects : ℕ := 5

theorem davids_physics_marks :
  let total_marks := average_marks * num_subjects
  let known_marks := english_marks + math_marks + chemistry_marks + biology_marks
  let physics_marks := total_marks - known_marks
  physics_marks = 42 := by sorry

end davids_physics_marks_l2649_264928


namespace james_flowers_per_day_l2649_264922

theorem james_flowers_per_day 
  (total_volunteers : ℕ) 
  (days_worked : ℕ) 
  (total_flowers : ℕ) 
  (h1 : total_volunteers = 5)
  (h2 : days_worked = 2)
  (h3 : total_flowers = 200)
  (h4 : total_flowers % (total_volunteers * days_worked) = 0) :
  total_flowers / (total_volunteers * days_worked) = 20 := by
sorry

end james_flowers_per_day_l2649_264922


namespace marble_remainder_l2649_264940

theorem marble_remainder (r p : ℤ) 
  (hr : r % 5 = 2) 
  (hp : p % 5 = 4) : 
  (r + p) % 5 = 1 := by
sorry

end marble_remainder_l2649_264940


namespace square_side_length_l2649_264911

theorem square_side_length (total_width total_height : ℕ) 
  (h_width : total_width = 4040)
  (h_height : total_height = 2420)
  (h_rectangles_equal : ∃ (r : ℕ), r = r) -- R₁ and R₂ have identical dimensions
  (h_squares : ∃ (s r : ℕ), s + r = s + r) -- S₁ and S₃ side length = S₂ side length + R₁ side length
  : ∃ (s : ℕ), s = 810 ∧ 
    ∃ (r : ℕ), 2 * r + s = total_height ∧ 
                2 * r + 3 * s = total_width :=
by
  sorry

end square_side_length_l2649_264911


namespace samson_sandwiches_l2649_264904

/-- The number of sandwiches Samson ate for breakfast on Tuesday -/
def tuesday_breakfast : ℕ := 1

theorem samson_sandwiches (monday_lunch : ℕ) (monday_dinner : ℕ) (monday_total : ℕ) :
  monday_lunch = 3 →
  monday_dinner = 2 * monday_lunch →
  monday_total = monday_lunch + monday_dinner →
  monday_total = tuesday_breakfast + 8 →
  tuesday_breakfast = 1 := by sorry

end samson_sandwiches_l2649_264904


namespace det_B_squared_minus_3B_l2649_264983

def B : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 3, 2]

theorem det_B_squared_minus_3B : Matrix.det ((B ^ 2) - 3 • B) = 88 := by
  sorry

end det_B_squared_minus_3B_l2649_264983


namespace count_special_divisors_l2649_264990

/-- The number of positive integer divisors of 998^49999 that are not divisors of 998^49998 -/
def special_divisors : ℕ := 99999

/-- 998 as a product of its prime factors -/
def factor_998 : ℕ × ℕ := (2, 499)

theorem count_special_divisors :
  (factor_998.1 * factor_998.2)^49999 = 998^49999 →
  (∃ (d : ℕ → ℕ × ℕ),
    (∀ (n : ℕ), n < special_divisors →
      (factor_998.1^(d n).1 * factor_998.2^(d n).2 ∣ 998^49999) ∧
      ¬(factor_998.1^(d n).1 * factor_998.2^(d n).2 ∣ 998^49998)) ∧
    (∀ (n m : ℕ), n < special_divisors → m < special_divisors → n ≠ m →
      factor_998.1^(d n).1 * factor_998.2^(d n).2 ≠ factor_998.1^(d m).1 * factor_998.2^(d m).2) ∧
    (∀ (k : ℕ), (k ∣ 998^49999) ∧ ¬(k ∣ 998^49998) →
      ∃ (n : ℕ), n < special_divisors ∧ k = factor_998.1^(d n).1 * factor_998.2^(d n).2)) :=
by sorry

end count_special_divisors_l2649_264990


namespace boat_speed_in_still_water_l2649_264930

/-- Proves that the speed of a boat in still water is 13 km/hr given the conditions -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 4)
  (h2 : downstream_distance = 68)
  (h3 : downstream_time = 4)
  : ∃ (boat_speed : ℝ), boat_speed = 13 ∧ 
    downstream_distance = (boat_speed + stream_speed) * downstream_time :=
by
  sorry

end boat_speed_in_still_water_l2649_264930


namespace angle_in_fourth_quadrant_l2649_264938

/-- Given that angle α satisfies the conditions sin(2α) < 0 and sin(α) - cos(α) < 0,
    prove that α is in the fourth quadrant. -/
theorem angle_in_fourth_quadrant (α : Real) 
    (h1 : Real.sin (2 * α) < 0) 
    (h2 : Real.sin α - Real.cos α < 0) : 
  π < α ∧ α < (3 * π) / 2 := by
  sorry

end angle_in_fourth_quadrant_l2649_264938


namespace expected_red_pairs_value_l2649_264920

/-- Represents a standard 104-card deck -/
structure Deck :=
  (cards : Finset (Fin 104))
  (size : cards.card = 104)

/-- Represents the color of a card -/
inductive Color
| Red
| Black

/-- Function to determine the color of a card -/
def color (card : Fin 104) : Color :=
  if card.val ≤ 51 then Color.Red else Color.Black

/-- Number of red cards in the deck -/
def num_red_cards : Nat := 52

/-- Calculates the expected number of adjacent red card pairs in a 104-card deck -/
def expected_red_pairs (d : Deck) : ℚ :=
  (num_red_cards : ℚ) * ((num_red_cards - 1) / (d.cards.card - 1))

/-- Theorem: The expected number of adjacent red card pairs is 2652/103 -/
theorem expected_red_pairs_value (d : Deck) :
  expected_red_pairs d = 2652 / 103 := by
  sorry

end expected_red_pairs_value_l2649_264920


namespace repeating_decimal_6_is_two_thirds_l2649_264927

def repeating_decimal_6 : ℚ := 0.6666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666

theorem repeating_decimal_6_is_two_thirds : repeating_decimal_6 = 2/3 := by
  sorry

end repeating_decimal_6_is_two_thirds_l2649_264927


namespace prob_no_standing_pairs_10_l2649_264965

/-- Represents the number of valid arrangements for n people where no two adjacent people form a standing pair -/
def b : ℕ → ℕ
| 0 => 1
| 1 => 2
| n+2 => 3 * b (n+1) - b n

/-- The probability of no standing pairs for n people -/
def prob_no_standing_pairs (n : ℕ) : ℚ :=
  (b n : ℚ) / (2^n : ℚ)

theorem prob_no_standing_pairs_10 :
  prob_no_standing_pairs 10 = 31 / 128 := by sorry

end prob_no_standing_pairs_10_l2649_264965


namespace stream_speed_l2649_264919

/-- Proves that given a boat with a speed of 57 km/h in still water, 
    if the time taken to row upstream is twice the time taken to row downstream 
    for the same distance, then the speed of the stream is 19 km/h. -/
theorem stream_speed (d : ℝ) (h : d > 0) : 
  let boat_speed := 57
  let stream_speed := 19
  (d / (boat_speed - stream_speed) = 2 * (d / (boat_speed + stream_speed))) →
  stream_speed = 19 := by
  sorry

end stream_speed_l2649_264919


namespace mothers_age_l2649_264955

theorem mothers_age (D M : ℕ) 
  (h1 : 2 * D + M = 70)
  (h2 : D + 2 * M = 95) : 
  M = 40 := by
sorry

end mothers_age_l2649_264955


namespace choir_members_count_l2649_264949

theorem choir_members_count : ∃ n₁ n₂ : ℕ, 
  150 < n₁ ∧ n₁ < 250 ∧
  150 < n₂ ∧ n₂ < 250 ∧
  n₁ % 3 = 1 ∧
  n₁ % 6 = 2 ∧
  n₁ % 8 = 3 ∧
  n₂ % 3 = 1 ∧
  n₂ % 6 = 2 ∧
  n₂ % 8 = 3 ∧
  n₁ = 195 ∧
  n₂ = 219 ∧
  ∀ n : ℕ, (150 < n ∧ n < 250 ∧ n % 3 = 1 ∧ n % 6 = 2 ∧ n % 8 = 3) → (n = 195 ∨ n = 219) :=
by sorry

end choir_members_count_l2649_264949


namespace distance_between_cities_l2649_264977

/-- The distance between two cities given the speeds of two travelers and their meeting point --/
theorem distance_between_cities (john_speed lewis_speed : ℝ) (meeting_point : ℝ) : 
  john_speed = 40 →
  lewis_speed = 60 →
  meeting_point = 160 →
  ∃ (distance : ℝ), 
    distance > 0 ∧ 
    (distance + meeting_point) / lewis_speed = (distance - meeting_point) / john_speed ∧
    distance = 800 := by
  sorry

end distance_between_cities_l2649_264977


namespace smallest_z_l2649_264971

theorem smallest_z (x y z : ℤ) : 
  x < y → y < z → 
  2 * y = x + z →  -- arithmetic progression
  z * z = x * y →  -- geometric progression
  z ≥ -2 :=
by sorry

end smallest_z_l2649_264971


namespace cyclists_meeting_time_l2649_264944

/-- Two cyclists on a circular track problem -/
theorem cyclists_meeting_time 
  (track_circumference : ℝ) 
  (speed1 speed2 : ℝ) 
  (h1 : track_circumference = 600) 
  (h2 : speed1 = 7) 
  (h3 : speed2 = 8) : 
  track_circumference / (speed1 + speed2) = 40 := by
  sorry

#check cyclists_meeting_time

end cyclists_meeting_time_l2649_264944


namespace power_of_i_third_quadrant_l2649_264962

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Statement 1: i^2023 = -i
theorem power_of_i : i^2023 = -i := by sorry

-- Statement 2: -2-i is in the third quadrant
theorem third_quadrant : 
  let z : ℂ := -2 - i
  z.re < 0 ∧ z.im < 0 := by sorry

end power_of_i_third_quadrant_l2649_264962


namespace shaded_percentage_is_59_l2649_264958

def large_square_side_length : ℕ := 5
def small_square_side_length : ℕ := 1
def border_squares_count : ℕ := 16
def shaded_border_squares_count : ℕ := 8
def central_region_shaded_fraction : ℚ := 3 / 4

theorem shaded_percentage_is_59 :
  let total_area : ℚ := (large_square_side_length ^ 2 : ℚ)
  let border_area : ℚ := (border_squares_count * small_square_side_length ^ 2 : ℚ)
  let central_area : ℚ := total_area - border_area
  let shaded_border_area : ℚ := (shaded_border_squares_count * small_square_side_length ^ 2 : ℚ)
  let shaded_central_area : ℚ := central_region_shaded_fraction * central_area
  let total_shaded_area : ℚ := shaded_border_area + shaded_central_area
  (total_shaded_area / total_area) * 100 = 59 :=
by sorry

end shaded_percentage_is_59_l2649_264958


namespace largest_multiple_of_9_under_100_l2649_264926

theorem largest_multiple_of_9_under_100 : 
  ∃ n : ℕ, n * 9 = 99 ∧ ∀ m : ℕ, m * 9 < 100 → m * 9 ≤ 99 :=
by sorry

end largest_multiple_of_9_under_100_l2649_264926


namespace log_difference_equals_eight_l2649_264979

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_difference_equals_eight :
  log 3 243 - log 3 (1/27) = 8 := by sorry

end log_difference_equals_eight_l2649_264979


namespace boat_stream_speed_l2649_264967

/-- Proves that given a boat with a speed of 36 kmph in still water,
    if it can cover 80 km downstream or 40 km upstream in the same time,
    then the speed of the stream is 12 kmph. -/
theorem boat_stream_speed 
  (boat_speed : ℝ) 
  (downstream_distance : ℝ) 
  (upstream_distance : ℝ) 
  (h1 : boat_speed = 36)
  (h2 : downstream_distance = 80)
  (h3 : upstream_distance = 40)
  (h4 : downstream_distance / (boat_speed + stream_speed) = upstream_distance / (boat_speed - stream_speed)) :
  stream_speed = 12 :=
by
  sorry

end boat_stream_speed_l2649_264967


namespace hiking_problem_l2649_264905

/-- Hiking problem -/
theorem hiking_problem (R_up : ℝ) (R_down : ℝ) (T_up : ℝ) (T_down : ℝ) (D_down : ℝ) :
  R_up = 7 →
  R_down = 1.5 * R_up →
  T_up = T_down →
  D_down = 21 →
  T_up = 2 :=
by sorry

end hiking_problem_l2649_264905


namespace jade_rate_ratio_l2649_264991

/-- The "jade rate" for a shape is the constant k in the volume formula V = kD³,
    where D is the characteristic length of the shape. -/
def jade_rate (volume : Real → Real) : Real :=
  volume 1

theorem jade_rate_ratio :
  let sphere_volume (a : Real) := (4 / 3) * Real.pi * (a / 2)^3
  let cylinder_volume (a : Real) := Real.pi * (a / 2)^2 * a
  let cube_volume (a : Real) := a^3
  let k₁ := jade_rate sphere_volume
  let k₂ := jade_rate cylinder_volume
  let k₃ := jade_rate cube_volume
  k₁ / k₂ = (Real.pi / 6) / (Real.pi / 4) ∧ k₂ / k₃ = Real.pi / 4 := by
  sorry


end jade_rate_ratio_l2649_264991


namespace cape_may_shark_count_l2649_264992

/-- The number of sharks in Daytona Beach -/
def daytona_sharks : ℕ := 12

/-- The number of sharks in Cape May -/
def cape_may_sharks : ℕ := 2 * daytona_sharks + 8

theorem cape_may_shark_count : cape_may_sharks = 32 := by
  sorry

end cape_may_shark_count_l2649_264992


namespace problem_solution_l2649_264978

theorem problem_solution (x y z : ℝ) 
  (h1 : y / (x - y) = x / (y + z))
  (h2 : z^2 = x*(y + z) - y*(x - y)) :
  (y^2 + z^2 - x^2) / (2*y*z) = 1/2 := by
sorry

end problem_solution_l2649_264978


namespace correct_article_usage_l2649_264923

/-- Represents the possible article choices for each blank -/
inductive Article
  | A
  | The
  | None

/-- Represents the sentence structure with two article blanks -/
structure Sentence where
  first_blank : Article
  second_blank : Article

/-- Defines the correct article usage based on the given conditions -/
def correct_usage : Sentence :=
  { first_blank := Article.A,  -- Gottlieb Daimler is referred to generally
    second_blank := Article.The }  -- The car invention is referred to specifically

/-- Theorem stating that the correct usage is "a" for the first blank and "the" for the second -/
theorem correct_article_usage :
  correct_usage = { first_blank := Article.A, second_blank := Article.The } :=
by sorry

end correct_article_usage_l2649_264923


namespace week_net_change_l2649_264915

/-- The net change in stock exchange points for a week -/
def net_change (monday tuesday wednesday thursday friday : Int) : Int :=
  monday + tuesday + wednesday + thursday + friday

/-- Theorem stating that the net change for the given week is -119 -/
theorem week_net_change :
  net_change (-150) 106 (-47) 182 (-210) = -119 := by
  sorry

end week_net_change_l2649_264915


namespace rectangle_area_l2649_264945

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 49 → 
  rectangle_width ^ 2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 147 := by
sorry

end rectangle_area_l2649_264945


namespace sin_sum_equality_l2649_264934

theorem sin_sum_equality : 
  Real.sin (7 * π / 30) + Real.sin (11 * π / 30) = 
  Real.sin (π / 30) + Real.sin (13 * π / 30) + 1 / 2 := by
  sorry

end sin_sum_equality_l2649_264934


namespace expression_evaluation_l2649_264974

theorem expression_evaluation (a b c d : ℤ) 
  (ha : a = 10) (hb : b = 15) (hc : c = 3) (hd : d = 2) : 
  (a - (b - c + d)) - ((a - b + d) - c) = 2 := by
  sorry

end expression_evaluation_l2649_264974


namespace farm_tax_problem_l2649_264935

/-- Represents the farm tax problem -/
theorem farm_tax_problem (total_tax : ℝ) (william_tax : ℝ) (taxable_percentage : ℝ) 
  (h1 : total_tax = 5000)
  (h2 : william_tax = 480)
  (h3 : taxable_percentage = 0.60) :
  william_tax / total_tax * 100 = 5.76 := by
  sorry

end farm_tax_problem_l2649_264935


namespace baseball_cards_count_l2649_264982

theorem baseball_cards_count (initial_cards additional_cards : ℕ) 
  (h1 : initial_cards = 87)
  (h2 : additional_cards = 13) :
  initial_cards + additional_cards = 100 :=
by sorry

end baseball_cards_count_l2649_264982


namespace shortest_chord_through_focus_of_ellipse_l2649_264941

/-- 
Given an ellipse with equation x²/16 + y²/9 = 1, 
prove that the length of the shortest chord passing through a focus is 9/2.
-/
theorem shortest_chord_through_focus_of_ellipse :
  let ellipse := fun (x y : ℝ) => x^2/16 + y^2/9 = 1
  ∃ (f : ℝ × ℝ), 
    (ellipse f.1 f.2) ∧ 
    (∀ (p q : ℝ × ℝ), ellipse p.1 p.2 ∧ ellipse q.1 q.2 ∧ 
      (∃ (t : ℝ), (1 - t) • f + t • p = q) →
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ 9/2) ∧
    (∃ (p q : ℝ × ℝ), ellipse p.1 p.2 ∧ ellipse q.1 q.2 ∧
      (∃ (t : ℝ), (1 - t) • f + t • p = q) ∧
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 9/2) :=
by sorry

end shortest_chord_through_focus_of_ellipse_l2649_264941


namespace meaningful_expression_l2649_264937

/-- The expression (x+3)/(x-1) + (x-2)^0 is meaningful if and only if x ≠ 1 -/
theorem meaningful_expression (x : ℝ) : 
  (∃ y : ℝ, y = (x + 3) / (x - 1) + (x - 2)^0) ↔ x ≠ 1 :=
sorry

end meaningful_expression_l2649_264937


namespace direct_inverse_variation_l2649_264943

theorem direct_inverse_variation (k : ℝ) (P₀ Q₀ R₀ P₁ R₁ : ℝ) :
  P₀ = k * Q₀ / Real.sqrt R₀ →
  P₀ = 9/4 →
  R₀ = 16/25 →
  Q₀ = 5/8 →
  P₁ = 27 →
  R₁ = 1/36 →
  k * (5/8) / Real.sqrt (16/25) = 9/4 →
  ∃ Q₁ : ℝ, P₁ = k * Q₁ / Real.sqrt R₁ ∧ Q₁ = 1.56 := by
sorry

end direct_inverse_variation_l2649_264943


namespace pennies_count_l2649_264969

def pennies_in_jar (nickels dimes quarters : ℕ) (ice_cream_cost leftover : ℕ) : ℕ :=
  let nickel_value := 5
  let dime_value := 10
  let quarter_value := 25
  let total_without_pennies := nickels * nickel_value + dimes * dime_value + quarters * quarter_value
  let total_in_jar := ice_cream_cost + leftover
  total_in_jar - total_without_pennies

theorem pennies_count (nickels dimes quarters : ℕ) (ice_cream_cost leftover : ℕ) :
  nickels = 85 → dimes = 35 → quarters = 26 → ice_cream_cost = 1500 → leftover = 48 →
  pennies_in_jar nickels dimes quarters ice_cream_cost leftover = 123 := by
  sorry

#eval pennies_in_jar 85 35 26 1500 48

end pennies_count_l2649_264969


namespace log_difference_inequality_l2649_264988

theorem log_difference_inequality (a b : ℝ) : 
  Real.log a - Real.log b = 3 * b - a → a > b ∧ b > 0 := by sorry

end log_difference_inequality_l2649_264988


namespace smallest_valid_number_last_four_digits_l2649_264996

def is_valid_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 5 ∨ d = 9

def contains_5_and_9 (n : ℕ) : Prop :=
  5 ∈ n.digits 10 ∧ 9 ∈ n.digits 10

theorem smallest_valid_number_last_four_digits :
  ∃ n : ℕ,
    (n > 0) ∧
    (n % 5 = 0) ∧
    (n % 9 = 0) ∧
    is_valid_number n ∧
    contains_5_and_9 n ∧
    (∀ m : ℕ, m > 0 ∧ m % 5 = 0 ∧ m % 9 = 0 ∧ is_valid_number m ∧ contains_5_and_9 m → n ≤ m) ∧
    (n % 10000 = 9995) :=
  sorry

end smallest_valid_number_last_four_digits_l2649_264996


namespace average_first_five_subjects_l2649_264948

/-- Given the average marks for 6 subjects and the marks for the 6th subject,
    calculate the average marks for the first 5 subjects. -/
theorem average_first_five_subjects
  (total_subjects : ℕ)
  (average_six_subjects : ℚ)
  (marks_sixth_subject : ℕ)
  (h1 : total_subjects = 6)
  (h2 : average_six_subjects = 78)
  (h3 : marks_sixth_subject = 98) :
  (average_six_subjects * total_subjects - marks_sixth_subject) / (total_subjects - 1) = 74 :=
by sorry

end average_first_five_subjects_l2649_264948


namespace exists_min_value_subject_to_constraint_l2649_264994

/-- The constraint function for a, b, c, d -/
def constraint (a b c d : ℝ) : Prop :=
  a^4 + b^4 + c^4 + d^4 = 16

/-- The function to be minimized -/
def objective (a b c d : ℝ) : ℝ :=
  a^3 + b^3 + c^3 + d^3

/-- Theorem stating the existence of a minimum value for the objective function
    subject to the given constraint -/
theorem exists_min_value_subject_to_constraint :
  ∃ (min : ℝ), ∀ (a b c d : ℝ), constraint a b c d →
    objective a b c d ≥ min ∧
    (∃ (a' b' c' d' : ℝ), constraint a' b' c' d' ∧ objective a' b' c' d' = min) :=
by sorry

end exists_min_value_subject_to_constraint_l2649_264994


namespace percentage_failed_hindi_l2649_264913

theorem percentage_failed_hindi (failed_english : ℝ) (failed_both : ℝ) (passed_both : ℝ) :
  failed_english = 50 →
  failed_both = 25 →
  passed_both = 50 →
  ∃ failed_hindi : ℝ, failed_hindi = 25 := by
  sorry

end percentage_failed_hindi_l2649_264913


namespace j_walking_speed_l2649_264984

/-- Represents the walking speed of J in kmph -/
def j_speed : ℝ := 5.945

/-- Represents the cycling speed of P in kmph -/
def p_speed : ℝ := 8

/-- Represents the time (in hours) between J's start and P's start -/
def time_difference : ℝ := 1.5

/-- Represents the total time (in hours) from J's start to when P catches up -/
def total_time : ℝ := 7.3

/-- Represents the time (in hours) P cycles before catching up to J -/
def p_cycle_time : ℝ := 5.8

/-- Represents the distance (in km) J is behind P when P catches up -/
def distance_behind : ℝ := 3

theorem j_walking_speed :
  p_speed * p_cycle_time = j_speed * total_time + distance_behind :=
sorry

end j_walking_speed_l2649_264984


namespace soccer_team_goals_l2649_264936

theorem soccer_team_goals (total_players : ℕ) (games_played : ℕ) (goals_other_players : ℕ) : 
  total_players = 24 →
  games_played = 15 →
  goals_other_players = 30 →
  (total_players / 3 * games_played + goals_other_players : ℕ) = 150 := by
  sorry

end soccer_team_goals_l2649_264936


namespace intersection_of_A_and_B_l2649_264914

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by
  sorry

end intersection_of_A_and_B_l2649_264914


namespace rayden_has_more_birds_l2649_264986

/-- The number of ducks Lily bought -/
def lily_ducks : ℕ := 20

/-- The number of geese Lily bought -/
def lily_geese : ℕ := 10

/-- The number of ducks Rayden bought -/
def rayden_ducks : ℕ := 3 * lily_ducks

/-- The number of geese Rayden bought -/
def rayden_geese : ℕ := 4 * lily_geese

/-- The difference in the total number of birds between Rayden and Lily -/
def bird_difference : ℕ := (rayden_ducks - lily_ducks) + (rayden_geese - lily_geese)

theorem rayden_has_more_birds :
  bird_difference = 70 := by sorry

end rayden_has_more_birds_l2649_264986


namespace exists_valid_assignment_with_difference_one_l2649_264939

/-- Represents a position on an infinite checkerboard -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Represents the color of a square on the checkerboard -/
inductive Color
  | White
  | Black

/-- Determines the color of a square based on its position -/
def color (p : Position) : Color :=
  if (p.x + p.y) % 2 = 0 then Color.White else Color.Black

/-- Represents an assignment of non-zero integers to white squares -/
def Assignment := Position → ℤ

/-- Checks if an assignment is valid (all non-zero integers on white squares) -/
def is_valid_assignment (f : Assignment) : Prop :=
  ∀ p, color p = Color.White → f p ≠ 0

/-- Calculates the product difference for a black square -/
def product_difference (f : Assignment) (p : Position) : ℤ :=
  f {x := p.x - 1, y := p.y} * f {x := p.x + 1, y := p.y} -
  f {x := p.x, y := p.y - 1} * f {x := p.x, y := p.y + 1}

/-- The main theorem: there exists a valid assignment satisfying the condition -/
theorem exists_valid_assignment_with_difference_one :
  ∃ f : Assignment, is_valid_assignment f ∧
    ∀ p, color p = Color.Black → product_difference f p = 1 :=
  sorry

end exists_valid_assignment_with_difference_one_l2649_264939


namespace complex_number_trigonometric_form_l2649_264970

/-- Prove that the complex number z = sin 36° + i cos 54° is equal to √2 sin 36° (cos 45° + i sin 45°) -/
theorem complex_number_trigonometric_form 
  (z : ℂ) 
  (h1 : z = Complex.ofReal (Real.sin (36 * π / 180)) + Complex.I * Complex.ofReal (Real.cos (54 * π / 180)))
  (h2 : Real.cos (54 * π / 180) = Real.sin (36 * π / 180)) :
  z = Complex.ofReal (Real.sqrt 2 * Real.sin (36 * π / 180)) * 
      (Complex.ofReal (Real.cos (45 * π / 180)) + Complex.I * Complex.ofReal (Real.sin (45 * π / 180))) :=
by sorry

end complex_number_trigonometric_form_l2649_264970


namespace ordering_abc_l2649_264999

theorem ordering_abc (a b c : ℝ) (ha : a = Real.log (11/10)) (hb : b = 1/10) (hc : c = 2/21) : b > a ∧ a > c := by
  sorry

end ordering_abc_l2649_264999


namespace triangle_property_l2649_264909

open Real

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a * sin B - Real.sqrt 3 * b * cos B * cos C = Real.sqrt 3 * c * (cos B)^2 →
  (B = π / 3 ∧
   (0 < C ∧ C < π / 2 → 1 < a^2 + b^2 ∧ a^2 + b^2 < 7)) :=
by sorry

end triangle_property_l2649_264909


namespace quadrilateral_vector_proof_l2649_264975

-- Define the space
variable (V : Type*) [AddCommGroup V] [Module ℝ V]

-- Define the points and vectors
variable (O A B C D M N : V)
variable (a b c : V)

-- State the theorem
theorem quadrilateral_vector_proof 
  (h1 : O + a = A) 
  (h2 : O + b = B) 
  (h3 : O + c = C) 
  (h4 : ∃ t : ℝ, M = O + t • a) 
  (h5 : M - O = 2 • (A - M)) 
  (h6 : N = (1/2) • B + (1/2) • C) :
  M - N = -(2/3) • a + (1/2) • b + (1/2) • c := by sorry

end quadrilateral_vector_proof_l2649_264975


namespace tetrahedron_volume_l2649_264985

/-- Given a tetrahedron with inradius R and face areas S₁, S₂, S₃, and S₄,
    its volume V is equal to (1/3)R(S₁ + S₂ + S₃ + S₄) -/
theorem tetrahedron_volume (R S₁ S₂ S₃ S₄ : ℝ) (hR : R > 0) (hS : S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0) :
  ∃ V : ℝ, V = (1/3) * R * (S₁ + S₂ + S₃ + S₄) ∧ V > 0 := by
  sorry

end tetrahedron_volume_l2649_264985


namespace tims_bodyguard_payment_l2649_264910

/-- The amount Tim pays his bodyguards in a week -/
def weekly_payment (num_bodyguards : ℕ) (hourly_rate : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  num_bodyguards * hourly_rate * hours_per_day * days_per_week

/-- Theorem stating the total amount Tim pays his bodyguards in a week -/
theorem tims_bodyguard_payment :
  weekly_payment 2 20 8 7 = 2240 := by
  sorry

#eval weekly_payment 2 20 8 7

end tims_bodyguard_payment_l2649_264910


namespace eighth_roll_last_probability_l2649_264953

/-- A standard six-sided die -/
def StandardDie : Type := Fin 6

/-- The probability of rolling a different number than the previous roll -/
def probDifferent : ℚ := 5/6

/-- The probability of rolling the same number as the previous roll -/
def probSame : ℚ := 1/6

/-- The number of rolls we're interested in -/
def numRolls : ℕ := 8

/-- The probability that the 8th roll is the last roll -/
def probEighthRollLast : ℚ := probDifferent^(numRolls - 2) * probSame

theorem eighth_roll_last_probability :
  probEighthRollLast = 15625/279936 := by sorry

end eighth_roll_last_probability_l2649_264953


namespace max_value_x_1_minus_2x_l2649_264918

theorem max_value_x_1_minus_2x : 
  ∃ (max : ℝ), max = 1/8 ∧ 
  (∀ x : ℝ, 0 < x → x < 1/2 → x * (1 - 2*x) ≤ max) ∧
  (∃ x : ℝ, 0 < x ∧ x < 1/2 ∧ x * (1 - 2*x) = max) := by
  sorry

end max_value_x_1_minus_2x_l2649_264918


namespace perpendicular_lines_l2649_264917

/-- Two lines y = ax - 2 and y = (a + 2)x + 1 are perpendicular if and only if a = -1 -/
theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, y = a * x - 2 ∧ y = (a + 2) * x + 1 → a * (a + 2) = -1) ↔ 
  a = -1 := by
  sorry

end perpendicular_lines_l2649_264917


namespace problem_solution_l2649_264968

-- Define x as the solution to the equation x = 1 + √3 / x
noncomputable def x : ℝ := Real.sqrt 3 + 1

-- State the theorem
theorem problem_solution :
  1 / ((x + 1) * (x - 3)) = (-Real.sqrt 3 - 4) / 13 := by
  sorry

end problem_solution_l2649_264968


namespace water_level_rise_l2649_264976

/-- Given a cube and a rectangular vessel, calculate the rise in water level when the cube is fully immersed. -/
theorem water_level_rise (cube_edge : ℝ) (vessel_length vessel_width : ℝ) (h_cube : cube_edge = 12) 
    (h_length : vessel_length = 20) (h_width : vessel_width = 15) : 
    (cube_edge ^ 3) / (vessel_length * vessel_width) = 5.76 := by
  sorry

end water_level_rise_l2649_264976


namespace sum_of_B_elements_l2649_264901

/-- A finite set with two elements -/
inductive TwoElementSet
  | e1
  | e2

/-- The mapping f from A to B -/
def f (x : TwoElementSet) : ℝ :=
  match x with
  | TwoElementSet.e1 => 1^2
  | TwoElementSet.e2 => 3^2

/-- The set B as a function from TwoElementSet to ℝ -/
def B : TwoElementSet → ℝ := f

theorem sum_of_B_elements : (B TwoElementSet.e1) + (B TwoElementSet.e2) = 10 :=
  sorry

end sum_of_B_elements_l2649_264901


namespace oak_trees_after_planting_l2649_264987

/-- The number of oak trees in the park after planting -/
def trees_after_planting (initial_trees new_trees : ℕ) : ℕ :=
  initial_trees + new_trees

/-- Theorem: There will be 11 oak trees after planting -/
theorem oak_trees_after_planting :
  trees_after_planting 9 2 = 11 := by
  sorry

end oak_trees_after_planting_l2649_264987


namespace tent_production_equation_l2649_264931

theorem tent_production_equation (x : ℝ) (h : x > 0) : 
  (7000 / x) - (7000 / (1.4 * x)) = 4 ↔ 
  ∃ (original_days actual_days : ℝ),
    original_days > 0 ∧ 
    actual_days > 0 ∧
    original_days = 7000 / x ∧ 
    actual_days = 7000 / (1.4 * x) ∧
    original_days - actual_days = 4 :=
by sorry

end tent_production_equation_l2649_264931


namespace s_range_l2649_264908

-- Define the piecewise function
noncomputable def s (t : ℝ) : ℝ :=
  if t ≥ 1 then 3 * t else 4 * t - t^2

-- State the theorem
theorem s_range :
  Set.range s = Set.Icc (-5 : ℝ) 9 := by sorry

end s_range_l2649_264908


namespace problem_statement_l2649_264959

theorem problem_statement (a b : ℝ) (h : a - 2*b - 3 = 0) : 9 - 2*a + 4*b = 3 := by
  sorry

end problem_statement_l2649_264959


namespace greatest_power_of_two_factor_l2649_264902

theorem greatest_power_of_two_factor (n : ℕ) : 
  ∃ (k : ℕ), 2^504 * k = 14^504 - 8^252 ∧ 
  ∀ (m : ℕ), 2^m * k = 14^504 - 8^252 → m ≤ 504 :=
sorry

end greatest_power_of_two_factor_l2649_264902


namespace xiaolong_exam_score_l2649_264951

theorem xiaolong_exam_score (total_questions : ℕ) (correct_points : ℕ) (incorrect_points : ℤ) 
  (xiaolong_score : ℕ) (max_answered : ℕ) :
  total_questions = 50 →
  correct_points = 3 →
  incorrect_points = -1 →
  xiaolong_score = 120 →
  max_answered = 48 →
  ∃ (correct incorrect : ℕ),
    correct + incorrect ≤ max_answered ∧
    correct * correct_points + incorrect * incorrect_points = xiaolong_score ∧
    correct ≤ 42 ∧
    ∀ (c i : ℕ), 
      c + i ≤ max_answered →
      c * correct_points + i * incorrect_points = xiaolong_score →
      c ≤ 42 :=
by sorry

end xiaolong_exam_score_l2649_264951


namespace triangle_inequality_l2649_264989

/-- For any triangle with sides a, b, c, angle A opposite side a, and semiperimeter p,
    the inequality (bc cos A) / (b + c) + a < p < (bc + a^2) / a holds. -/
theorem triangle_inequality (a b c : ℝ) (A : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_angle : 0 < A ∧ A < π) :
  let p := (a + b + c) / 2
  (b * c * Real.cos A) / (b + c) + a < p ∧ p < (b * c + a^2) / a :=
by sorry

end triangle_inequality_l2649_264989


namespace jade_transactions_l2649_264963

theorem jade_transactions (mabel_transactions : ℕ) 
  (anthony_transactions : ℕ) (cal_transactions : ℕ) (jade_transactions : ℕ) : 
  mabel_transactions = 90 →
  anthony_transactions = mabel_transactions + (mabel_transactions * 10 / 100) →
  cal_transactions = anthony_transactions * 2 / 3 →
  jade_transactions = cal_transactions + 16 →
  jade_transactions = 82 := by
sorry

end jade_transactions_l2649_264963


namespace tangent_unique_tangent_values_l2649_264903

/-- A line y = kx + 1 is tangent to the curve y = x^3 + ax + b at the point (1, 3) -/
def is_tangent (a b : ℝ) : Prop :=
  ∃ k : ℝ,
  (1 : ℝ)^3 + a * 1 + b = 3 ∧  -- The point (1, 3) is on the curve
  k * 1 + 1 = 3 ∧              -- The point (1, 3) is on the line
  3 * (1 : ℝ)^2 + a = k        -- The slope of the curve at x = 1 equals the slope of the line

/-- The values of a and b for which the line is tangent to the curve at (1, 3) are unique -/
theorem tangent_unique : ∃! (a b : ℝ), is_tangent a b :=
sorry

/-- The unique values of a and b for which the line is tangent to the curve at (1, 3) are -1 and 3 respectively -/
theorem tangent_values : ∃! (a b : ℝ), is_tangent a b ∧ a = -1 ∧ b = 3 :=
sorry

end tangent_unique_tangent_values_l2649_264903


namespace local_max_implies_c_eq_6_l2649_264981

/-- The function f(x) = x(x-c)^2 has a local maximum at x=2 -/
def has_local_max_at_2 (c : ℝ) : Prop :=
  let f := fun x => x * (x - c)^2
  (∃ δ > 0, ∀ x, |x - 2| < δ → f x ≤ f 2) ∧
  (∀ ε > 0, ∃ x, |x - 2| < ε ∧ f x < f 2)

/-- If f(x) = x(x-c)^2 has a local maximum at x=2, then c = 6 -/
theorem local_max_implies_c_eq_6 :
  ∀ c : ℝ, has_local_max_at_2 c → c = 6 := by
  sorry

end local_max_implies_c_eq_6_l2649_264981


namespace student_arrangement_count_l2649_264998

/-- The number of ways to arrange students into communities --/
def arrange_students (total_students : ℕ) (selected_students : ℕ) (communities : ℕ) (min_per_community : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct number of arrangements for the given problem --/
theorem student_arrangement_count :
  arrange_students 7 6 2 2 = 350 := by
  sorry

end student_arrangement_count_l2649_264998


namespace least_sum_equation_l2649_264980

theorem least_sum_equation (x y z : ℕ+) 
  (h1 : 6 * z.val = 2 * x.val) 
  (h2 : x.val + y.val + z.val = 26) : 
  6 * z.val = 36 := by
sorry

end least_sum_equation_l2649_264980


namespace product_of_differences_l2649_264946

theorem product_of_differences (m n : ℝ) 
  (hm : m = 1 / (Real.sqrt 3 + Real.sqrt 2)) 
  (hn : n = 1 / (Real.sqrt 3 - Real.sqrt 2)) : 
  (m - 1) * (n - 1) = -2 * Real.sqrt 3 := by
  sorry

end product_of_differences_l2649_264946


namespace median_of_special_list_l2649_264960

/-- The sum of integers from 1 to n -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total count of numbers in the list -/
def total_count : ℕ := triangular_number 250

/-- The position of the median in the list -/
def median_position : ℕ := total_count / 2 + 1

/-- The number that appears at the median position -/
def median_number : ℕ := 177

theorem median_of_special_list :
  median_number = 177 ∧
  triangular_number (median_number - 1) < median_position ∧
  median_position ≤ triangular_number median_number :=
sorry

end median_of_special_list_l2649_264960


namespace geometric_sequence_sum_eight_l2649_264925

/-- Represents a geometric sequence with common ratio 2 -/
def GeometricSequence (a : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => 2 * GeometricSequence a n

/-- Sum of the first n terms of the geometric sequence -/
def SumGeometric (a : ℝ) (n : ℕ) : ℝ :=
  (List.range n).map (GeometricSequence a) |>.sum

theorem geometric_sequence_sum_eight (a : ℝ) :
  SumGeometric a 4 = 1 → SumGeometric a 8 = 17 := by
  sorry

#check geometric_sequence_sum_eight

end geometric_sequence_sum_eight_l2649_264925


namespace right_triangle_area_l2649_264972

/-- Given a right triangle with one leg of length a and the ratio of its circumradius
    to inradius being 5:2, its area is either 2a²/3 or 3a²/8 -/
theorem right_triangle_area (a : ℝ) (h : a > 0) :
  ∃ (R r : ℝ), R > 0 ∧ r > 0 ∧ R / r = 5 / 2 ∧
  (∃ (b c : ℝ), b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧
   (1/2 * a * b = 2*a^2/3 ∨ 1/2 * a * b = 3*a^2/8)) :=
by sorry


end right_triangle_area_l2649_264972


namespace average_of_numbers_l2649_264900

def numbers : List ℕ := [12, 13, 14, 510, 520, 530, 1120, 1, 1252140, 2345]

theorem average_of_numbers :
  (numbers.sum : ℚ) / numbers.length = 125320.5 := by sorry

end average_of_numbers_l2649_264900


namespace solution_set_of_inequalities_l2649_264916

theorem solution_set_of_inequalities :
  ∀ x : ℝ, (2 * x > -1 ∧ x - 1 ≤ 0) ↔ (-1/2 < x ∧ x ≤ 1) := by sorry

end solution_set_of_inequalities_l2649_264916


namespace johnson_martinez_tie_l2649_264929

/-- Represents the months of the baseball season --/
inductive Month
| Mar
| Apr
| May
| Jun
| Jul
| Aug
| Sep

/-- Calculates the cumulative home runs for a player --/
def cumulativeHomeRuns (monthlyData : List Nat) : List Nat :=
  List.scanl (· + ·) 0 monthlyData

/-- Checks if two lists are equal up to a certain index --/
def equalUpTo (l1 l2 : List Nat) (index : Nat) : Bool :=
  (l1.take index) = (l2.take index)

/-- Finds the first index where two lists become equal --/
def firstEqualIndex (l1 l2 : List Nat) : Option Nat :=
  (List.range l1.length).find? (fun i => l1[i]! = l2[i]!)

theorem johnson_martinez_tie (johnsonData martinezData : List Nat) 
    (h1 : johnsonData = [3, 8, 15, 12, 5, 7, 14])
    (h2 : martinezData = [0, 3, 9, 20, 7, 12, 13]) : 
    firstEqualIndex 
      (cumulativeHomeRuns johnsonData) 
      (cumulativeHomeRuns martinezData) = some 6 := by
  sorry

#check johnson_martinez_tie

end johnson_martinez_tie_l2649_264929


namespace algebraic_expressions_l2649_264932

theorem algebraic_expressions (x y : ℝ) 
  (h1 : x + y = 5) 
  (h2 : (x - 2) * (y - 2) = -3) : 
  x * y = 3 ∧ x^2 + 4*x*y + y^2 = 31 ∧ x^2 + x*y + 5*y = 25 := by
  sorry

end algebraic_expressions_l2649_264932


namespace calculation_proof_l2649_264973

theorem calculation_proof : (1955 - 1875)^2 / 64 = 100 := by
  sorry

end calculation_proof_l2649_264973


namespace total_money_calculation_l2649_264956

theorem total_money_calculation (p q r total : ℝ) 
  (h1 : r = (2/3) * total) 
  (h2 : r = 3600) : 
  total = 5400 := by
sorry

end total_money_calculation_l2649_264956


namespace square_to_three_squares_l2649_264957

/-- A partition of a square is a list of polygons that cover the square without overlap -/
def Partition (a : ℝ) := List (List (ℝ × ℝ))

/-- A square is a list of four points representing its vertices -/
def Square := List (ℝ × ℝ)

/-- Predicate to check if a partition is valid (covers the whole square without overlap) -/
def is_valid_partition (a : ℝ) (p : Partition a) : Prop := sorry

/-- Predicate to check if a list of points forms a square -/
def is_square (s : Square) : Prop := sorry

/-- Predicate to check if a partition can be rearranged to form given squares -/
def can_form_squares (a : ℝ) (p : Partition a) (squares : List Square) : Prop := sorry

/-- Theorem stating that a square can be cut into 4 parts to form 3 squares -/
theorem square_to_three_squares (a : ℝ) : 
  ∃ (p : Partition a) (s₁ s₂ s₃ : Square), 
    is_valid_partition a p ∧ 
    p.length = 4 ∧
    is_square s₁ ∧ is_square s₂ ∧ is_square s₃ ∧
    can_form_squares a p [s₁, s₂, s₃] := by
  sorry

end square_to_three_squares_l2649_264957


namespace radius_of_inscribed_circle_in_curvilinear_triangle_l2649_264924

/-- 
Given a rhombus with height h and acute angle α, and two inscribed circles:
1. One circle inscribed in the rhombus
2. Another circle inscribed in the curvilinear triangle formed by the rhombus and the first circle

This theorem states that the radius r of the second circle (inscribed in the curvilinear triangle)
is equal to (h/2) * tan²(45° - α/4)
-/
theorem radius_of_inscribed_circle_in_curvilinear_triangle 
  (h : ℝ) (α : ℝ) (h_pos : h > 0) (α_acute : 0 < α ∧ α < π/2) :
  ∃ r : ℝ, r = (h/2) * (Real.tan (π/4 - α/4))^2 ∧ 
  r > 0 ∧ 
  r < h/2 := by
sorry

end radius_of_inscribed_circle_in_curvilinear_triangle_l2649_264924


namespace omelet_time_is_100_l2649_264961

/-- Time to prepare and cook omelets -/
def total_omelet_time (
  pepper_chop_time : ℕ)
  (onion_chop_time : ℕ)
  (mushroom_slice_time : ℕ)
  (tomato_dice_time : ℕ)
  (cheese_grate_time : ℕ)
  (vegetable_saute_time : ℕ)
  (egg_cheese_cook_time : ℕ)
  (num_peppers : ℕ)
  (num_onions : ℕ)
  (num_mushrooms : ℕ)
  (num_tomatoes : ℕ)
  (num_omelets : ℕ) : ℕ :=
  let prep_time := 
    pepper_chop_time * num_peppers +
    onion_chop_time * num_onions +
    mushroom_slice_time * num_mushrooms +
    tomato_dice_time * num_tomatoes +
    cheese_grate_time * num_omelets
  let cook_time := vegetable_saute_time + egg_cheese_cook_time
  let omelets_during_prep := prep_time / cook_time
  let remaining_omelets := num_omelets - omelets_during_prep
  prep_time + remaining_omelets * cook_time

/-- Theorem: The total time to prepare and cook 10 omelets is 100 minutes -/
theorem omelet_time_is_100 :
  total_omelet_time 3 4 2 3 1 4 6 8 4 6 6 10 = 100 := by
  sorry

end omelet_time_is_100_l2649_264961


namespace coloring_book_shelves_l2649_264907

theorem coloring_book_shelves (initial_stock : ℝ) (acquired : ℝ) (books_per_shelf : ℝ) 
  (h1 : initial_stock = 40.0)
  (h2 : acquired = 20.0)
  (h3 : books_per_shelf = 4.0) :
  (initial_stock + acquired) / books_per_shelf = 15 := by
  sorry

end coloring_book_shelves_l2649_264907


namespace pyramid_equal_volume_division_l2649_264964

theorem pyramid_equal_volume_division (m : ℝ) (hm : m > 0) :
  ∃ (x y z : ℝ),
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    x + y + z = m ∧
    x^3 = (1/3) * m^3 ∧
    (x + y)^3 = (2/3) * m^3 ∧
    x = m / Real.rpow 3 (1/3) ∧
    y = (m / Real.rpow 3 (1/3)) * (Real.rpow 2 (1/3) - 1) ∧
    z = m * (1 - Real.rpow (2/3) (1/3)) :=
by sorry

end pyramid_equal_volume_division_l2649_264964


namespace cos_42_cos_18_minus_cos_48_sin_18_l2649_264942

theorem cos_42_cos_18_minus_cos_48_sin_18 :
  Real.cos (42 * π / 180) * Real.cos (18 * π / 180) - 
  Real.cos (48 * π / 180) * Real.sin (18 * π / 180) = 1/2 := by
  sorry

end cos_42_cos_18_minus_cos_48_sin_18_l2649_264942


namespace estevan_blankets_l2649_264993

theorem estevan_blankets (initial_blankets : ℕ) : 
  (initial_blankets / 3 : ℚ) + 2 = 10 → initial_blankets = 24 := by
  sorry

end estevan_blankets_l2649_264993


namespace prob_at_least_one_qualified_prob_merchant_rejects_l2649_264950

-- Define the probability of a product being qualified
def p_qualified : ℝ := 0.8

-- Define the number of products inspected by the company
def n_company_inspect : ℕ := 4

-- Define the total number of products sent to the merchant
def n_total : ℕ := 20

-- Define the number of unqualified products
def n_unqualified : ℕ := 3

-- Define the number of products inspected by the merchant
def n_merchant_inspect : ℕ := 2

-- Theorem for part I
theorem prob_at_least_one_qualified :
  1 - (1 - p_qualified) ^ n_company_inspect = 0.9984 := by sorry

-- Theorem for part II
theorem prob_merchant_rejects :
  (Nat.choose (n_total - n_unqualified) 1 * Nat.choose n_unqualified 1 +
   Nat.choose n_unqualified 2) / Nat.choose n_total 2 = 27 / 95 := by sorry

end prob_at_least_one_qualified_prob_merchant_rejects_l2649_264950
