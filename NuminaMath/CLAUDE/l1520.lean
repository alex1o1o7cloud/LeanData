import Mathlib

namespace train_speed_excluding_stoppages_l1520_152089

/-- Given a train that travels at 21 kmph including stoppages and stops for 18 minutes per hour,
    its speed excluding stoppages is 30 kmph. -/
theorem train_speed_excluding_stoppages
  (speed_with_stops : ℝ)
  (stop_time : ℝ)
  (h1 : speed_with_stops = 21)
  (h2 : stop_time = 18)
  : (speed_with_stops * 60) / (60 - stop_time) = 30 := by
  sorry

end train_speed_excluding_stoppages_l1520_152089


namespace f_positive_iff_triangle_l1520_152086

/-- A polynomial function representing the triangle inequality condition -/
def f (x y z : ℝ) : ℝ := (x + y + z) * (-x + y + z) * (x - y + z) * (x + y - z)

/-- Predicate to check if three real numbers can form the sides of a triangle -/
def is_triangle (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y > z ∧ x + z > y ∧ y + z > x

/-- Theorem stating that f is positive iff x, y, z can form a triangle -/
theorem f_positive_iff_triangle (x y z : ℝ) :
  f x y z > 0 ↔ is_triangle (|x|) (|y|) (|z|) := by sorry

end f_positive_iff_triangle_l1520_152086


namespace marks_reading_time_marks_reading_proof_l1520_152073

theorem marks_reading_time (increase : ℕ) (target : ℕ) (days_in_week : ℕ) : ℕ :=
  let initial_daily_hours : ℕ := (target - increase) / days_in_week
  initial_daily_hours

theorem marks_reading_proof :
  marks_reading_time 4 18 7 = 2 := by
  sorry

end marks_reading_time_marks_reading_proof_l1520_152073


namespace regular_150_sided_polygon_diagonals_l1520_152015

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 150 sides has 11025 diagonals -/
theorem regular_150_sided_polygon_diagonals :
  num_diagonals 150 = 11025 := by sorry

end regular_150_sided_polygon_diagonals_l1520_152015


namespace quadratic_solution_sum_l1520_152023

theorem quadratic_solution_sum (x y : ℝ) : 
  x + y = 5 → 2 * x * y = 5 → 
  ∃ (a b c d : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (x = (a + b * Real.sqrt c) / d ∨ x = (a - b * Real.sqrt c) / d) ∧
    a + b + c + d = 23 := by
  sorry

end quadratic_solution_sum_l1520_152023


namespace line_plane_relations_l1520_152010

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem line_plane_relations 
  (l m : Line) (α : Plane) 
  (h : perpendicular l α) : 
  (perpendicular m α → parallel_lines m l) ∧ 
  (parallel m α → perpendicular_lines m l) ∧ 
  (parallel_lines m l → perpendicular m α) := by
  sorry

end line_plane_relations_l1520_152010


namespace outfit_count_l1520_152050

/-- The number of outfits that can be made with different colored shirts and hats -/
def number_of_outfits : ℕ :=
  let red_shirts := 7
  let blue_shirts := 5
  let green_shirts := 8
  let pants := 10
  let green_hats := 10
  let red_hats := 6
  let blue_hats := 7
  (red_shirts * pants * (green_hats + blue_hats)) +
  (blue_shirts * pants * (green_hats + red_hats)) +
  (green_shirts * pants * (red_hats + blue_hats))

theorem outfit_count : number_of_outfits = 3030 := by
  sorry

end outfit_count_l1520_152050


namespace sqrt_five_fourth_power_l1520_152004

theorem sqrt_five_fourth_power : (Real.sqrt 5) ^ 4 = 25 := by
  sorry

end sqrt_five_fourth_power_l1520_152004


namespace wendy_polished_110_glasses_l1520_152093

/-- The number of small glasses Wendy polished -/
def small_glasses : ℕ := 50

/-- The additional number of large glasses compared to small glasses -/
def additional_large_glasses : ℕ := 10

/-- The total number of glasses Wendy polished -/
def total_glasses : ℕ := small_glasses + (small_glasses + additional_large_glasses)

/-- Proves that Wendy polished 110 glasses in total -/
theorem wendy_polished_110_glasses : total_glasses = 110 := by
  sorry

end wendy_polished_110_glasses_l1520_152093


namespace ernest_wire_problem_l1520_152012

theorem ernest_wire_problem (total_parts : ℕ) (used_parts : ℕ) (unused_length : ℝ) :
  total_parts = 5 ∧ used_parts = 3 ∧ unused_length = 20 →
  total_parts * (unused_length / (total_parts - used_parts)) = 50 := by
  sorry

end ernest_wire_problem_l1520_152012


namespace oranges_per_group_l1520_152092

/-- Given the total number of oranges and the number of orange groups,
    prove that the number of oranges per group is 2. -/
theorem oranges_per_group (total_oranges : ℕ) (orange_groups : ℕ) 
  (h1 : total_oranges = 356) (h2 : orange_groups = 178) :
  total_oranges / orange_groups = 2 := by
  sorry


end oranges_per_group_l1520_152092


namespace solve_for_r_l1520_152027

theorem solve_for_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = 2 := by
  sorry

end solve_for_r_l1520_152027


namespace angle_condition_implies_y_range_l1520_152011

/-- Given points A(-1,1) and B(3,y), and vector a = (1,2), if the angle between AB and a is acute, 
    then y ∈ (-1,9) ∪ (9,+∞). -/
theorem angle_condition_implies_y_range (y : ℝ) : 
  let A : ℝ × ℝ := (-1, 1)
  let B : ℝ × ℝ := (3, y)
  let a : ℝ × ℝ := (1, 2)
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  (AB.1 * a.1 + AB.2 * a.2 > 0) → -- Dot product > 0 implies acute angle
  (y ∈ Set.Ioo (-1 : ℝ) 9 ∪ Set.Ioi 9) := by
sorry

end angle_condition_implies_y_range_l1520_152011


namespace angle_around_point_l1520_152040

theorem angle_around_point (x : ℝ) : x = 110 :=
  let total_angle : ℝ := 360
  let given_angle : ℝ := 140
  have h1 : x + x + given_angle = total_angle := by sorry
  sorry

end angle_around_point_l1520_152040


namespace M_always_positive_l1520_152079

theorem M_always_positive (x y : ℝ) : 3 * x^2 - 8 * x * y + 9 * y^2 - 4 * x + 6 * y + 13 > 0 := by
  sorry

end M_always_positive_l1520_152079


namespace triangle_abc_properties_l1520_152066

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (A < π) →
  (B > 0) → (B < π) →
  (C > 0) → (C < π) →
  (A + B + C = π) →
  ((Real.sqrt 3 * a) / (1 + Real.cos A) = c / Real.sin C) →
  (a = Real.sqrt 3) →
  (c - b = (Real.sqrt 6 - Real.sqrt 2) / 2) →
  (A = π / 3 ∧ (1/2 * b * c * Real.sin A = (3 + Real.sqrt 3) / 4)) := by
  sorry

end triangle_abc_properties_l1520_152066


namespace pool_visitors_l1520_152046

theorem pool_visitors (women : ℕ) (women_students : ℕ) (men_more : ℕ) (men_nonstudents : ℕ) 
  (h1 : women = 1518)
  (h2 : women_students = 536)
  (h3 : men_more = 525)
  (h4 : men_nonstudents = 1257) :
  women_students + ((women + men_more) - men_nonstudents) = 1322 := by
  sorry

end pool_visitors_l1520_152046


namespace sum_of_zeros_infimum_l1520_152018

noncomputable section

def f (x : ℝ) : ℝ := if x ≥ 1 then Real.log x else 1 - x / 2

def F (m : ℝ) (x : ℝ) : ℝ := f (f x + 1) + m

theorem sum_of_zeros_infimum (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ F m x₁ = 0 ∧ F m x₂ = 0) →
  ∃ s : ℝ, s = 4 - 2 * Real.log 2 ∧ ∀ x₁ x₂ : ℝ, F m x₁ = 0 → F m x₂ = 0 → x₁ + x₂ ≥ s :=
sorry

end sum_of_zeros_infimum_l1520_152018


namespace equal_split_donation_l1520_152053

def total_donation : ℝ := 15000
def donation1 : ℝ := 3500
def donation2 : ℝ := 2750
def donation3 : ℝ := 3870
def donation4 : ℝ := 2475
def num_remaining_homes : ℕ := 4

theorem equal_split_donation :
  let donated_sum := donation1 + donation2 + donation3 + donation4
  let remaining := total_donation - donated_sum
  remaining / num_remaining_homes = 601.25 := by
sorry

end equal_split_donation_l1520_152053


namespace unique_prime_power_equation_l1520_152014

theorem unique_prime_power_equation :
  ∃! (p n : ℕ), Prime p ∧ n > 0 ∧ (1 + p)^n = 1 + p*n + n^p := by sorry

end unique_prime_power_equation_l1520_152014


namespace gum_pack_size_l1520_152056

theorem gum_pack_size (mint_gum orange_gum y : ℕ) : 
  mint_gum = 24 → 
  orange_gum = 36 → 
  (mint_gum - 2 * y) / orange_gum = mint_gum / (orange_gum + 4 * y) → 
  y = 3 := by
sorry

end gum_pack_size_l1520_152056


namespace set_intersection_theorem_l1520_152087

theorem set_intersection_theorem (x : ℝ) :
  { x : ℝ | x ≥ -2 } ∩ ({ x : ℝ | x > 0 }ᶜ) = { x : ℝ | -2 ≤ x ∧ x ≤ 0 } := by
  sorry

end set_intersection_theorem_l1520_152087


namespace rectangle_length_l1520_152028

/-- The length of a rectangle with width 4 cm and area equal to a square with side length 8 cm -/
theorem rectangle_length (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  square_side = 8 →
  rect_width = 4 →
  square_side * square_side = rect_width * rect_length →
  rect_length = 16 := by
  sorry

end rectangle_length_l1520_152028


namespace sufficient_not_necessary_condition_l1520_152055

theorem sufficient_not_necessary_condition : 
  (∃ x : ℝ, x^2 = 1 ∧ x ≠ -1) ∧ 
  (∀ x : ℝ, x = -1 → x^2 = 1) := by
  sorry

end sufficient_not_necessary_condition_l1520_152055


namespace park_short_trees_after_planting_l1520_152024

/-- The number of short trees in the park after planting -/
def total_short_trees (initial_short_trees newly_planted_short_trees : ℕ) : ℕ :=
  initial_short_trees + newly_planted_short_trees

/-- Theorem stating that the total number of short trees after planting is 98 -/
theorem park_short_trees_after_planting :
  total_short_trees 41 57 = 98 := by
  sorry


end park_short_trees_after_planting_l1520_152024


namespace combined_cost_price_l1520_152069

def usa_stock : ℝ := 100
def uk_stock : ℝ := 150
def germany_stock : ℝ := 200

def usa_discount : ℝ := 0.06
def uk_discount : ℝ := 0.10
def germany_discount : ℝ := 0.07

def usa_brokerage : ℝ := 0.015
def uk_brokerage : ℝ := 0.02
def germany_brokerage : ℝ := 0.025

def usa_transaction : ℝ := 5
def uk_transaction : ℝ := 3
def germany_transaction : ℝ := 2

def maintenance_charge : ℝ := 0.005
def taxation_rate : ℝ := 0.15

def usd_to_gbp : ℝ := 0.75
def usd_to_eur : ℝ := 0.85

theorem combined_cost_price :
  let usa_cost := (usa_stock * (1 - usa_discount) * (1 + usa_brokerage) + usa_transaction) * (1 + maintenance_charge)
  let uk_cost := (uk_stock * (1 - uk_discount) * (1 + uk_brokerage) + uk_transaction) * (1 + maintenance_charge) / usd_to_gbp
  let germany_cost := (germany_stock * (1 - germany_discount) * (1 + germany_brokerage) + germany_transaction) * (1 + maintenance_charge) / usd_to_eur
  let total_cost := usa_cost + uk_cost + germany_cost
  total_cost * (1 + taxation_rate) = 594.75 := by sorry

end combined_cost_price_l1520_152069


namespace lcm_factor_problem_l1520_152026

theorem lcm_factor_problem (A B : ℕ) (hcf lcm x : ℕ) : 
  A > 0 → B > 0 → 
  Nat.gcd A B = hcf →
  hcf = 20 →
  A = 280 →
  lcm = Nat.lcm A B →
  lcm = 20 * 13 * x →
  x = 14 := by
sorry

end lcm_factor_problem_l1520_152026


namespace triangle_height_l1520_152006

/-- Given a triangle with area 615 m² and a side of 123 m, 
    the perpendicular height to that side is 10 m. -/
theorem triangle_height (A : ℝ) (b : ℝ) (h : ℝ) 
  (area_eq : A = 615) 
  (base_eq : b = 123) 
  (area_formula : A = (1/2) * b * h) : h = 10 := by
  sorry

end triangle_height_l1520_152006


namespace terrell_weight_lifting_l1520_152022

/-- The number of times Terrell lifts the 20-pound weights -/
def original_lifts : ℕ := 12

/-- The weight of each dumbbell in the original set (in pounds) -/
def original_weight : ℕ := 20

/-- The weight of each dumbbell in the new set (in pounds) -/
def new_weight : ℕ := 10

/-- The number of dumbbells Terrell lifts each time -/
def num_dumbbells : ℕ := 2

/-- Calculates the total weight lifted -/
def total_weight (weight : ℕ) (lifts : ℕ) : ℕ :=
  num_dumbbells * weight * lifts

/-- The number of times Terrell needs to lift the new weights to achieve the same total weight -/
def required_lifts : ℕ := total_weight original_weight original_lifts / (num_dumbbells * new_weight)

theorem terrell_weight_lifting :
  required_lifts = 24 ∧
  total_weight new_weight required_lifts = total_weight original_weight original_lifts :=
by sorry

end terrell_weight_lifting_l1520_152022


namespace no_two_digit_reverse_sum_twice_square_l1520_152091

theorem no_two_digit_reverse_sum_twice_square : 
  ¬ ∃ (N : ℕ), 
    (10 ≤ N ∧ N ≤ 99) ∧ 
    ∃ (k : ℕ), 
      N + (10 * (N % 10) + N / 10) = 2 * k^2 := by
  sorry

end no_two_digit_reverse_sum_twice_square_l1520_152091


namespace unknown_number_divisor_l1520_152043

theorem unknown_number_divisor : ∃ x : ℕ, 
  x > 0 ∧ 
  100 % x = 16 ∧ 
  200 % x = 4 ∧ 
  ∀ y : ℕ, y > 0 → 100 % y = 16 → 200 % y = 4 → y ≤ x :=
sorry

end unknown_number_divisor_l1520_152043


namespace total_pears_picked_l1520_152070

theorem total_pears_picked (alyssa_pears nancy_pears : ℕ) 
  (h1 : alyssa_pears = 42) 
  (h2 : nancy_pears = 17) : 
  alyssa_pears + nancy_pears = 59 := by
  sorry

end total_pears_picked_l1520_152070


namespace binomial_seven_choose_two_l1520_152052

theorem binomial_seven_choose_two : (7 : ℕ).choose 2 = 21 := by
  sorry

end binomial_seven_choose_two_l1520_152052


namespace box_volume_conversion_l1520_152064

theorem box_volume_conversion (box_volume_cubic_feet : ℝ) :
  box_volume_cubic_feet = 216 →
  box_volume_cubic_feet / 27 = 8 :=
by sorry

end box_volume_conversion_l1520_152064


namespace one_mile_equals_600_rods_l1520_152042

/-- Conversion factor from miles to furlongs -/
def mile_to_furlong : ℚ := 12

/-- Conversion factor from furlongs to rods -/
def furlong_to_rod : ℚ := 50

/-- The number of rods in one mile -/
def rods_in_mile : ℚ := mile_to_furlong * furlong_to_rod

/-- Theorem stating that one mile is equal to 600 rods -/
theorem one_mile_equals_600_rods : rods_in_mile = 600 := by
  sorry

end one_mile_equals_600_rods_l1520_152042


namespace namjoon_used_seven_pencils_l1520_152076

/-- Represents the number of pencils each person has at different stages --/
structure PencilCount where
  initial : Nat
  after_taehyung_gives : Nat
  final : Nat

/-- The problem setup --/
def problem : PencilCount × PencilCount := 
  ({ initial := 10, after_taehyung_gives := 7, final := 6 },  -- Taehyung's pencils
   { initial := 10, after_taehyung_gives := 13, final := 6 }) -- Namjoon's pencils

/-- Calculates the number of pencils Namjoon used --/
def pencils_namjoon_used (p : PencilCount × PencilCount) : Nat :=
  p.2.after_taehyung_gives - p.2.final

/-- Theorem stating that Namjoon used 7 pencils --/
theorem namjoon_used_seven_pencils :
  pencils_namjoon_used problem = 7 := by
  sorry

end namjoon_used_seven_pencils_l1520_152076


namespace soccer_league_games_l1520_152094

/-- The number of games played in a league with a given number of teams and games per pair of teams. -/
def games_played (n : ℕ) (g : ℕ) : ℕ := n * (n - 1) * g / 2

/-- Theorem: In a league with 10 teams, where each team plays 4 games with each other team, 
    the total number of games played is 180. -/
theorem soccer_league_games : games_played 10 4 = 180 := by
  sorry

end soccer_league_games_l1520_152094


namespace knights_round_table_l1520_152030

theorem knights_round_table (n : ℕ) 
  (h1 : ∃ (f e : ℕ), f = e ∧ f + e = n) : 
  4 ∣ n := by
sorry

end knights_round_table_l1520_152030


namespace second_car_speed_l1520_152083

/-- Proves that given the conditions of the problem, the speed of the second car is 100 km/hr -/
theorem second_car_speed (car_a_speed : ℝ) (car_a_time : ℝ) (second_car_time : ℝ) (distance_ratio : ℝ) :
  car_a_speed = 50 →
  car_a_time = 6 →
  second_car_time = 1 →
  distance_ratio = 3 →
  (car_a_speed * car_a_time) / (distance_ratio * second_car_time) = 100 := by
  sorry

end second_car_speed_l1520_152083


namespace cheries_whistlers_l1520_152045

/-- Represents the number of boxes of fireworks --/
def koby_boxes : ℕ := 2

/-- Represents the number of boxes of fireworks --/
def cherie_boxes : ℕ := 1

/-- Represents the number of sparklers in each of Koby's boxes --/
def koby_sparklers_per_box : ℕ := 3

/-- Represents the number of whistlers in each of Koby's boxes --/
def koby_whistlers_per_box : ℕ := 5

/-- Represents the number of sparklers in Cherie's box --/
def cherie_sparklers : ℕ := 8

/-- Represents the total number of fireworks Koby and Cherie have --/
def total_fireworks : ℕ := 33

/-- Theorem stating that Cherie's box contains 9 whistlers --/
theorem cheries_whistlers :
  (koby_boxes * koby_sparklers_per_box + koby_boxes * koby_whistlers_per_box +
   cherie_sparklers + (total_fireworks - (koby_boxes * koby_sparklers_per_box +
   koby_boxes * koby_whistlers_per_box + cherie_sparklers))) = total_fireworks ∧
  (total_fireworks - (koby_boxes * koby_sparklers_per_box +
   koby_boxes * koby_whistlers_per_box + cherie_sparklers)) = 9 :=
by sorry

end cheries_whistlers_l1520_152045


namespace intersection_A_B_l1520_152095

-- Define the sets A and B
def A : Set ℝ := {x | x ≠ 3 ∧ x ≥ 2}
def B : Set ℝ := {x | 3 ≤ x ∧ x ≤ 5}

-- Define the interval (3,5]
def interval_3_5 : Set ℝ := {x | 3 < x ∧ x ≤ 5}

-- Theorem statement
theorem intersection_A_B : A ∩ B = interval_3_5 := by
  sorry

end intersection_A_B_l1520_152095


namespace root_difference_is_one_l1520_152084

theorem root_difference_is_one (p : ℝ) : 
  let α := (p + 1) / 2
  let β := (p - 1) / 2
  α - β = 1 ∧ 
  α^2 - p*α + (p^2 - 1)/4 = 0 ∧ 
  β^2 - p*β + (p^2 - 1)/4 = 0 ∧
  α ≥ β := by
sorry

end root_difference_is_one_l1520_152084


namespace min_value_z_plus_inv_z_squared_l1520_152000

/-- Given a complex number z with positive real part, and a parallelogram formed by the points 0, z, 1/z, and z + 1/z with an area of 12/13, the minimum value of |z + 1/z|² is 16/13. -/
theorem min_value_z_plus_inv_z_squared (z : ℂ) (h_real_pos : 0 < z.re) 
  (h_area : abs (z.im * (1/z).re - z.re * (1/z).im) = 12/13) :
  ∃ d : ℝ, d^2 = 16/13 ∧ ∀ w : ℂ, w.re > 0 → 
    abs (w.im * (1/w).re - w.re * (1/w).im) = 12/13 → 
    d^2 ≤ Complex.normSq (w + 1/w) := by
  sorry

end min_value_z_plus_inv_z_squared_l1520_152000


namespace symmetry_axis_of_sin_cos_function_l1520_152062

theorem symmetry_axis_of_sin_cos_function :
  ∃ (x : ℝ), x = π / 12 ∧
  ∀ (y : ℝ), y = Real.sin (2 * x) - Real.sqrt 3 * Real.cos (2 * x) →
  (∀ (t : ℝ), y = Real.sin (2 * (x + t)) - Real.sqrt 3 * Real.cos (2 * (x + t)) ↔
               y = Real.sin (2 * (x - t)) - Real.sqrt 3 * Real.cos (2 * (x - t))) :=
sorry

end symmetry_axis_of_sin_cos_function_l1520_152062


namespace min_value_sin_function_l1520_152067

theorem min_value_sin_function (f : ℝ → ℝ) (h : ∀ x, f x = Real.sin (2 * x - π / 4)) :
  ∃ x ∈ Set.Icc 0 (π / 2), ∀ y ∈ Set.Icc 0 (π / 2), f x ≤ f y ∧ f x = -Real.sqrt 2 / 2 := by
  sorry

end min_value_sin_function_l1520_152067


namespace book_price_change_l1520_152017

theorem book_price_change (P : ℝ) (h : P > 0) :
  let price_after_decrease : ℝ := P * 0.5
  let final_price : ℝ := P * 1.2
  ∃ x : ℝ, price_after_decrease * (1 + x / 100) = final_price ∧ x = 140 :=
by sorry

end book_price_change_l1520_152017


namespace chocolate_eggs_weight_l1520_152034

/-- Calculates the total weight of remaining chocolate eggs after discarding one box -/
theorem chocolate_eggs_weight (total_eggs : ℕ) (weight_per_egg : ℕ) (num_boxes : ℕ) :
  total_eggs = 12 →
  weight_per_egg = 10 →
  num_boxes = 4 →
  (total_eggs - (total_eggs / num_boxes)) * weight_per_egg = 90 := by
  sorry

#check chocolate_eggs_weight

end chocolate_eggs_weight_l1520_152034


namespace bonus_distribution_solution_l1520_152029

/-- Represents the bonus distribution problem -/
def BonusDistribution (total : ℚ) (ac_sum : ℚ) (common_ratio : ℚ) (d_bonus : ℚ) : Prop :=
  let a := d_bonus / (common_ratio^3)
  let b := d_bonus / (common_ratio^2)
  let c := d_bonus / common_ratio
  (a + b + c + d_bonus = total) ∧ 
  (a + c = ac_sum) ∧
  (0 < common_ratio) ∧ 
  (common_ratio < 1)

/-- The theorem stating the correct solution to the bonus distribution problem -/
theorem bonus_distribution_solution :
  BonusDistribution 68780 36200 (9/10) 14580 := by
  sorry

#check bonus_distribution_solution

end bonus_distribution_solution_l1520_152029


namespace inequality_equivalence_l1520_152008

theorem inequality_equivalence (x : ℝ) :
  (7 * x - 2 < 3 * (x + 2) ↔ x < 2) ∧
  ((x - 1) / 3 ≥ (x - 3) / 12 + 1 ↔ x ≥ 13 / 3) := by
  sorry

end inequality_equivalence_l1520_152008


namespace tangent_line_at_origin_l1520_152051

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - x - 1) * Real.exp (a * x)

theorem tangent_line_at_origin (a : ℝ) (h : a ≠ 0) :
  let tangent_line (x : ℝ) := -x - 1
  ∀ x, tangent_line x = f a 0 + (deriv (f a)) 0 * x :=
by sorry

end tangent_line_at_origin_l1520_152051


namespace max_wednesday_pizzas_exists_five_pizzas_wednesday_l1520_152085

/-- Represents the number of pizzas baked on each day -/
structure PizzaSchedule where
  saturday : ℕ
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Checks if the pizza schedule satisfies the given conditions -/
def isValidSchedule (schedule : PizzaSchedule) : Prop :=
  let total := 50
  schedule.saturday = (3 * total) / 5 ∧
  schedule.sunday = (3 * (total - schedule.saturday)) / 5 ∧
  schedule.monday < schedule.sunday ∧
  schedule.tuesday < schedule.monday ∧
  schedule.wednesday < schedule.tuesday ∧
  schedule.saturday + schedule.sunday + schedule.monday + schedule.tuesday + schedule.wednesday = total

/-- Theorem stating the maximum number of pizzas that could be baked on Wednesday -/
theorem max_wednesday_pizzas :
  ∀ (schedule : PizzaSchedule), isValidSchedule schedule → schedule.wednesday ≤ 5 := by
  sorry

/-- Theorem stating that there exists a valid schedule with 5 pizzas on Wednesday -/
theorem exists_five_pizzas_wednesday :
  ∃ (schedule : PizzaSchedule), isValidSchedule schedule ∧ schedule.wednesday = 5 := by
  sorry

end max_wednesday_pizzas_exists_five_pizzas_wednesday_l1520_152085


namespace enrique_commission_is_300_l1520_152060

-- Define the commission rate
def commission_rate : Real := 0.15

-- Define the sales data
def suits_sold : Nat := 2
def suit_price : Real := 700.00
def shirts_sold : Nat := 6
def shirt_price : Real := 50.00
def loafers_sold : Nat := 2
def loafer_price : Real := 150.00

-- Calculate total sales
def total_sales : Real :=
  (suits_sold : Real) * suit_price +
  (shirts_sold : Real) * shirt_price +
  (loafers_sold : Real) * loafer_price

-- Calculate Enrique's commission
def enrique_commission : Real := commission_rate * total_sales

-- Theorem to prove
theorem enrique_commission_is_300 :
  enrique_commission = 300.00 := by sorry

end enrique_commission_is_300_l1520_152060


namespace max_intersections_three_circles_two_lines_l1520_152059

/-- The maximum number of intersection points between circles -/
def max_circle_intersections (n : ℕ) : ℕ := n * (n - 1)

/-- The maximum number of intersection points between circles and lines -/
def max_circle_line_intersections (circles : ℕ) (lines : ℕ) : ℕ :=
  circles * lines * 2

/-- The maximum number of intersection points between lines -/
def max_line_intersections (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The total maximum number of intersection points -/
def total_max_intersections (circles : ℕ) (lines : ℕ) : ℕ :=
  max_circle_intersections circles +
  max_circle_line_intersections circles lines +
  max_line_intersections lines

theorem max_intersections_three_circles_two_lines :
  total_max_intersections 3 2 = 19 := by sorry

end max_intersections_three_circles_two_lines_l1520_152059


namespace folded_rectangle_long_side_l1520_152088

/-- A rectangle with a specific folding property -/
structure FoldedRectangle where
  short_side : ℝ
  long_side : ℝ
  folded_congruent : Bool

/-- The theorem stating the relationship between short and long sides in the folded rectangle -/
theorem folded_rectangle_long_side 
  (rect : FoldedRectangle) 
  (h1 : rect.short_side = 8) 
  (h2 : rect.folded_congruent = true) : 
  rect.long_side = 12 := by
  sorry

#check folded_rectangle_long_side

end folded_rectangle_long_side_l1520_152088


namespace joan_seashells_l1520_152019

def seashell_problem (initial found : ℕ) (given_away : ℕ) (additional_found : ℕ) (traded : ℕ) (lost : ℕ) : ℕ :=
  initial - given_away + additional_found - traded - lost

theorem joan_seashells :
  seashell_problem 79 63 45 20 5 = 36 := by
  sorry

end joan_seashells_l1520_152019


namespace quadratic_inequality_condition_l1520_152082

theorem quadratic_inequality_condition (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + 2 * x + 1 < 0) → a < 1 := by
  sorry

end quadratic_inequality_condition_l1520_152082


namespace cyclic_sum_inequality_l1520_152007

theorem cyclic_sum_inequality (x y z : ℝ) (a b : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 1) :
  (∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 1 → 
    x*y + y*z + z*x ≥ a*(y^2*z^2 + z^2*x^2 + x^2*y^2) + b*x*y*z) ↔ 
  (b = 9 - a ∧ 0 < a ∧ a ≤ 4) :=
sorry

end cyclic_sum_inequality_l1520_152007


namespace league_games_count_l1520_152065

/-- The number of teams in each division -/
def teams_per_division : ℕ := 9

/-- The number of times each team plays other teams in its own division -/
def intra_division_games : ℕ := 3

/-- The number of times each team plays teams in the other division -/
def inter_division_games : ℕ := 2

/-- The number of divisions in the league -/
def num_divisions : ℕ := 2

/-- The total number of games scheduled in the league -/
def total_games : ℕ :=
  (num_divisions * (teams_per_division.choose 2 * intra_division_games)) +
  (teams_per_division * teams_per_division * inter_division_games)

theorem league_games_count : total_games = 378 := by
  sorry

end league_games_count_l1520_152065


namespace book_categorization_l1520_152049

/-- Proves that given 800 books initially divided into 4 equal categories, 
    then each category divided into 5 groups, the number of final categories 
    when each group is further divided into categories of 20 books each is 40. -/
theorem book_categorization (total_books : Nat) (initial_categories : Nat) 
    (groups_per_category : Nat) (books_per_final_category : Nat) 
    (h1 : total_books = 800)
    (h2 : initial_categories = 4)
    (h3 : groups_per_category = 5)
    (h4 : books_per_final_category = 20) : 
    (total_books / initial_categories / groups_per_category / books_per_final_category) * 
    (initial_categories * groups_per_category) = 40 := by
  sorry

#check book_categorization

end book_categorization_l1520_152049


namespace solve_system_l1520_152002

theorem solve_system (c d : ℝ) 
  (eq1 : 5 + c = 7 - d) 
  (eq2 : 6 + d = 10 + c) : 
  5 - c = 6 := by
sorry

end solve_system_l1520_152002


namespace arithmetic_sequence_general_term_l1520_152077

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  geometric_subsequence : (a 3) ^ 2 = a 2 * a 7
  initial_condition : 2 * a 1 + a 2 = 1

/-- The general term of the arithmetic sequence is 5/3 - n -/
theorem arithmetic_sequence_general_term (seq : ArithmeticSequence) :
  ∀ n, seq.a n = 5/3 - n := by
  sorry

end arithmetic_sequence_general_term_l1520_152077


namespace twelve_students_pairs_l1520_152044

/-- The number of unique pairs in a group of n elements -/
def uniquePairs (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of unique pairs in a group of 12 students is 66 -/
theorem twelve_students_pairs : uniquePairs 12 = 66 := by
  sorry

end twelve_students_pairs_l1520_152044


namespace problem_solution_l1520_152075

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * (a + 1) * x^2 + 6 * a * x

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 6 * x^2 - 6 * (a + 1) * x + 6 * a

-- Define M(a) and m(a)
def M (a : ℝ) : ℝ := max (f a 1) (f a 2)
def m (a : ℝ) : ℝ := min (f a 1) (f a 2)

-- Define h(a)
def h (a : ℝ) : ℝ := M a - m a

theorem problem_solution :
  (∀ a : ℝ, f' a 0 = 3 → a = 1/2) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → f a x + f a (-x) ≥ 12 * Real.log x) →
    a ≤ -1 - Real.exp (-1)) ∧
  (∀ a : ℝ, a > 1 →
    (∃ min_h : ℝ, min_h = 8/27 ∧
      ∀ a' : ℝ, a' > 1 → h a' ≥ min_h)) :=
by sorry

end problem_solution_l1520_152075


namespace advanced_purchase_tickets_l1520_152048

/-- Proves that the number of advanced-purchase tickets sold is 40 --/
theorem advanced_purchase_tickets (total_tickets : ℕ) (total_amount : ℕ) 
  (advanced_price : ℕ) (door_price : ℕ) (h1 : total_tickets = 140) 
  (h2 : total_amount = 1720) (h3 : advanced_price = 8) (h4 : door_price = 14) :
  ∃ (advanced_tickets : ℕ) (door_tickets : ℕ),
    advanced_tickets + door_tickets = total_tickets ∧
    advanced_price * advanced_tickets + door_price * door_tickets = total_amount ∧
    advanced_tickets = 40 := by
  sorry

end advanced_purchase_tickets_l1520_152048


namespace min_mn_value_l1520_152096

def f (x a : ℝ) : ℝ := |x - a|

theorem min_mn_value (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, f x 1 ≤ 1 ↔ x ∈ Set.Icc 0 2) →
  1 / m + 1 / (2 * n) = 1 →
  ∀ k, m * n ≥ k → k = 2 :=
sorry

end min_mn_value_l1520_152096


namespace lisa_additional_marbles_l1520_152099

/-- The minimum number of additional marbles Lisa needs -/
def minimum_additional_marbles (num_friends : ℕ) (current_marbles : ℕ) : ℕ :=
  let min_marbles_per_friend := 3
  let max_marbles_per_friend := min_marbles_per_friend + num_friends - 1
  let total_marbles_needed := num_friends * (min_marbles_per_friend + max_marbles_per_friend) / 2
  max (total_marbles_needed - current_marbles) 0

/-- Theorem stating the minimum number of additional marbles Lisa needs -/
theorem lisa_additional_marbles :
  minimum_additional_marbles 12 50 = 52 := by
  sorry

#eval minimum_additional_marbles 12 50

end lisa_additional_marbles_l1520_152099


namespace watson_second_graders_l1520_152036

/-- Represents the number of students in each grade and the total in Ms. Watson's class -/
structure ClassComposition where
  total : Nat
  kindergartners : Nat
  firstGraders : Nat
  thirdGraders : Nat
  absentStudents : Nat

/-- Calculates the number of second graders in the class -/
def secondGraders (c : ClassComposition) : Nat :=
  c.total - (c.kindergartners + c.firstGraders + c.thirdGraders + c.absentStudents)

/-- Theorem stating the number of second graders in Ms. Watson's class -/
theorem watson_second_graders :
  let c : ClassComposition := {
    total := 120,
    kindergartners := 34,
    firstGraders := 48,
    thirdGraders := 5,
    absentStudents := 6
  }
  secondGraders c = 27 := by sorry

end watson_second_graders_l1520_152036


namespace gcd_120_168_l1520_152047

theorem gcd_120_168 : Nat.gcd 120 168 = 24 := by
  sorry

end gcd_120_168_l1520_152047


namespace mia_correctness_rate_l1520_152037

/-- Represents the correctness rate of homework problems -/
structure HomeworkStats where
  individual_ratio : ℚ  -- Ratio of problems solved individually
  together_ratio : ℚ    -- Ratio of problems solved together
  liam_individual_correct : ℚ  -- Liam's correctness rate for individual problems
  liam_total_correct : ℚ       -- Liam's total correctness rate
  mia_individual_correct : ℚ   -- Mia's correctness rate for individual problems

/-- Calculates Mia's overall percentage of correct answers -/
def mia_overall_correct (stats : HomeworkStats) : ℚ :=
  stats.individual_ratio * stats.mia_individual_correct + 
  stats.together_ratio * ((stats.liam_total_correct - stats.individual_ratio * stats.liam_individual_correct) / stats.together_ratio)

/-- Theorem stating Mia's overall correctness rate given the problem conditions -/
theorem mia_correctness_rate (stats : HomeworkStats) 
  (h1 : stats.individual_ratio = 2/3)
  (h2 : stats.together_ratio = 1/3)
  (h3 : stats.liam_individual_correct = 70/100)
  (h4 : stats.liam_total_correct = 82/100)
  (h5 : stats.mia_individual_correct = 85/100) :
  mia_overall_correct stats = 92/100 := by
  sorry  -- Proof omitted

#eval mia_overall_correct {
  individual_ratio := 2/3,
  together_ratio := 1/3,
  liam_individual_correct := 70/100,
  liam_total_correct := 82/100,
  mia_individual_correct := 85/100
}

end mia_correctness_rate_l1520_152037


namespace A_not_perfect_square_l1520_152009

/-- A number formed by 600 times the digit 6 followed by any number of zeros -/
def A (n : ℕ) : ℕ := 666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666666 * (10^n)

/-- The 2-adic valuation of a natural number -/
def two_adic_valuation (m : ℕ) : ℕ :=
  if m = 0 then 0 else (m.factors.filter (· = 2)).length

/-- Theorem: A is not a perfect square for any number of trailing zeros -/
theorem A_not_perfect_square (n : ℕ) : ¬ ∃ (k : ℕ), A n = k^2 := by
  sorry

end A_not_perfect_square_l1520_152009


namespace modular_home_cost_l1520_152039

-- Define the parameters of the modular home
def total_area : ℝ := 3500
def kitchen_area : ℝ := 500
def kitchen_cost : ℝ := 35000
def bathroom_area : ℝ := 250
def bathroom_cost : ℝ := 15000
def bedroom_area : ℝ := 350
def bedroom_cost : ℝ := 21000
def living_area : ℝ := 600
def living_area_cost_per_sqft : ℝ := 100
def upgraded_cost_per_sqft : ℝ := 150

def num_kitchens : ℕ := 1
def num_bathrooms : ℕ := 3
def num_bedrooms : ℕ := 4
def num_living_areas : ℕ := 1

-- Define the theorem
theorem modular_home_cost :
  let total_module_area := kitchen_area * num_kitchens + bathroom_area * num_bathrooms +
                           bedroom_area * num_bedrooms + living_area * num_living_areas
  let remaining_area := total_area - total_module_area
  let upgraded_area := remaining_area / 2
  let total_cost := kitchen_cost * num_kitchens + bathroom_cost * num_bathrooms +
                    bedroom_cost * num_bedrooms + living_area * living_area_cost_per_sqft +
                    upgraded_area * upgraded_cost_per_sqft * 2
  total_cost = 261500 := by sorry

end modular_home_cost_l1520_152039


namespace simplify_expression_l1520_152054

theorem simplify_expression : (45 * 2^10) / (15 * 2^5) * 5 = 480 := by
  sorry

end simplify_expression_l1520_152054


namespace square_inequality_for_negatives_l1520_152098

theorem square_inequality_for_negatives (a b : ℝ) (h : a < b ∧ b < 0) : a^2 > b^2 := by
  sorry

end square_inequality_for_negatives_l1520_152098


namespace all_judgments_correct_l1520_152057

theorem all_judgments_correct (a b c : ℕ) (ha : a = 2^22) (hb : b = 3^11) (hc : c = 12^9) :
  (a > b) ∧ (a * b > c) ∧ (b < c) := by
  sorry

end all_judgments_correct_l1520_152057


namespace kirsten_stole_14_meatballs_l1520_152003

/-- The number of meatballs Kirsten stole -/
def meatballs_stolen (initial final : ℕ) : ℕ := initial - final

/-- Proof that Kirsten stole 14 meatballs -/
theorem kirsten_stole_14_meatballs (initial final : ℕ) 
  (h_initial : initial = 25)
  (h_final : final = 11) : 
  meatballs_stolen initial final = 14 := by
  sorry

end kirsten_stole_14_meatballs_l1520_152003


namespace solution_in_quadrant_I_l1520_152025

theorem solution_in_quadrant_I (k : ℝ) :
  (∃ x y : ℝ, 3 * x - 4 * y = 6 ∧ k * x + 2 * y = 8 ∧ x > 0 ∧ y > 0) ↔ -3/2 < k ∧ k < 4 := by
  sorry

end solution_in_quadrant_I_l1520_152025


namespace emma_age_l1520_152074

/-- Represents the ages of the individuals --/
structure Ages where
  oliver : ℕ
  nancy : ℕ
  liam : ℕ
  emma : ℕ

/-- The age relationships between Oliver, Nancy, Liam, and Emma --/
def age_relationships (ages : Ages) : Prop :=
  ages.oliver + 5 = ages.nancy ∧
  ages.nancy = ages.liam + 6 ∧
  ages.emma = ages.liam + 4 ∧
  ages.oliver = 16

/-- Theorem stating that given the age relationships and Oliver's age, Emma is 19 years old --/
theorem emma_age (ages : Ages) : age_relationships ages → ages.emma = 19 := by
  sorry

end emma_age_l1520_152074


namespace sum_of_ninth_powers_l1520_152061

theorem sum_of_ninth_powers (a b : ℝ) 
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^9 + b^9 = 76 := by
  sorry

end sum_of_ninth_powers_l1520_152061


namespace c_grazing_months_l1520_152021

/-- Represents the number of oxen-months for each person -/
def oxen_months (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

/-- Represents the total rent of the pasture -/
def total_rent : ℕ := 175

/-- Represents c's share of the rent -/
def c_share : ℕ := 45

/-- Theorem stating that c put his oxen for grazing for 3 months -/
theorem c_grazing_months :
  ∃ (x : ℕ),
    x = 3 ∧
    c_share * (oxen_months 10 7 + oxen_months 12 5 + oxen_months 15 x) =
    total_rent * oxen_months 15 x :=
by sorry

end c_grazing_months_l1520_152021


namespace minimum_researchers_l1520_152005

theorem minimum_researchers (genetics : ℕ) (microbiology : ℕ) (both : ℕ)
  (h1 : genetics = 120)
  (h2 : microbiology = 90)
  (h3 : both = 40) :
  genetics + microbiology - both = 170 := by
  sorry

end minimum_researchers_l1520_152005


namespace isosceles_triangle_vertex_angle_sine_l1520_152001

/-- An isosceles triangle with a base angle tangent of 2/3 has a vertex angle sine of 12/13 -/
theorem isosceles_triangle_vertex_angle_sine (α β : Real) : 
  -- α is a base angle of the isosceles triangle
  -- β is the vertex angle of the isosceles triangle
  -- The triangle is isosceles
  β = π - 2 * α →
  -- The tangent of the base angle is 2/3
  Real.tan α = 2 / 3 →
  -- The sine of the vertex angle is 12/13
  Real.sin β = 12 / 13 := by
  sorry

end isosceles_triangle_vertex_angle_sine_l1520_152001


namespace parabola_point_focus_distance_l1520_152035

/-- Theorem: Distance between a point on a parabola and its focus
Given a parabola y^2 = 16x with focus F at (4, 0), and a point P on the parabola
that is 12 units away from the x-axis, the distance between P and F is 13 units. -/
theorem parabola_point_focus_distance
  (P : ℝ × ℝ) -- Point P on the parabola
  (h_on_parabola : (P.2)^2 = 16 * P.1) -- P satisfies the parabola equation
  (h_distance_from_x_axis : abs P.2 = 12) -- P is 12 units from x-axis
  : Real.sqrt ((P.1 - 4)^2 + P.2^2) = 13 := by
  sorry

end parabola_point_focus_distance_l1520_152035


namespace arithmetic_calculations_l1520_152013

theorem arithmetic_calculations : 
  (-6 * (-2) + (-5) * 16 = -68) ∧ 
  ((-1)^4 + (1/4) * (2 * (-6) - (-4)^2) = -8) := by
  sorry

end arithmetic_calculations_l1520_152013


namespace negation_of_implication_l1520_152071

theorem negation_of_implication (x : ℝ) :
  ¬(x > 1 → x^2 > 1) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by sorry

end negation_of_implication_l1520_152071


namespace wickets_before_last_match_is_175_l1520_152038

/-- Represents a bowler's statistics -/
structure BowlerStats where
  initialAverage : ℝ
  lastMatchWickets : ℕ
  lastMatchRuns : ℕ
  averageDecrease : ℝ

/-- Calculates the number of wickets taken before the last match -/
def wicketsBeforeLastMatch (stats : BowlerStats) : ℕ :=
  sorry

/-- Theorem stating that for the given conditions, the number of wickets before the last match is 175 -/
theorem wickets_before_last_match_is_175 (stats : BowlerStats) 
  (h1 : stats.initialAverage = 12.4)
  (h2 : stats.lastMatchWickets = 8)
  (h3 : stats.lastMatchRuns = 26)
  (h4 : stats.averageDecrease = 0.4) :
  wicketsBeforeLastMatch stats = 175 := by
  sorry

end wickets_before_last_match_is_175_l1520_152038


namespace polynomial_equality_l1520_152063

theorem polynomial_equality (x : ℝ) : 
  (x - 2)^5 + 5*(x - 2)^4 + 10*(x - 2)^3 + 10*(x - 2)^2 + 5*(x - 2) + 1 = (x - 1)^5 := by
  sorry

end polynomial_equality_l1520_152063


namespace product_evaluation_l1520_152058

theorem product_evaluation (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((x^2 + y^2 + z^2)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) * (x^2*y^2 + y^2*z^2 + z^2*x^2)⁻¹ * ((x*y)⁻¹ + (y*z)⁻¹ + (z*x)⁻¹)) =
  ((x*y + y*z + z*x) * (x + y + z)) / (x^2 * y^2 * z^2 * (x^2 + y^2 + z^2)) :=
by sorry

end product_evaluation_l1520_152058


namespace dance_steps_total_time_l1520_152097

/-- The time spent learning seven dance steps -/
def dance_steps_time : ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ → ℝ :=
  fun step1 step2 step3 step4 step5 step6 step7 =>
    step1 + step2 + step3 + step4 + step5 + step6 + step7

theorem dance_steps_total_time :
  ∀ (step1 : ℝ),
    step1 = 50 →
    let step2 := step1 / 3
    let step3 := step1 + step2
    let step4 := 1.75 * step1
    let step5 := step2 + 25
    let step6 := step3 + step5 - 40
    let step7 := step1 + step2 + step4 + 10
    ∃ (ε : ℝ), ε > 0 ∧ 
      |dance_steps_time step1 step2 step3 step4 step5 step6 step7 - 495.02| < ε :=
by
  sorry


end dance_steps_total_time_l1520_152097


namespace remainder_nineteen_power_nineteen_plus_nineteen_mod_twenty_l1520_152068

theorem remainder_nineteen_power_nineteen_plus_nineteen_mod_twenty :
  (19^19 + 19) % 20 = 18 := by
  sorry

end remainder_nineteen_power_nineteen_plus_nineteen_mod_twenty_l1520_152068


namespace palindrome_divisible_by_11_sum_divisible_by_11_divisible_by_11_condition_balanced_sum_of_palindromes_is_palindrome_l1520_152031

/-- A four-digit number is balanced if the sum of its first two digits equals the sum of its last two digits. -/
def IsBalanced (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 + (n / 100 % 10) = (n / 10 % 10) + n % 10)

/-- A four-digit number is a palindrome if its first digit equals its last digit and its second digit equals its third digit. -/
def IsPalindrome (n : ℕ) : Prop :=
  n ≥ 1000 ∧ n < 10000 ∧
  (n / 1000 = n % 10) ∧ ((n / 100) % 10 = (n / 10) % 10)

/-- Any four-digit palindrome is divisible by 11. -/
theorem palindrome_divisible_by_11 (n : ℕ) (h : IsPalindrome n) : 11 ∣ n :=
  sorry

/-- The sum of two numbers divisible by 11 is divisible by 11. -/
theorem sum_divisible_by_11 (a b : ℕ) (ha : 11 ∣ a) (hb : 11 ∣ b) : 11 ∣ (a + b) :=
  sorry

/-- A four-digit number divisible by 11 satisfies the condition that twice its first digit minus its last digit is congruent to 0 modulo 11. -/
theorem divisible_by_11_condition (n : ℕ) (h : 11 ∣ n) (h_four_digit : n ≥ 1000 ∧ n < 10000) :
  (2 * (n / 1000) - n % 10) % 11 = 0 :=
  sorry

/-- Main theorem: A four-digit balanced number that is the sum of two palindrome numbers must itself be a palindrome. -/
theorem balanced_sum_of_palindromes_is_palindrome (n : ℕ) 
  (h_balanced : IsBalanced n) 
  (h_sum_of_palindromes : ∃ a b : ℕ, IsPalindrome a ∧ IsPalindrome b ∧ n = a + b) :
  IsPalindrome n :=
  sorry

end palindrome_divisible_by_11_sum_divisible_by_11_divisible_by_11_condition_balanced_sum_of_palindromes_is_palindrome_l1520_152031


namespace arithmetic_sequence_sum_l1520_152041

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_sum1 : a 1 + a 4 + a 7 = 45)
  (h_sum2 : a 2 + a 5 + a 8 = 39) :
  a 3 + a 6 + a 9 = 33 :=
sorry

end arithmetic_sequence_sum_l1520_152041


namespace identity_is_unique_divisibility_function_l1520_152078

/-- A function f: ℕ → ℕ satisfying the divisibility condition -/
def DivisibilityFunction (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, (m^2 + n)^2 % (f m^2 + f n) = 0

/-- The theorem stating that the identity function is the only function satisfying the divisibility condition -/
theorem identity_is_unique_divisibility_function :
  ∀ f : ℕ → ℕ, DivisibilityFunction f ↔ ∀ n : ℕ, f n = n :=
sorry

end identity_is_unique_divisibility_function_l1520_152078


namespace total_evaluations_is_2680_l1520_152090

/-- Represents a class with its exam components and student count -/
structure ExamClass where
  students : ℕ
  multipleChoice : ℕ
  shortAnswer : ℕ
  essay : ℕ
  otherEvaluations : ℕ

/-- Calculates the total evaluations for a single class -/
def classEvaluations (c : ExamClass) : ℕ :=
  c.students * (c.multipleChoice + c.shortAnswer + c.essay) + c.otherEvaluations

/-- The exam classes as defined in the problem -/
def examClasses : List ExamClass := [
  ⟨30, 12, 0, 3, 30⟩,  -- Class A
  ⟨25, 15, 5, 2, 5⟩,   -- Class B
  ⟨35, 10, 0, 3, 5⟩,   -- Class C
  ⟨40, 11, 4, 3, 40⟩,  -- Class D
  ⟨20, 14, 5, 2, 5⟩    -- Class E
]

/-- The theorem stating that the total evaluations equal 2680 -/
theorem total_evaluations_is_2680 :
  (examClasses.map classEvaluations).sum = 2680 := by
  sorry

end total_evaluations_is_2680_l1520_152090


namespace circular_track_time_theorem_l1520_152080

/-- Represents a circular track with two points -/
structure CircularTrack :=
  (total_time : ℝ)
  (time_closer_to_point : ℝ)

/-- Theorem: If a runner on a circular track is closer to one point for half the total running time,
    then the total running time is twice the time the runner is closer to that point -/
theorem circular_track_time_theorem (track : CircularTrack) 
  (h1 : track.time_closer_to_point > 0)
  (h2 : track.time_closer_to_point = track.total_time / 2) : 
  track.total_time = 2 * track.time_closer_to_point :=
sorry

end circular_track_time_theorem_l1520_152080


namespace constant_expression_value_l1520_152020

-- Define the triangle DEF
structure Triangle where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

-- Define the properties of the triangle
def Triangle.sideLengths (t : Triangle) : ℝ × ℝ × ℝ := sorry
def Triangle.circumradius (t : Triangle) : ℝ := sorry
def Triangle.orthocenter (t : Triangle) : ℝ × ℝ := sorry
def Triangle.circumcircle (t : Triangle) : Set (ℝ × ℝ) := sorry

-- Define the constant expression
def constantExpression (t : Triangle) (Q : ℝ × ℝ) : ℝ :=
  let (d, e, f) := t.sideLengths
  let H := t.orthocenter
  let S := t.circumradius
  (Q.1 - t.D.1)^2 + (Q.2 - t.D.2)^2 +
  (Q.1 - t.E.1)^2 + (Q.2 - t.E.2)^2 +
  (Q.1 - t.F.1)^2 + (Q.2 - t.F.2)^2 -
  ((Q.1 - H.1)^2 + (Q.2 - H.2)^2)

-- State the theorem
theorem constant_expression_value (t : Triangle) :
  ∀ Q ∈ t.circumcircle, constantExpression t Q = 
    let (d, e, f) := t.sideLengths
    let S := t.circumradius
    d^2 + e^2 + f^2 - 4 * S^2 :=
sorry

end constant_expression_value_l1520_152020


namespace parabolas_common_tangent_l1520_152032

/-- Given two parabolas C₁ and C₂, prove that if they have exactly one common tangent,
    then a = -1/2 and the equation of the common tangent is y = x - 1/4 -/
theorem parabolas_common_tangent (a : ℝ) :
  let C₁ := λ x : ℝ => x^2 + 2*x
  let C₂ := λ x : ℝ => -x^2 + a
  (∃! l : ℝ → ℝ, ∃ x₁ x₂ : ℝ,
    (∀ x, l x = (2*x₁ + 2)*x - x₁^2) ∧
    (∀ x, l x = -2*x₂*x + x₂^2 + a) ∧
    l (C₁ x₁) = C₁ x₁ ∧
    l (C₂ x₂) = C₂ x₂) →
  a = -1/2 ∧ (λ x : ℝ => x - 1/4) = l
  := by sorry

end parabolas_common_tangent_l1520_152032


namespace c_is_power_of_two_l1520_152016

/-- Represents a string of base-ten digits -/
def DigitString : Type := List Nat

/-- Checks if a DigitString represents a number divisible by m -/
def isDivisibleBy (s : DigitString) (m : Nat) : Prop := sorry

/-- Counts the number of valid splits of a DigitString -/
def c (S : DigitString) (m : Nat) : Nat := sorry

/-- A natural number is a power of 2 -/
def isPowerOfTwo (n : Nat) : Prop := ∃ k : Nat, n = 2^k

theorem c_is_power_of_two (m : Nat) (S : DigitString) (h1 : m > 1) (h2 : S ≠ []) :
  c S m = 0 ∨ isPowerOfTwo (c S m) := by sorry

end c_is_power_of_two_l1520_152016


namespace quadratic_form_minimum_l1520_152072

theorem quadratic_form_minimum (x y z : ℝ) :
  x^2 + 2*x*y + 3*y^2 + 2*x*z + 3*z^2 ≥ 0 ∧
  (x^2 + 2*x*y + 3*y^2 + 2*x*z + 3*z^2 = 0 ↔ x = 0 ∧ y = 0 ∧ z = 0) :=
by sorry

end quadratic_form_minimum_l1520_152072


namespace stair_step_black_squares_l1520_152081

/-- Represents the number of squares added to form a row in the stair-step pattern -/
def squaresAdded (n : ℕ) : ℕ :=
  if n ≤ 1 then 0 else n - 2

/-- Calculates the total number of squares in the nth row of the stair-step pattern -/
def totalSquares (n : ℕ) : ℕ :=
  1 + (Finset.range n).sum squaresAdded

/-- Calculates the number of black squares in a row with a given total number of squares -/
def blackSquares (total : ℕ) : ℕ :=
  (total - 1) / 2

/-- Theorem: The 20th row of the stair-step pattern contains 85 black squares -/
theorem stair_step_black_squares :
  blackSquares (totalSquares 20) = 85 := by
  sorry


end stair_step_black_squares_l1520_152081


namespace martha_tech_support_ratio_l1520_152033

/-- Proves that the ratio of yelling time to hold time is 1:2 given the conditions of Martha's tech support experience. -/
theorem martha_tech_support_ratio :
  ∀ (router_time hold_time yelling_time : ℕ),
    router_time = 10 →
    hold_time = 6 * router_time →
    router_time + hold_time + yelling_time = 100 →
    (yelling_time : ℚ) / hold_time = 1 / 2 := by
  sorry

end martha_tech_support_ratio_l1520_152033
