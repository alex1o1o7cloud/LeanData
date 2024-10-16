import Mathlib

namespace NUMINAMATH_CALUDE_school_students_count_l196_19613

theorem school_students_count (blue_percent : ℚ) (red_percent : ℚ) (green_percent : ℚ) (other_count : ℕ) :
  blue_percent = 44/100 →
  red_percent = 28/100 →
  green_percent = 10/100 →
  other_count = 162 →
  ∃ (total : ℕ), 
    (blue_percent + red_percent + green_percent < 1) ∧
    (1 - (blue_percent + red_percent + green_percent)) * total = other_count ∧
    total = 900 :=
by
  sorry

end NUMINAMATH_CALUDE_school_students_count_l196_19613


namespace NUMINAMATH_CALUDE_inequality_solution_set_l196_19699

def solution_set (x : ℝ) : Prop := x < 1/3 ∨ x > 2

theorem inequality_solution_set :
  ∀ x : ℝ, (3*x - 1)/(x - 2) > 0 ↔ solution_set x :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l196_19699


namespace NUMINAMATH_CALUDE_kingsley_pants_per_day_l196_19667

/-- Represents the number of shirts Jenson makes per day -/
def jenson_shirts_per_day : ℕ := 3

/-- Represents the amount of fabric used for one shirt in yards -/
def fabric_per_shirt : ℕ := 2

/-- Represents the amount of fabric used for one pair of pants in yards -/
def fabric_per_pants : ℕ := 5

/-- Represents the total amount of fabric needed every 3 days in yards -/
def total_fabric_3days : ℕ := 93

/-- Theorem stating that Kingsley makes 5 pairs of pants per day given the conditions -/
theorem kingsley_pants_per_day :
  ∃ (p : ℕ), 
    p * fabric_per_pants + jenson_shirts_per_day * fabric_per_shirt = total_fabric_3days / 3 ∧
    p = 5 := by
  sorry

end NUMINAMATH_CALUDE_kingsley_pants_per_day_l196_19667


namespace NUMINAMATH_CALUDE_darius_age_l196_19671

theorem darius_age (jenna_age darius_age : ℕ) : 
  jenna_age = 13 →
  jenna_age = darius_age + 5 →
  jenna_age + darius_age = 21 →
  darius_age = 8 := by
sorry

end NUMINAMATH_CALUDE_darius_age_l196_19671


namespace NUMINAMATH_CALUDE_rectangular_field_area_l196_19637

/-- The area of a rectangular field with one side 15 meters and diagonal 17 meters is 120 square meters. -/
theorem rectangular_field_area (side : ℝ) (diagonal : ℝ) (area : ℝ) : 
  side = 15 → diagonal = 17 → area = side * Real.sqrt (diagonal^2 - side^2) → area = 120 :=
by sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l196_19637


namespace NUMINAMATH_CALUDE_sqrt_product_plus_one_l196_19628

theorem sqrt_product_plus_one : 
  Real.sqrt ((35 : ℝ) * 34 * 33 * 32 + 1) = 1121 := by sorry

end NUMINAMATH_CALUDE_sqrt_product_plus_one_l196_19628


namespace NUMINAMATH_CALUDE_ping_pong_ball_price_l196_19620

theorem ping_pong_ball_price 
  (quantity : ℕ) 
  (discount_rate : ℚ) 
  (total_paid : ℚ) 
  (h1 : quantity = 10000)
  (h2 : discount_rate = 30 / 100)
  (h3 : total_paid = 700) :
  let original_price := total_paid / ((1 - discount_rate) * quantity)
  original_price = 1 / 10 := by
sorry

end NUMINAMATH_CALUDE_ping_pong_ball_price_l196_19620


namespace NUMINAMATH_CALUDE_amount_per_bulb_is_fifty_cents_l196_19644

/-- The amount Jane earned for planting flower bulbs -/
def total_earned : ℚ := 75

/-- The number of tulip bulbs Jane planted -/
def tulip_bulbs : ℕ := 20

/-- The number of daffodil bulbs Jane planted -/
def daffodil_bulbs : ℕ := 30

/-- The number of iris bulbs Jane planted -/
def iris_bulbs : ℕ := tulip_bulbs / 2

/-- The number of crocus bulbs Jane planted -/
def crocus_bulbs : ℕ := 3 * daffodil_bulbs

/-- The total number of bulbs Jane planted -/
def total_bulbs : ℕ := tulip_bulbs + iris_bulbs + daffodil_bulbs + crocus_bulbs

/-- The amount paid per bulb -/
def amount_per_bulb : ℚ := total_earned / total_bulbs

theorem amount_per_bulb_is_fifty_cents : amount_per_bulb = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_amount_per_bulb_is_fifty_cents_l196_19644


namespace NUMINAMATH_CALUDE_equation_solution_l196_19650

theorem equation_solution : ∃! y : ℝ, 5 * y - 100 = 125 ∧ y = 45 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l196_19650


namespace NUMINAMATH_CALUDE_mika_stickers_total_l196_19664

/-- The total number of stickers Mika has -/
def total_stickers (initial bought birthday sister mother : ℝ) : ℝ :=
  initial + bought + birthday + sister + mother

/-- Theorem stating that Mika has 130.0 stickers in total -/
theorem mika_stickers_total :
  total_stickers 20.0 26.0 20.0 6.0 58.0 = 130.0 := by
  sorry

end NUMINAMATH_CALUDE_mika_stickers_total_l196_19664


namespace NUMINAMATH_CALUDE_horner_method_f_at_3_l196_19673

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^5 - 2x^3 + 3x^2 - x + 1 -/
def f : List ℝ := [1, -2, 3, 0, -1, 1]

/-- Theorem: Horner's method applied to f(x) at x = 3 yields v₃ = 24 -/
theorem horner_method_f_at_3 :
  horner f 3 = 24 := by
  sorry

#eval horner f 3

end NUMINAMATH_CALUDE_horner_method_f_at_3_l196_19673


namespace NUMINAMATH_CALUDE_R_value_when_S_is_12_l196_19632

-- Define the relationship between R and S
def R (g : ℝ) (S : ℝ) : ℝ := g * S - 6

-- State the theorem
theorem R_value_when_S_is_12 : 
  ∃ g : ℝ, (R g 6 = 12) → (R g 12 = 30) :=
by
  sorry

end NUMINAMATH_CALUDE_R_value_when_S_is_12_l196_19632


namespace NUMINAMATH_CALUDE_gcd_divisibility_l196_19668

theorem gcd_divisibility (p q r s : ℕ+) 
  (h1 : Nat.gcd p.val q.val = 40)
  (h2 : Nat.gcd q.val r.val = 50)
  (h3 : Nat.gcd r.val s.val = 75)
  (h4 : 80 < Nat.gcd s.val p.val)
  (h5 : Nat.gcd s.val p.val < 120) :
  17 ∣ p.val :=
by sorry

end NUMINAMATH_CALUDE_gcd_divisibility_l196_19668


namespace NUMINAMATH_CALUDE_computation_proof_l196_19605

theorem computation_proof : 18 * (216 / 3 + 36 / 6 + 4 / 9 + 2 + 1 / 18) = 1449 := by
  sorry

end NUMINAMATH_CALUDE_computation_proof_l196_19605


namespace NUMINAMATH_CALUDE_probability_two_yellow_balls_probability_two_yellow_balls_is_one_third_l196_19609

/-- The probability of drawing two yellow balls from a bag containing 1 white and 2 yellow balls --/
theorem probability_two_yellow_balls : ℚ :=
  let total_balls : ℕ := 3
  let yellow_balls : ℕ := 2
  let first_draw : ℚ := yellow_balls / total_balls
  let second_draw : ℚ := (yellow_balls - 1) / (total_balls - 1)
  first_draw * second_draw

/-- Proof that the probability of drawing two yellow balls is 1/3 --/
theorem probability_two_yellow_balls_is_one_third :
  probability_two_yellow_balls = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_yellow_balls_probability_two_yellow_balls_is_one_third_l196_19609


namespace NUMINAMATH_CALUDE_second_derivative_f_l196_19685

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x + x * Real.sin x

theorem second_derivative_f (x : ℝ) (hx : x ≠ 0) :
  (deriv^[2] f) x = (2 * x * Real.exp x * (1 - x)) / x^4 + Real.cos x - x * Real.sin x :=
sorry

end NUMINAMATH_CALUDE_second_derivative_f_l196_19685


namespace NUMINAMATH_CALUDE_ashutosh_completion_time_l196_19683

/-- The time it takes Suresh to complete the job alone -/
def suresh_time : ℝ := 15

/-- The time Suresh works on the job -/
def suresh_work_time : ℝ := 9

/-- The time it takes Ashutosh to complete the remaining job -/
def ashutosh_remaining_time : ℝ := 14

/-- The time it takes Ashutosh to complete the job alone -/
def ashutosh_time : ℝ := 35

theorem ashutosh_completion_time :
  (suresh_work_time / suresh_time) + 
  ((1 - suresh_work_time / suresh_time) / ashutosh_time) = 
  (1 / ashutosh_remaining_time) := by
  sorry

#check ashutosh_completion_time

end NUMINAMATH_CALUDE_ashutosh_completion_time_l196_19683


namespace NUMINAMATH_CALUDE_decimal_to_fraction_simplest_l196_19639

theorem decimal_to_fraction_simplest : 
  ∃ (a b : ℕ), 
    a > 0 ∧ b > 0 ∧ 
    (a : ℚ) / (b : ℚ) = 0.84375 ∧
    ∀ (c d : ℕ), c > 0 ∧ d > 0 ∧ (c : ℚ) / (d : ℚ) = 0.84375 → b ≤ d ∧
    a + b = 59 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_simplest_l196_19639


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l196_19622

theorem absolute_value_inequality (x : ℝ) :
  3 ≤ |x + 2| ∧ |x + 2| ≤ 6 ↔ (1 ≤ x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x ≤ -5) :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l196_19622


namespace NUMINAMATH_CALUDE_girls_percentage_after_boy_added_l196_19680

theorem girls_percentage_after_boy_added (initial_boys initial_girls added_boys : ℕ) 
  (h1 : initial_boys = 11)
  (h2 : initial_girls = 13)
  (h3 : added_boys = 1) :
  (initial_girls : ℚ) / ((initial_boys + added_boys + initial_girls) : ℚ) = 52 / 100 := by
sorry

end NUMINAMATH_CALUDE_girls_percentage_after_boy_added_l196_19680


namespace NUMINAMATH_CALUDE_project_time_ratio_l196_19603

/-- Proves that the ratio of time charged by Pat to Kate is 2:1 given the problem conditions -/
theorem project_time_ratio : 
  ∀ (p k m : ℕ) (r : ℚ),
  p + k + m = 153 →
  p = r * k →
  p = m / 3 →
  m = k + 85 →
  r = 2 := by
sorry

end NUMINAMATH_CALUDE_project_time_ratio_l196_19603


namespace NUMINAMATH_CALUDE_binary_101101_eq_45_l196_19646

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 101101₂ -/
def binary_101101 : List Bool := [true, false, true, true, false, true]

/-- Theorem stating that the decimal equivalent of 101101₂ is 45 -/
theorem binary_101101_eq_45 : binary_to_decimal binary_101101 = 45 := by
  sorry

#eval binary_to_decimal binary_101101

end NUMINAMATH_CALUDE_binary_101101_eq_45_l196_19646


namespace NUMINAMATH_CALUDE_village_population_equation_l196_19676

/-- The initial population of a village in Sri Lanka -/
def initial_population : ℕ := 4500

/-- The fraction of people who survived the bombardment -/
def survival_rate : ℚ := 9/10

/-- The fraction of people who remained in the village after some left due to fear -/
def remaining_rate : ℚ := 4/5

/-- The final population of the village -/
def final_population : ℕ := 3240

/-- Theorem stating that the initial population satisfies the given conditions -/
theorem village_population_equation :
  ↑initial_population * (survival_rate * remaining_rate) = final_population := by
  sorry

end NUMINAMATH_CALUDE_village_population_equation_l196_19676


namespace NUMINAMATH_CALUDE_football_club_balance_l196_19647

def initial_balance : ℝ := 100
def players_sold : ℕ := 2
def selling_price : ℝ := 10
def players_bought : ℕ := 4
def buying_price : ℝ := 15

theorem football_club_balance :
  initial_balance + players_sold * selling_price - players_bought * buying_price = 60 :=
by sorry

end NUMINAMATH_CALUDE_football_club_balance_l196_19647


namespace NUMINAMATH_CALUDE_homework_problem_l196_19669

/-- Given a homework assignment with a total number of problems, 
    finished problems, and remaining pages, calculate the number of 
    problems per page assuming each page has the same number of problems. -/
def problems_per_page (total : ℕ) (finished : ℕ) (pages : ℕ) : ℕ :=
  (total - finished) / pages

/-- Theorem stating that for the given homework scenario, 
    there are 7 problems per page. -/
theorem homework_problem : 
  problems_per_page 40 26 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_homework_problem_l196_19669


namespace NUMINAMATH_CALUDE_ab_plus_cd_equals_98_l196_19697

theorem ab_plus_cd_equals_98 
  (a b c d : ℝ) 
  (h1 : a + b + c = 3) 
  (h2 : a + b + d = 9) 
  (h3 : a + c + d = 24) 
  (h4 : b + c + d = 15) : 
  a * b + c * d = 98 := by
sorry

end NUMINAMATH_CALUDE_ab_plus_cd_equals_98_l196_19697


namespace NUMINAMATH_CALUDE_james_baked_1380_muffins_l196_19615

/-- The number of muffins Arthur baked -/
def arthur_muffins : ℕ := 115

/-- The multiplier for James's muffins compared to Arthur's -/
def james_multiplier : ℕ := 12

/-- The number of muffins James baked -/
def james_muffins : ℕ := arthur_muffins * james_multiplier

/-- Proof that James baked 1380 muffins -/
theorem james_baked_1380_muffins : james_muffins = 1380 := by
  sorry

end NUMINAMATH_CALUDE_james_baked_1380_muffins_l196_19615


namespace NUMINAMATH_CALUDE_translation_right_5_units_l196_19612

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point horizontally -/
def translateRight (p : Point) (units : ℝ) : Point :=
  { x := p.x + units, y := p.y }

theorem translation_right_5_units :
  let P : Point := { x := -2, y := -3 }
  let P' : Point := translateRight P 5
  P'.x = 3 ∧ P'.y = -3 := by
  sorry

end NUMINAMATH_CALUDE_translation_right_5_units_l196_19612


namespace NUMINAMATH_CALUDE_max_value_of_f_l196_19648

def f (x : ℝ) : ℝ := 10 * x - 4 * x^2

theorem max_value_of_f :
  ∃ (max : ℝ), max = 25/4 ∧ ∀ (x : ℝ), f x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l196_19648


namespace NUMINAMATH_CALUDE_average_side_length_of_squares_l196_19621

theorem average_side_length_of_squares (a b c : ℝ) 
  (ha : a = 25) (hb : b = 64) (hc : c = 144) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_side_length_of_squares_l196_19621


namespace NUMINAMATH_CALUDE_min_value_of_sum_min_value_is_four_fifths_min_value_equality_l196_19653

theorem min_value_of_sum (a b : ℝ) : 
  a > 0 → b > 0 → a + 2*b = 2 → 
  ∀ x y : ℝ, x > 0 → y > 0 → x + 2*y = 2 → 
  1/(1+a) + 1/(2+2*b) ≤ 1/(1+x) + 1/(2+2*y) :=
by
  sorry

theorem min_value_is_four_fifths (a b : ℝ) :
  a > 0 → b > 0 → a + 2*b = 2 → 
  1/(1+a) + 1/(2+2*b) ≥ 4/5 :=
by
  sorry

theorem min_value_equality (a b : ℝ) :
  a > 0 → b > 0 → a + 2*b = 2 → 
  (1/(1+a) + 1/(2+2*b) = 4/5) ↔ (a = 3/2 ∧ b = 1/4) :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_of_sum_min_value_is_four_fifths_min_value_equality_l196_19653


namespace NUMINAMATH_CALUDE_andy_candy_problem_l196_19651

/-- The number of teachers who gave Andy candy canes -/
def num_teachers : ℕ := sorry

/-- The number of candy canes Andy gets from his parents -/
def candy_from_parents : ℕ := 2

/-- The number of candy canes Andy gets from each teacher -/
def candy_per_teacher : ℕ := 3

/-- The fraction of candy canes Andy buys compared to what he was given -/
def buy_fraction : ℚ := 1 / 7

/-- The number of candy canes that cause one cavity -/
def candy_per_cavity : ℕ := 4

/-- The total number of cavities Andy gets -/
def total_cavities : ℕ := 16

theorem andy_candy_problem :
  let total_candy := candy_from_parents + num_teachers * candy_per_teacher
  let bought_candy := (total_candy : ℚ) * buy_fraction
  (↑total_candy + bought_candy) / candy_per_cavity = total_cavities ↔ num_teachers = 18 := by
  sorry

end NUMINAMATH_CALUDE_andy_candy_problem_l196_19651


namespace NUMINAMATH_CALUDE_ancient_chinese_math_problem_l196_19617

theorem ancient_chinese_math_problem (x y : ℚ) : 
  8 * x = y + 3 ∧ 7 * x = y - 4 → (y + 3) / 8 = (y - 4) / 7 := by
  sorry

end NUMINAMATH_CALUDE_ancient_chinese_math_problem_l196_19617


namespace NUMINAMATH_CALUDE_pencils_per_row_l196_19695

/-- Given 6 pencils placed equally into 2 rows, prove that there are 3 pencils in each row. -/
theorem pencils_per_row (total_pencils : ℕ) (num_rows : ℕ) (pencils_per_row : ℕ) : 
  total_pencils = 6 → num_rows = 2 → total_pencils = num_rows * pencils_per_row → pencils_per_row = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_row_l196_19695


namespace NUMINAMATH_CALUDE_library_book_count_l196_19686

def library_books (initial_books : ℕ) (books_bought_two_years_ago : ℕ) (additional_books_last_year : ℕ) (books_donated : ℕ) : ℕ :=
  initial_books + books_bought_two_years_ago + (books_bought_two_years_ago + additional_books_last_year) - books_donated

theorem library_book_count : 
  library_books 500 300 100 200 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_library_book_count_l196_19686


namespace NUMINAMATH_CALUDE_probability_of_sum_10_15_18_l196_19677

/-- The number of possible outcomes when rolling three dice -/
def total_outcomes : ℕ := 216

/-- The number of ways to get a sum of 10 when rolling three dice -/
def ways_to_get_10 : ℕ := 27

/-- The number of ways to get a sum of 15 when rolling three dice -/
def ways_to_get_15 : ℕ := 9

/-- The number of ways to get a sum of 18 when rolling three dice -/
def ways_to_get_18 : ℕ := 1

/-- The probability of rolling three dice and getting a sum of 10, 15, or 18 -/
theorem probability_of_sum_10_15_18 : 
  (ways_to_get_10 + ways_to_get_15 + ways_to_get_18 : ℚ) / total_outcomes = 37 / 216 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_sum_10_15_18_l196_19677


namespace NUMINAMATH_CALUDE_circle_radius_sqrt_61_l196_19625

/-- Given a circle with center on the x-axis passing through points (2,5) and (3,2),
    prove that its radius is √61. -/
theorem circle_radius_sqrt_61 (x : ℝ) :
  (∀ (y : ℝ), y = 0 →  -- Center is on x-axis
    (x - 2)^2 + (y - 5)^2 = (x - 3)^2 + (y - 2)^2) →  -- Points (2,5) and (3,2) are equidistant from center
  (x - 2)^2 + 5^2 = 61 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_sqrt_61_l196_19625


namespace NUMINAMATH_CALUDE_thread_needed_proof_l196_19655

def thread_per_keychain : ℕ := 12
def friends_from_classes : ℕ := 6
def friends_from_clubs : ℕ := friends_from_classes / 2

def total_friends : ℕ := friends_from_classes + friends_from_clubs

theorem thread_needed_proof : 
  thread_per_keychain * total_friends = 108 := by
  sorry

end NUMINAMATH_CALUDE_thread_needed_proof_l196_19655


namespace NUMINAMATH_CALUDE_ink_remaining_proof_l196_19635

/-- The total area a full marker can cover, in square inches -/
def full_marker_coverage : ℝ := 48

/-- The area covered by the rectangles, in square inches -/
def area_covered : ℝ := 24

/-- The percentage of ink remaining after covering the rectangles -/
def ink_remaining_percentage : ℝ := 50

theorem ink_remaining_proof :
  (full_marker_coverage - area_covered) / full_marker_coverage * 100 = ink_remaining_percentage :=
by sorry

end NUMINAMATH_CALUDE_ink_remaining_proof_l196_19635


namespace NUMINAMATH_CALUDE_no_solution_implies_m_leq_two_l196_19616

theorem no_solution_implies_m_leq_two (m : ℝ) : 
  (∀ x : ℝ, ¬(x - 1 > 1 ∧ x < m)) → m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_leq_two_l196_19616


namespace NUMINAMATH_CALUDE_triangle_angle_proof_l196_19656

theorem triangle_angle_proof (a b c : ℝ) : 
  a = 60 → b = 40 → a + b + c = 180 → c = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_proof_l196_19656


namespace NUMINAMATH_CALUDE_root_transformation_l196_19649

theorem root_transformation (p q r s : ℂ) : 
  (p^4 - 5*p^2 + 6 = 0) ∧ 
  (q^4 - 5*q^2 + 6 = 0) ∧ 
  (r^4 - 5*r^2 + 6 = 0) ∧ 
  (s^4 - 5*s^2 + 6 = 0) →
  ((p+q)/(r+s))^4 + 4*((p+q)/(r+s))^3 + 6*((p+q)/(r+s))^2 + 4*((p+q)/(r+s)) + 1 = 0 ∧
  ((p+r)/(q+s))^4 + 4*((p+r)/(q+s))^3 + 6*((p+r)/(q+s))^2 + 4*((p+r)/(q+s)) + 1 = 0 ∧
  ((p+s)/(q+r))^4 + 4*((p+s)/(q+r))^3 + 6*((p+s)/(q+r))^2 + 4*((p+s)/(q+r)) + 1 = 0 ∧
  ((q+r)/(p+s))^4 + 4*((q+r)/(p+s))^3 + 6*((q+r)/(p+s))^2 + 4*((q+r)/(p+s)) + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l196_19649


namespace NUMINAMATH_CALUDE_original_number_proof_l196_19698

theorem original_number_proof (numbers : Finset ℕ) (original_sum : ℕ) (changed_sum : ℕ) : 
  numbers.card = 7 →
  original_sum / numbers.card = 7 →
  changed_sum / numbers.card = 8 →
  changed_sum = original_sum - (original_sum / numbers.card) + 9 →
  original_sum / numbers.card = 2 := by
sorry

end NUMINAMATH_CALUDE_original_number_proof_l196_19698


namespace NUMINAMATH_CALUDE_jenny_tim_age_difference_l196_19658

/-- Represents the ages of family members --/
structure FamilyAges where
  tim : ℕ
  rommel : ℕ
  jenny : ℕ
  uncle : ℕ
  aunt : ℚ

/-- Defines the relationships between family members' ages --/
def validFamilyAges (ages : FamilyAges) : Prop :=
  ages.tim = 5 ∧
  ages.rommel = 3 * ages.tim ∧
  ages.jenny = ages.rommel + 2 ∧
  ages.uncle = 2 * (ages.rommel + ages.jenny) ∧
  ages.aunt = (ages.uncle + ages.jenny) / 2

/-- Theorem stating the age difference between Jenny and Tim --/
theorem jenny_tim_age_difference (ages : FamilyAges) 
  (h : validFamilyAges ages) : ages.jenny - ages.tim = 12 := by
  sorry

end NUMINAMATH_CALUDE_jenny_tim_age_difference_l196_19658


namespace NUMINAMATH_CALUDE_min_cone_volume_with_sphere_l196_19641

/-- The minimum volume of a cone that contains a sphere of radius 1 touching its base -/
theorem min_cone_volume_with_sphere (r : ℝ) (h : r = 1) : 
  ∃ (V : ℝ), V = Real.pi * 8 / 3 ∧ 
  (∀ (cone_volume : ℝ), 
    (∃ (R h : ℝ), 
      cone_volume = Real.pi * R^2 * h / 3 ∧ 
      r^2 + (R - r)^2 = h^2) → 
    V ≤ cone_volume) :=
by sorry

end NUMINAMATH_CALUDE_min_cone_volume_with_sphere_l196_19641


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l196_19642

theorem complex_fraction_equality : (4 - 2*Complex.I) / (1 + Complex.I) = 1 - 3*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l196_19642


namespace NUMINAMATH_CALUDE_required_plane_satisfies_conditions_l196_19636

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane equation in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- The given plane equation -/
def givenPlane : PlaneEquation := { A := 2, B := -1, C := 4, D := -7 }

/-- The two points that the required plane passes through -/
def point1 : Point3D := { x := 2, y := -1, z := 0 }
def point2 : Point3D := { x := 0, y := 3, z := 1 }

/-- The equation of the required plane -/
def requiredPlane : PlaneEquation := { A := 17, B := 10, C := -6, D := -24 }

/-- Function to check if a point satisfies a plane equation -/
def satisfiesPlaneEquation (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- Function to check if two planes are perpendicular -/
def arePlanesPerp (eq1 eq2 : PlaneEquation) : Prop :=
  eq1.A * eq2.A + eq1.B * eq2.B + eq1.C * eq2.C = 0

/-- Theorem stating that the required plane satisfies all conditions -/
theorem required_plane_satisfies_conditions :
  satisfiesPlaneEquation point1 requiredPlane ∧
  satisfiesPlaneEquation point2 requiredPlane ∧
  arePlanesPerp requiredPlane givenPlane ∧
  requiredPlane.A > 0 ∧
  Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs requiredPlane.A) (Int.natAbs requiredPlane.B)) (Int.natAbs requiredPlane.C)) (Int.natAbs requiredPlane.D) = 1 :=
by sorry


end NUMINAMATH_CALUDE_required_plane_satisfies_conditions_l196_19636


namespace NUMINAMATH_CALUDE_valid_arrangements_count_l196_19674

/-- Represents a seating arrangement in the minibus -/
def SeatingArrangement := Fin 6 → Fin 6

/-- Checks if a seating arrangement is valid (no sibling sits directly in front) -/
def is_valid_arrangement (arr : SeatingArrangement) : Prop :=
  ∀ i j : Fin 3, arr i ≠ arr (i + 3)

/-- The total number of valid seating arrangements -/
def total_valid_arrangements : ℕ := sorry

/-- Theorem stating that the number of valid seating arrangements is 12 -/
theorem valid_arrangements_count : total_valid_arrangements = 12 := by sorry

end NUMINAMATH_CALUDE_valid_arrangements_count_l196_19674


namespace NUMINAMATH_CALUDE_quartic_roots_equivalence_l196_19662

theorem quartic_roots_equivalence (x : ℂ) :
  (3 * x^4 + 2 * x^3 - 8 * x^2 + 2 * x + 3 = 0) ↔
  (∃ y : ℂ, (y = x + 1/x) ∧
    ((y = (-1 + Real.sqrt 43) / 3) ∨ (y = (-1 - Real.sqrt 43) / 3))) := by
  sorry

end NUMINAMATH_CALUDE_quartic_roots_equivalence_l196_19662


namespace NUMINAMATH_CALUDE_max_sum_is_42_l196_19624

/-- Represents the configuration of numbers in the squares -/
structure SquareConfig where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  e : ℕ
  numbers : Finset ℕ
  sum_equality : a + b + e = b + d + e
  valid_numbers : numbers = {2, 5, 8, 11, 14, 17}
  all_different : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

/-- The maximum sum of either horizontal or vertical line is 42 -/
theorem max_sum_is_42 (config : SquareConfig) : 
  (max (config.a + config.b + config.e) (config.b + config.d + config.e)) ≤ 42 ∧ 
  ∃ (config : SquareConfig), (config.a + config.b + config.e) = 42 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_is_42_l196_19624


namespace NUMINAMATH_CALUDE_profit_per_meter_l196_19643

/-- Given a cloth sale scenario, calculate the profit per meter. -/
theorem profit_per_meter
  (meters_sold : ℕ)
  (total_selling_price : ℕ)
  (cost_price_per_meter : ℕ)
  (h1 : meters_sold = 60)
  (h2 : total_selling_price = 8400)
  (h3 : cost_price_per_meter = 128) :
  (total_selling_price - meters_sold * cost_price_per_meter) / meters_sold = 12 := by
sorry


end NUMINAMATH_CALUDE_profit_per_meter_l196_19643


namespace NUMINAMATH_CALUDE_parabola_coefficient_l196_19675

/-- A parabola with equation x = ay² + by + c, vertex at (3, -1), and passing through (7, 3) has a = 1/4 -/
theorem parabola_coefficient (a b c : ℝ) : 
  (∀ y : ℝ, 3 = a * (-1)^2 + b * (-1) + c) →  -- vertex condition
  (∀ y : ℝ, 7 = a * 3^2 + b * 3 + c) →        -- point condition
  a = (1 : ℝ) / 4 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l196_19675


namespace NUMINAMATH_CALUDE_monotonicity_and_extrema_l196_19607

def f (x : ℝ) : ℝ := 2*x^3 + 3*x^2 - 12*x + 1

theorem monotonicity_and_extrema :
  (∀ x < -2, (deriv f) x > 0) ∧
  (∀ x ∈ Set.Ioo (-2 : ℝ) 1, (deriv f) x < 0) ∧
  (∀ x > 1, (deriv f) x > 0) ∧
  IsLocalMax f (-2) ∧
  IsLocalMin f 1 ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≤ f 5) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 5, f x ≥ f 1) ∧
  f 5 = 266 ∧
  f 1 = -6 :=
sorry

end NUMINAMATH_CALUDE_monotonicity_and_extrema_l196_19607


namespace NUMINAMATH_CALUDE_tom_payment_l196_19629

/-- The amount Tom paid to the shopkeeper for apples and mangoes -/
def total_amount (apple_quantity : ℕ) (apple_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  apple_quantity * apple_rate + mango_quantity * mango_rate

/-- Theorem stating that Tom paid 1145 to the shopkeeper -/
theorem tom_payment : total_amount 8 70 9 65 = 1145 := by
  sorry

#eval total_amount 8 70 9 65

end NUMINAMATH_CALUDE_tom_payment_l196_19629


namespace NUMINAMATH_CALUDE_markup_percentage_l196_19604

theorem markup_percentage (cost selling_price markup : ℝ) : 
  markup = selling_price - cost →
  markup = 0.0909090909090909 * selling_price →
  markup = 0.1 * cost := by
  sorry

end NUMINAMATH_CALUDE_markup_percentage_l196_19604


namespace NUMINAMATH_CALUDE_volume_maximized_at_10cm_l196_19614

/-- Represents the dimensions of the original sheet --/
structure SheetDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the volume of the container given sheet dimensions and cut length --/
def containerVolume (sheet : SheetDimensions) (cutLength : ℝ) : ℝ :=
  (sheet.length - 2 * cutLength) * (sheet.width - 2 * cutLength) * cutLength

/-- Theorem stating that the volume is maximized when cut length is 10cm --/
theorem volume_maximized_at_10cm (sheet : SheetDimensions) 
  (h1 : sheet.length = 90)
  (h2 : sheet.width = 48) :
  ∃ (maxCutLength : ℝ), maxCutLength = 10 ∧ 
  ∀ (x : ℝ), 0 < x → x < sheet.width / 2 → x < sheet.length / 2 → 
  containerVolume sheet x ≤ containerVolume sheet maxCutLength :=
sorry

end NUMINAMATH_CALUDE_volume_maximized_at_10cm_l196_19614


namespace NUMINAMATH_CALUDE_radio_selling_price_l196_19681

/-- Calculates the selling price of an item given its cost price and loss percentage. -/
def sellingPrice (costPrice : ℚ) (lossPercentage : ℚ) : ℚ :=
  costPrice * (1 - lossPercentage / 100)

/-- Theorem stating that a radio purchased for Rs 490 with a 5% loss has a selling price of Rs 465.5. -/
theorem radio_selling_price :
  sellingPrice 490 5 = 465.5 := by
  sorry

#eval sellingPrice 490 5

end NUMINAMATH_CALUDE_radio_selling_price_l196_19681


namespace NUMINAMATH_CALUDE_equation_solution_exists_l196_19694

theorem equation_solution_exists : ∃ x : ℝ, 85 * x^2 + ((20 - 7) * 4)^3 / 2 - 15 * 7 = 75000 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_exists_l196_19694


namespace NUMINAMATH_CALUDE_functional_equation_solution_l196_19638

/-- A polynomial satisfying the given functional equation -/
def FunctionalEquationPolynomial (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, P (x^2 - 2*x) = (P (x - 2))^2

/-- The form of the polynomial satisfying the functional equation -/
def PolynomialForm (P : ℝ → ℝ) : Prop :=
  ∃ n : ℕ+, ∀ x : ℝ, P x = (x + 1)^(n : ℕ)

/-- Theorem stating that any non-zero polynomial satisfying the functional equation
    must be of the form (x + 1)^n for some positive integer n -/
theorem functional_equation_solution :
  ∀ P : ℝ → ℝ, (∃ x : ℝ, P x ≠ 0) → FunctionalEquationPolynomial P → PolynomialForm P :=
by sorry


end NUMINAMATH_CALUDE_functional_equation_solution_l196_19638


namespace NUMINAMATH_CALUDE_hannahs_peppers_total_l196_19679

theorem hannahs_peppers_total :
  let green_peppers : ℝ := 0.3333333333333333
  let red_peppers : ℝ := 0.4444444444444444
  let yellow_peppers : ℝ := 0.2222222222222222
  let orange_peppers : ℝ := 0.7777777777777778
  green_peppers + red_peppers + yellow_peppers + orange_peppers = 1.7777777777777777 := by
  sorry

end NUMINAMATH_CALUDE_hannahs_peppers_total_l196_19679


namespace NUMINAMATH_CALUDE_total_original_cost_l196_19663

theorem total_original_cost (x y z : ℝ) : 
  x * (1 + 0.3) = 351 →
  y * (1 + 0.25) = 275 →
  z * (1 + 0.2) = 96 →
  x + y + z = 570 := by
sorry

end NUMINAMATH_CALUDE_total_original_cost_l196_19663


namespace NUMINAMATH_CALUDE_square_has_most_symmetry_axes_l196_19661

/-- The number of symmetry axes for a line segment -/
def line_segment_symmetry_axes : ℕ := 2

/-- The number of symmetry axes for an angle -/
def angle_symmetry_axes : ℕ := 1

/-- The minimum number of symmetry axes for an isosceles triangle -/
def isosceles_triangle_min_symmetry_axes : ℕ := 1

/-- The maximum number of symmetry axes for an isosceles triangle -/
def isosceles_triangle_max_symmetry_axes : ℕ := 3

/-- The number of symmetry axes for a square -/
def square_symmetry_axes : ℕ := 4

/-- Theorem stating that a square has the most symmetry axes among the given shapes -/
theorem square_has_most_symmetry_axes :
  square_symmetry_axes > line_segment_symmetry_axes ∧
  square_symmetry_axes > angle_symmetry_axes ∧
  square_symmetry_axes > isosceles_triangle_min_symmetry_axes ∧
  square_symmetry_axes > isosceles_triangle_max_symmetry_axes :=
sorry

end NUMINAMATH_CALUDE_square_has_most_symmetry_axes_l196_19661


namespace NUMINAMATH_CALUDE_intersection_points_vary_l196_19665

theorem intersection_points_vary (A B C : ℝ) (hA : A > 0) (hC : C > 0) (hB : B ≥ 0) :
  ∃ x y : ℝ, y = A * x^2 + B * x + C ∧ y^2 + 2 * x = x^2 + 4 * y + C ∧
  ∃ A' B' C' : ℝ, A' > 0 ∧ C' > 0 ∧ B' ≥ 0 ∧
    (∃ x1 y1 x2 y2 : ℝ, x1 ≠ x2 ∧
      y1 = A' * x1^2 + B' * x1 + C' ∧ y1^2 + 2 * x1 = x1^2 + 4 * y1 + C' ∧
      y2 = A' * x2^2 + B' * x2 + C' ∧ y2^2 + 2 * x2 = x2^2 + 4 * y2 + C') :=
by sorry

end NUMINAMATH_CALUDE_intersection_points_vary_l196_19665


namespace NUMINAMATH_CALUDE_first_day_over_500_is_saturday_l196_19606

def days : List String := ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

def pens_on_day (day : Nat) : Nat :=
  if day = 0 then 5
  else if day = 1 then 10
  else 10 * (3 ^ (day - 1))

def first_day_over_500 : String :=
  days[(days.findIdx? (fun d => pens_on_day (days.indexOf d) > 500)).getD 0]

theorem first_day_over_500_is_saturday : first_day_over_500 = "Saturday" := by
  sorry

end NUMINAMATH_CALUDE_first_day_over_500_is_saturday_l196_19606


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l196_19691

theorem quadratic_equal_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 + x + m = 0 ∧ (∀ y : ℝ, y^2 + y + m = 0 → y = x)) → m = 1/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l196_19691


namespace NUMINAMATH_CALUDE_otimes_h_h_otimes_h_h_l196_19611

-- Define the binary operation ⊗
def otimes (x y : ℝ) : ℝ := x^2 + y^2

-- Theorem statement
theorem otimes_h_h_otimes_h_h (h : ℝ) : otimes (otimes h h) (otimes h h) = 8 * h^4 := by
  sorry

end NUMINAMATH_CALUDE_otimes_h_h_otimes_h_h_l196_19611


namespace NUMINAMATH_CALUDE_line_segment_endpoint_l196_19608

/-- Given a line segment with midpoint (3, -1) and one endpoint at (7, 2),
    prove that the other endpoint is at (-1, -4). -/
theorem line_segment_endpoint (midpoint endpoint1 endpoint2 : ℝ × ℝ) :
  midpoint = (3, -1) →
  endpoint1 = (7, 2) →
  midpoint = ((endpoint1.1 + endpoint2.1) / 2, (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2 = (-1, -4) := by
  sorry

end NUMINAMATH_CALUDE_line_segment_endpoint_l196_19608


namespace NUMINAMATH_CALUDE_show_attendance_l196_19602

/-- The number of children attending the show -/
def num_children : ℕ := 200

/-- The price of an adult ticket in dollars -/
def adult_ticket_price : ℕ := 32

/-- The total amount collected in dollars -/
def total_amount : ℕ := 16000

/-- The number of adults attending the show -/
def num_adults : ℕ := 400

theorem show_attendance :
  (num_adults * adult_ticket_price + num_children * (adult_ticket_price / 2) = total_amount) ∧
  (num_adults = 400) := by
  sorry

end NUMINAMATH_CALUDE_show_attendance_l196_19602


namespace NUMINAMATH_CALUDE_word_sum_equation_l196_19654

-- Define the type for digits (0-9)
def Digit := Fin 10

-- Define the function that represents the word "TWENTY"
def twenty (t w e n y : Digit) : ℕ :=
  10 * (10 * t.val + w.val) + e.val * 10 + n.val * 1 + y.val

-- Define the function that represents the word "TEN"
def ten (t e n : Digit) : ℕ :=
  10 * t.val + e.val * 1 + n.val

-- Main theorem
theorem word_sum_equation :
  ∃! (e g h i n t w y : Digit),
    twenty t w e n y + twenty t w e n y + twenty t w e n y + ten t e n + ten t e n = 80 ∧
    e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ n ∧ e ≠ t ∧ e ≠ w ∧ e ≠ y ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ n ∧ g ≠ t ∧ g ≠ w ∧ g ≠ y ∧
    h ≠ i ∧ h ≠ n ∧ h ≠ t ∧ h ≠ w ∧ h ≠ y ∧
    i ≠ n ∧ i ≠ t ∧ i ≠ w ∧ i ≠ y ∧
    n ≠ t ∧ n ≠ w ∧ n ≠ y ∧
    t ≠ w ∧ t ≠ y ∧
    w ≠ y := by
  sorry

end NUMINAMATH_CALUDE_word_sum_equation_l196_19654


namespace NUMINAMATH_CALUDE_archer_arrow_recovery_percentage_l196_19688

-- Define the given constants
def shots_per_day : ℕ := 200
def days_per_week : ℕ := 4
def arrow_cost : ℚ := 5.5
def team_payment_percentage : ℚ := 0.7
def archer_weekly_spend : ℚ := 1056

-- Define the theorem
theorem archer_arrow_recovery_percentage :
  let total_shots := shots_per_day * days_per_week
  let total_cost := archer_weekly_spend / (1 - team_payment_percentage)
  let arrows_bought := total_cost / arrow_cost
  let arrows_recovered := total_shots - arrows_bought
  arrows_recovered / total_shots = 1/5 := by
sorry

end NUMINAMATH_CALUDE_archer_arrow_recovery_percentage_l196_19688


namespace NUMINAMATH_CALUDE_equal_max_attendance_l196_19670

-- Define the days of the week
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

-- Define the people
inductive Person
| Anna
| Bill
| Carl
| Dana

-- Define a function that returns whether a person can attend on a given day
def canAttend (p : Person) (d : Day) : Bool :=
  match p, d with
  | Person.Anna, Day.Monday => false
  | Person.Anna, Day.Wednesday => false
  | Person.Anna, Day.Friday => false
  | Person.Bill, Day.Tuesday => false
  | Person.Bill, Day.Thursday => false
  | Person.Carl, Day.Monday => false
  | Person.Carl, Day.Tuesday => false
  | Person.Carl, Day.Thursday => false
  | Person.Carl, Day.Friday => false
  | Person.Dana, Day.Wednesday => false
  | _, _ => true

-- Define a function that counts the number of people who can attend on a given day
def attendanceCount (d : Day) : Nat :=
  List.foldl (fun count p => count + if canAttend p d then 1 else 0) 0 [Person.Anna, Person.Bill, Person.Carl, Person.Dana]

-- Statement to prove
theorem equal_max_attendance :
  ∀ d1 d2 : Day, attendanceCount d1 = attendanceCount d2 ∧ attendanceCount d1 = 2 :=
sorry

end NUMINAMATH_CALUDE_equal_max_attendance_l196_19670


namespace NUMINAMATH_CALUDE_cuboid_volume_l196_19657

/-- The volume of a cuboid that can be divided into 3 equal cubes, each with edges measuring 6 cm, is 648 cm³. -/
theorem cuboid_volume (cuboid : Real) (cube : Real) :
  (cuboid = 3 * cube) →  -- The cuboid is divided into 3 equal parts
  (cube = 6^3) →         -- Each part is a cube with edges measuring 6 cm
  (cuboid = 648) :=      -- The volume of the original cuboid is 648 cm³
by sorry

end NUMINAMATH_CALUDE_cuboid_volume_l196_19657


namespace NUMINAMATH_CALUDE_complex_modulus_l196_19631

theorem complex_modulus (z : ℂ) : (2 - I) * z = 3 + I → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l196_19631


namespace NUMINAMATH_CALUDE_third_number_is_two_l196_19630

def is_valid_sequence (seq : List Nat) : Prop :=
  seq.length = 37 ∧
  seq.toFinset = Finset.range 37 ∧
  ∀ i j, i < j → j < seq.length → (seq.take j).sum % seq[j]! = 0

theorem third_number_is_two (seq : List Nat) :
  is_valid_sequence seq →
  seq[0]! = 37 →
  seq[1]! = 1 →
  seq[2]! = 2 :=
by sorry

end NUMINAMATH_CALUDE_third_number_is_two_l196_19630


namespace NUMINAMATH_CALUDE_lcm_of_20_45_75_l196_19618

theorem lcm_of_20_45_75 : Nat.lcm (Nat.lcm 20 45) 75 = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_75_l196_19618


namespace NUMINAMATH_CALUDE_wedding_rsvp_theorem_l196_19627

def total_guests : ℕ := 200
def yes_percent : ℚ := 83 / 100
def no_percent : ℚ := 9 / 100

theorem wedding_rsvp_theorem :
  (total_guests : ℚ) - (yes_percent * total_guests + no_percent * total_guests) = 16 := by
  sorry

end NUMINAMATH_CALUDE_wedding_rsvp_theorem_l196_19627


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l196_19690

theorem complex_fraction_calculation : 
  (6 + 3/5 - (17/2 - 1/3) / (7/2)) * (2 + 5/18 + 11/12) = 368/27 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l196_19690


namespace NUMINAMATH_CALUDE_quadratic_shift_sum_l196_19682

/-- Given a quadratic function f(x) = 3x^2 + 5x + 9, when shifted 6 units to the left,
    results in a new quadratic function g(x) = ax^2 + bx + c.
    This theorem proves that a + b + c = 191. -/
theorem quadratic_shift_sum (f g : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = 3*x^2 + 5*x + 9) →
  (∀ x, g x = f (x + 6)) →
  (∀ x, g x = a*x^2 + b*x + c) →
  a + b + c = 191 := by
sorry

end NUMINAMATH_CALUDE_quadratic_shift_sum_l196_19682


namespace NUMINAMATH_CALUDE_third_quadrant_condition_l196_19634

def is_in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem third_quadrant_condition (a : ℝ) :
  is_in_third_quadrant ((1 + I) * (a - I)) ↔ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_third_quadrant_condition_l196_19634


namespace NUMINAMATH_CALUDE_sphere_volume_diameter_6_l196_19623

/-- The volume of a sphere with diameter 6 is 36π. -/
theorem sphere_volume_diameter_6 : 
  let d : ℝ := 6
  let r : ℝ := d / 2
  let V : ℝ := (4 / 3) * Real.pi * r ^ 3
  V = 36 * Real.pi := by sorry

end NUMINAMATH_CALUDE_sphere_volume_diameter_6_l196_19623


namespace NUMINAMATH_CALUDE_middle_three_average_l196_19600

theorem middle_three_average (a b c d e : ℕ) : 
  a < b ∧ b < c ∧ c < d ∧ d < e →  -- five different positive integers
  (a + b + c + d + e) / 5 = 5 →  -- average is 5
  e - a = 14 →  -- maximum possible difference
  (b + c + d) / 3 = 3 := by  -- average of middle three is 3
sorry

end NUMINAMATH_CALUDE_middle_three_average_l196_19600


namespace NUMINAMATH_CALUDE_integral_comparison_l196_19645

theorem integral_comparison : ∃ (a b c : ℝ),
  (a = ∫ x in (0:ℝ)..1, x) ∧
  (b = ∫ x in (0:ℝ)..1, x^2) ∧
  (c = ∫ x in (0:ℝ)..1, Real.sqrt x) ∧
  (b < a ∧ a < c) :=
by sorry

end NUMINAMATH_CALUDE_integral_comparison_l196_19645


namespace NUMINAMATH_CALUDE_new_average_age_l196_19626

/-- Calculates the new average age of a class after a student leaves and the teacher's age is included -/
theorem new_average_age 
  (initial_students : Nat) 
  (initial_average_age : ℝ) 
  (leaving_student_age : ℝ) 
  (teacher_age : ℝ) 
  (h1 : initial_students = 30)
  (h2 : initial_average_age = 10)
  (h3 : leaving_student_age = 11)
  (h4 : teacher_age = 41) : 
  let total_initial_age : ℝ := initial_students * initial_average_age
  let remaining_age : ℝ := total_initial_age - leaving_student_age
  let new_total_age : ℝ := remaining_age + teacher_age
  let new_count : Nat := initial_students
  new_total_age / new_count = 11 := by
  sorry


end NUMINAMATH_CALUDE_new_average_age_l196_19626


namespace NUMINAMATH_CALUDE_grid_value_theorem_l196_19652

/-- Represents a 5x5 grid of integers -/
def Grid := Fin 5 → Fin 5 → ℤ

/-- Checks if a sequence of 5 integers forms an arithmetic progression -/
def isArithmeticSequence (seq : Fin 5 → ℤ) : Prop :=
  ∀ i j k : Fin 5, i < j → j < k → seq j - seq i = seq k - seq j

/-- Checks if all rows and columns of a grid form arithmetic sequences -/
def isValidGrid (g : Grid) : Prop :=
  (∀ row : Fin 5, isArithmeticSequence (λ col => g row col)) ∧
  (∀ col : Fin 5, isArithmeticSequence (λ row => g row col))

theorem grid_value_theorem (g : Grid) :
  isValidGrid g →
  g 1 1 = 74 →
  g 2 4 = 186 →
  g 3 2 = 103 →
  g 4 0 = 0 →
  g 0 3 = 142 := by
  sorry

#check grid_value_theorem

end NUMINAMATH_CALUDE_grid_value_theorem_l196_19652


namespace NUMINAMATH_CALUDE_third_flip_probability_is_one_sixth_l196_19689

/-- Represents the "Treasure Box" game in the "Lucky 52" program --/
structure TreasureBoxGame where
  total_logos : ℕ
  winning_logos : ℕ
  flips : ℕ
  flipped_winning_logos : ℕ

/-- The probability of winning on the third flip in the Treasure Box game --/
def third_flip_probability (game : TreasureBoxGame) : ℚ :=
  let remaining_logos := game.total_logos - game.flipped_winning_logos
  let remaining_winning_logos := game.winning_logos - game.flipped_winning_logos
  remaining_winning_logos / remaining_logos

/-- Theorem stating the probability of winning on the third flip --/
theorem third_flip_probability_is_one_sixth 
  (game : TreasureBoxGame) 
  (h1 : game.total_logos = 20)
  (h2 : game.winning_logos = 5)
  (h3 : game.flips = 3)
  (h4 : game.flipped_winning_logos = 2) :
  third_flip_probability game = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_third_flip_probability_is_one_sixth_l196_19689


namespace NUMINAMATH_CALUDE_number_puzzle_l196_19684

theorem number_puzzle (N : ℚ) : (5/4) * N = (4/5) * N + 45 → N = 100 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l196_19684


namespace NUMINAMATH_CALUDE_t_upper_bound_F_positive_l196_19687

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := t / x - f x
noncomputable def F (x : ℝ) : ℝ := f x - 1 / Real.exp x + 2 / (Real.exp 1 * x)

-- Theorem 1
theorem t_upper_bound (t : ℝ) :
  (∀ x > 0, g t x ≤ f x) → t ≤ -2 / Real.exp 1 :=
sorry

-- Theorem 2
theorem F_positive (x : ℝ) :
  x > 0 → F x > 0 :=
sorry

end NUMINAMATH_CALUDE_t_upper_bound_F_positive_l196_19687


namespace NUMINAMATH_CALUDE_middle_number_is_four_l196_19659

def is_valid_triple (a b c : ℕ) : Prop :=
  0 < a ∧ a < b ∧ b < c ∧ a + b + c = 13

def multiple_possibilities_for_bc (a : ℕ) : Prop :=
  ∃ b₁ c₁ b₂ c₂, b₁ ≠ b₂ ∧ is_valid_triple a b₁ c₁ ∧ is_valid_triple a b₂ c₂

def multiple_possibilities_for_ab (c : ℕ) : Prop :=
  ∃ a₁ b₁ a₂ b₂, a₁ ≠ a₂ ∧ is_valid_triple a₁ b₁ c ∧ is_valid_triple a₂ b₂ c

def multiple_possibilities_for_ac (b : ℕ) : Prop :=
  ∃ a₁ c₁ a₂ c₂, a₁ ≠ a₂ ∧ is_valid_triple a₁ b c₁ ∧ is_valid_triple a₂ b c₂

theorem middle_number_is_four (a b c : ℕ) :
  is_valid_triple a b c →
  multiple_possibilities_for_bc a →
  multiple_possibilities_for_ab c →
  multiple_possibilities_for_ac b →
  b = 4 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_is_four_l196_19659


namespace NUMINAMATH_CALUDE_star_star_equation_l196_19678

theorem star_star_equation : 
  ∀ (a b : ℕ), a * b = 34 → (a = 2 ∧ b = 17) ∨ (a = 1 ∧ b = 34) ∨ (a = 17 ∧ b = 2) ∨ (a = 34 ∧ b = 1) :=
by sorry

end NUMINAMATH_CALUDE_star_star_equation_l196_19678


namespace NUMINAMATH_CALUDE_product_mod_twenty_l196_19666

theorem product_mod_twenty : (53 * 76 * 91) % 20 = 8 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_twenty_l196_19666


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l196_19619

theorem indefinite_integral_proof (x : ℝ) : 
  (deriv (λ x => -1/4 * (7*x - 10) * Real.cos (4*x) - 7/16 * Real.sin (4*x))) x = 
  (7*x - 10) * Real.sin (4*x) := by
  sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l196_19619


namespace NUMINAMATH_CALUDE_edward_money_theorem_l196_19696

def remaining_money (initial_amount spent_amount : ℕ) : ℕ :=
  initial_amount - spent_amount

theorem edward_money_theorem (initial_amount spent_amount : ℕ) 
  (h1 : initial_amount ≥ spent_amount) :
  remaining_money initial_amount spent_amount = initial_amount - spent_amount :=
by
  sorry

#eval remaining_money 18 16  -- Should evaluate to 2

end NUMINAMATH_CALUDE_edward_money_theorem_l196_19696


namespace NUMINAMATH_CALUDE_geometric_sequence_parabola_vertex_l196_19692

-- Define a geometric sequence
def is_geometric_sequence (a b c d : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2*x + 3

-- Define the vertex of a parabola
def is_vertex (x y : ℝ) : Prop :=
  parabola x = y ∧ ∀ t : ℝ, parabola t ≥ y

-- Theorem statement
theorem geometric_sequence_parabola_vertex (a b c d : ℝ) :
  is_geometric_sequence a b c d →
  is_vertex b c →
  a * d = 2 := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_parabola_vertex_l196_19692


namespace NUMINAMATH_CALUDE_gregs_shopping_expenditure_l196_19601

theorem gregs_shopping_expenditure (total_spent : ℕ) (shoe_price_difference : ℕ) :
  total_spent = 300 →
  shoe_price_difference = 9 →
  ∃ (shirt_price shoe_price : ℕ),
    shirt_price + shoe_price = total_spent ∧
    shoe_price = 2 * shirt_price + shoe_price_difference ∧
    shirt_price = 97 :=
by sorry

end NUMINAMATH_CALUDE_gregs_shopping_expenditure_l196_19601


namespace NUMINAMATH_CALUDE_james_two_semester_cost_l196_19660

/-- The cost of James's two semesters at community college -/
def two_semester_cost (units_per_semester : ℕ) (cost_per_unit : ℕ) : ℕ :=
  2 * units_per_semester * cost_per_unit

/-- Proof that James pays $2000 for two semesters -/
theorem james_two_semester_cost :
  two_semester_cost 20 50 = 2000 := by
  sorry

#eval two_semester_cost 20 50

end NUMINAMATH_CALUDE_james_two_semester_cost_l196_19660


namespace NUMINAMATH_CALUDE_three_digit_sum_magic_l196_19640

/-- Given a three-digit number abc where a, b, and c are digits in base 10,
    if the sum of (acb), (bca), (bac), (cab), and (cba) is 3333,
    then abc = 555. -/
theorem three_digit_sum_magic (a b c : Nat) : 
  a < 10 → b < 10 → c < 10 →
  (100 * a + 10 * c + b) + 
  (100 * b + 10 * c + a) + 
  (100 * b + 10 * a + c) + 
  (100 * c + 10 * a + b) + 
  (100 * c + 10 * b + a) = 3333 →
  100 * a + 10 * b + c = 555 := by
  sorry


end NUMINAMATH_CALUDE_three_digit_sum_magic_l196_19640


namespace NUMINAMATH_CALUDE_unique_solution_condition_l196_19672

theorem unique_solution_condition (a b : ℝ) :
  (∃! x : ℝ, 4 * x - 7 + a = b * x + 4) ↔ b ≠ 4 := by sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l196_19672


namespace NUMINAMATH_CALUDE_algebraic_identities_l196_19693

theorem algebraic_identities :
  (∀ (a : ℝ), a ≠ 0 → 2 * a^5 + a^7 / a^2 = 3 * a^5) ∧
  (∀ (x y : ℝ), (x + y) * (x - y) + x * (2 * y - x) = 2 * x * y - y^2) :=
by sorry

end NUMINAMATH_CALUDE_algebraic_identities_l196_19693


namespace NUMINAMATH_CALUDE_other_x_intercept_is_negative_one_l196_19610

/-- A quadratic function with vertex (h, k) and one x-intercept at (r, 0) -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h : ℝ
  k : ℝ
  r : ℝ
  vertex_x : h = 2
  vertex_y : k = -3
  intercept : r = 5

/-- The x-coordinate of the other x-intercept of the quadratic function -/
def other_x_intercept (f : QuadraticFunction) : ℝ := 2 * f.h - f.r

theorem other_x_intercept_is_negative_one (f : QuadraticFunction) :
  other_x_intercept f = -1 := by
  sorry

end NUMINAMATH_CALUDE_other_x_intercept_is_negative_one_l196_19610


namespace NUMINAMATH_CALUDE_factorization_problems_l196_19633

theorem factorization_problems (a b x y : ℝ) : 
  (2 * x * (a - b) - (b - a) = (a - b) * (2 * x + 1)) ∧ 
  ((x^2 + y^2)^2 - 4 * x^2 * y^2 = (x - y)^2 * (x + y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l196_19633
