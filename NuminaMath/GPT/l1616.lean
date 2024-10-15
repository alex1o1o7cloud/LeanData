import Mathlib

namespace NUMINAMATH_GPT_flu_infection_equation_l1616_161678

theorem flu_infection_equation (x : ℝ) :
  (1 + x)^2 = 144 :=
sorry

end NUMINAMATH_GPT_flu_infection_equation_l1616_161678


namespace NUMINAMATH_GPT_intersect_sets_example_l1616_161687

open Set

theorem intersect_sets_example : 
  let A := {x : ℝ | -1 < x ∧ x ≤ 3}
  let B := {x : ℝ | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4}
  A ∩ B = {x : ℝ | x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3} :=
by
  sorry

end NUMINAMATH_GPT_intersect_sets_example_l1616_161687


namespace NUMINAMATH_GPT_avg_of_nine_numbers_l1616_161699

theorem avg_of_nine_numbers (average : ℝ) (sum : ℝ) (h : average = (sum / 9)) (h_avg : average = 5.3) : sum = 47.7 := by
  sorry

end NUMINAMATH_GPT_avg_of_nine_numbers_l1616_161699


namespace NUMINAMATH_GPT_number_20_l1616_161671

def Jo (n : ℕ) : ℕ :=
  1 + 5 * (n - 1)

def Blair (n : ℕ) : ℕ :=
  3 + 5 * (n - 1)

def number_at_turn (k : ℕ) : ℕ :=
  if k % 2 = 1 then Jo ((k + 1) / 2) else Blair (k / 2)

theorem number_20 : number_at_turn 20 = 48 :=
by
  sorry

end NUMINAMATH_GPT_number_20_l1616_161671


namespace NUMINAMATH_GPT_number_of_integer_chords_through_point_l1616_161645

theorem number_of_integer_chords_through_point {r : ℝ} {c : ℝ} 
    (hr: r = 13) (hc : c = 12) : 
    ∃ n : ℕ, n = 17 :=
by
  -- Suppose O is the center and P is a point inside the circle such that OP = 12
  -- Given radius r = 13, we need to show there are 17 different integer chord lengths
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_number_of_integer_chords_through_point_l1616_161645


namespace NUMINAMATH_GPT_lydia_eats_apple_age_l1616_161601

-- Define the conditions
def years_to_bear_fruit : ℕ := 7
def age_when_planted : ℕ := 4
def current_age : ℕ := 9

-- Define the theorem statement
theorem lydia_eats_apple_age : 
  (age_when_planted + years_to_bear_fruit = 11) :=
by
  sorry

end NUMINAMATH_GPT_lydia_eats_apple_age_l1616_161601


namespace NUMINAMATH_GPT_sum_of_solutions_l1616_161690

theorem sum_of_solutions (x : ℝ) : (∃ x₁ x₂ : ℝ, (x - 4)^2 = 16 ∧ x = x₁ ∨ x = x₂ ∧ x₁ + x₂ = 8) :=
by sorry

end NUMINAMATH_GPT_sum_of_solutions_l1616_161690


namespace NUMINAMATH_GPT_shopkeeper_loss_percentage_l1616_161661

theorem shopkeeper_loss_percentage {cp sp : ℝ} (h1 : cp = 100) (h2 : sp = cp * 1.1) (h_loss : sp * 0.33 = cp * (1 - x / 100)) :
  x = 70 :=
by
  sorry

end NUMINAMATH_GPT_shopkeeper_loss_percentage_l1616_161661


namespace NUMINAMATH_GPT_parallel_lines_condition_iff_l1616_161617

def line_parallel (a : ℝ) : Prop :=
  let l1_slope := -1 / -a
  let l2_slope := -(a - 1) / -12
  l1_slope = l2_slope

theorem parallel_lines_condition_iff (a : ℝ) :
  (a = 4) ↔ line_parallel a := by
  sorry

end NUMINAMATH_GPT_parallel_lines_condition_iff_l1616_161617


namespace NUMINAMATH_GPT_min_value_2a_plus_b_l1616_161650

theorem min_value_2a_plus_b {a b : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : 3 * a + b = a^2 + a * b) :
  2 * a + b ≥ 2 * Real.sqrt 2 + 3 :=
sorry

end NUMINAMATH_GPT_min_value_2a_plus_b_l1616_161650


namespace NUMINAMATH_GPT_fill_tank_time_l1616_161682

variable (A_rate := 1/3)
variable (B_rate := 1/4)
variable (C_rate := -1/4)

def combined_rate := A_rate + B_rate + C_rate

theorem fill_tank_time (hA : A_rate = 1/3) (hB : B_rate = 1/4) (hC : C_rate = -1/4) : (1 / combined_rate) = 3 := by
  sorry

end NUMINAMATH_GPT_fill_tank_time_l1616_161682


namespace NUMINAMATH_GPT_number_of_bikes_l1616_161644

theorem number_of_bikes (total_wheels : ℕ) (car_wheels : ℕ) (tricycle_wheels : ℕ) (roller_skate_wheels : ℕ) (trash_can_wheels : ℕ) (bike_wheels : ℕ) (num_bikes : ℕ) :
  total_wheels = 25 →
  car_wheels = 2 * 4 →
  tricycle_wheels = 3 →
  roller_skate_wheels = 4 →
  trash_can_wheels = 2 →
  bike_wheels = 2 →
  (total_wheels - (car_wheels + tricycle_wheels + roller_skate_wheels + trash_can_wheels)) = bike_wheels * num_bikes →
  num_bikes = 4 := 
by
  intros total_wheels_eq total_car_wheels_eq tricycle_wheels_eq roller_skate_wheels_eq trash_can_wheels_eq bike_wheels_eq remaining_wheels_eq
  sorry

end NUMINAMATH_GPT_number_of_bikes_l1616_161644


namespace NUMINAMATH_GPT_find_x_range_l1616_161691

theorem find_x_range {x : ℝ} : 
  (∀ (m : ℝ), abs m ≤ 2 → m * x^2 - 2 * x - m + 1 < 0 ) →
  ( ( -1 + Real.sqrt 7 ) / 2 < x ∧ x < ( 1 + Real.sqrt 3 ) / 2 ) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_x_range_l1616_161691


namespace NUMINAMATH_GPT_investment_period_l1616_161639

theorem investment_period (P : ℝ) (r1 r2 : ℝ) (diff : ℝ) (t : ℝ) :
  P = 900 ∧ r1 = 0.04 ∧ r2 = 0.045 ∧ (P * r2 * t) - (P * r1 * t) = 31.50 → t = 7 :=
by
  sorry

end NUMINAMATH_GPT_investment_period_l1616_161639


namespace NUMINAMATH_GPT_condition_sufficiency_not_necessity_l1616_161660

variable {x y : ℝ}

theorem condition_sufficiency_not_necessity (hx : x ≥ 0) (hy : y ≥ 0) :
  (xy > 0 → |x + y| = |x| + |y|) ∧ (|x + y| = |x| + |y| → xy ≥ 0) :=
sorry

end NUMINAMATH_GPT_condition_sufficiency_not_necessity_l1616_161660


namespace NUMINAMATH_GPT_rectangle_symmetry_l1616_161651

-- Define basic geometric terms and the notion of symmetry
structure Rectangle where
  length : ℝ
  width : ℝ
  (length_pos : 0 < length)
  (width_pos : 0 < width)

def is_axes_of_symmetry (r : Rectangle) (n : ℕ) : Prop :=
  -- A hypothetical function that determines whether a rectangle r has n axes of symmetry
  sorry

theorem rectangle_symmetry (r : Rectangle) : is_axes_of_symmetry r 2 := 
  -- This theorem states that a rectangle has exactly 2 axes of symmetry
  sorry

end NUMINAMATH_GPT_rectangle_symmetry_l1616_161651


namespace NUMINAMATH_GPT_determine_a_l1616_161633

-- Define the function f(x)
def f (x a : ℝ) : ℝ := x^2 - a * x + 3

-- Define the condition that f(x) >= a for all x in the interval [-1, +∞)
def condition (a : ℝ) : Prop := ∀ x : ℝ, x ≥ -1 → f x a ≥ a

-- The theorem to prove:
theorem determine_a : ∀ a : ℝ, condition a ↔ a ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_l1616_161633


namespace NUMINAMATH_GPT_fraction_spent_on_DVDs_l1616_161626

theorem fraction_spent_on_DVDs (initial_money spent_on_books additional_books_cost remaining_money_spent fraction remaining_money_after_DVDs : ℚ) : 
  initial_money = 320 ∧
  spent_on_books = initial_money / 4 ∧
  additional_books_cost = 10 ∧
  remaining_money_spent = 230 ∧
  remaining_money_after_DVDs = 130 ∧
  remaining_money_spent = initial_money - (spent_on_books + additional_books_cost) ∧
  remaining_money_after_DVDs = remaining_money_spent - (fraction * remaining_money_spent + 8) 
  → fraction = 46 / 115 :=
by
  intros
  sorry

end NUMINAMATH_GPT_fraction_spent_on_DVDs_l1616_161626


namespace NUMINAMATH_GPT_financing_term_years_l1616_161646

def monthly_payment : Int := 150
def total_financed_amount : Int := 9000

theorem financing_term_years : 
  (total_financed_amount / monthly_payment) / 12 = 5 := 
by
  sorry

end NUMINAMATH_GPT_financing_term_years_l1616_161646


namespace NUMINAMATH_GPT_initial_girls_are_11_l1616_161619

variable {n : ℕ}  -- Assume n (the total number of students initially) is a natural number

def initial_num_girls (n : ℕ) : ℕ := (n / 2)

def total_students_after_changes (n : ℕ) : ℕ := n - 2

def num_girls_after_changes (n : ℕ) : ℕ := (n / 2) - 3

def is_40_percent_girls (n : ℕ) : Prop := (num_girls_after_changes n) * 10 = 4 * (total_students_after_changes n)

theorem initial_girls_are_11 :
  is_40_percent_girls 22 → initial_num_girls 22 = 11 :=
by
  sorry

end NUMINAMATH_GPT_initial_girls_are_11_l1616_161619


namespace NUMINAMATH_GPT_proportion_solution_l1616_161688

theorem proportion_solution (a b c x : ℝ) (h : a / x = 4 * a * b / (17.5 * c)) : 
  x = 17.5 * c / (4 * b) := 
sorry

end NUMINAMATH_GPT_proportion_solution_l1616_161688


namespace NUMINAMATH_GPT_expression_simplification_l1616_161656

variable (x : ℝ)

-- Define the expression as given in the problem
def Expr : ℝ := (3 * x^2 + 4 * x + 8) * (x - 2) - (x - 2) * (x^2 + 5 * x - 72) + (4 * x - 15) * (x - 2) * (x + 3)

-- Lean statement to verify that the expression simplifies to the given polynomial
theorem expression_simplification : Expr x = 6 * x^3 - 16 * x^2 + 43 * x - 70 := by
  sorry

end NUMINAMATH_GPT_expression_simplification_l1616_161656


namespace NUMINAMATH_GPT_triangle_angle_A_l1616_161662

theorem triangle_angle_A (A B C : ℝ) (a b c : ℝ) (hC : C = Real.pi / 6) (hCos : c = 2 * a * Real.cos B) : A = (5 * Real.pi) / 12 :=
  sorry

end NUMINAMATH_GPT_triangle_angle_A_l1616_161662


namespace NUMINAMATH_GPT_obtuse_triangle_iff_l1616_161621

theorem obtuse_triangle_iff (x : ℝ) :
    (x > 1 ∧ x < 3) ↔ (x + (x + 1) > (x + 2) ∧
                        (x + 1) + (x + 2) > x ∧
                        (x + 2) + x > (x + 1) ∧
                        (x + 2)^2 > x^2 + (x + 1)^2) :=
by
  sorry

end NUMINAMATH_GPT_obtuse_triangle_iff_l1616_161621


namespace NUMINAMATH_GPT_xy_problem_l1616_161627

theorem xy_problem (x y : ℝ) (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_xy_problem_l1616_161627


namespace NUMINAMATH_GPT_right_triangle_perimeter_l1616_161615

theorem right_triangle_perimeter (a b : ℝ) (c : ℝ) (h1 : a * b = 72) 
  (h2 : c ^ 2 = a ^ 2 + b ^ 2) (h3 : a = 12) :
  a + b + c = 18 + 6 * Real.sqrt 5 := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_perimeter_l1616_161615


namespace NUMINAMATH_GPT_neg_neg_eq_l1616_161616

theorem neg_neg_eq (n : ℤ) : -(-n) = n :=
  sorry

example : -(-2023) = 2023 :=
by apply neg_neg_eq

end NUMINAMATH_GPT_neg_neg_eq_l1616_161616


namespace NUMINAMATH_GPT_ralph_has_18_fewer_pictures_l1616_161666

/-- Ralph has 58 pictures of wild animals. Derrick has 76 pictures of wild animals.
    Prove that Ralph has 18 fewer pictures of wild animals compared to Derrick. -/
theorem ralph_has_18_fewer_pictures :
  let Ralph_pictures := 58
  let Derrick_pictures := 76
  76 - 58 = 18 :=
by
  let Ralph_pictures := 58
  let Derrick_pictures := 76
  show 76 - 58 = 18
  sorry

end NUMINAMATH_GPT_ralph_has_18_fewer_pictures_l1616_161666


namespace NUMINAMATH_GPT_remainder_of_sum_division_l1616_161657

theorem remainder_of_sum_division (f y : ℤ) (a b : ℤ) (h_f : f = 5 * a + 3) (h_y : y = 5 * b + 4) :  
  (f + y) % 5 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_division_l1616_161657


namespace NUMINAMATH_GPT_number_of_boxes_l1616_161638

variable (boxes : ℕ) -- number of boxes
variable (mangoes_per_box : ℕ) -- mangoes per box
variable (total_mangoes : ℕ) -- total mangoes

def dozen : ℕ := 12

-- Condition: each box contains 10 dozen mangoes
def condition1 : mangoes_per_box = 10 * dozen := by 
  sorry

-- Condition: total mangoes in all boxes together is 4320
def condition2 : total_mangoes = 4320 := by
  sorry

-- Proof problem: prove that the number of boxes is 36
theorem number_of_boxes (h1 : mangoes_per_box = 10 * dozen) 
                        (h2 : total_mangoes = 4320) :
  boxes = 4320 / (10 * dozen) :=
  by
  sorry

end NUMINAMATH_GPT_number_of_boxes_l1616_161638


namespace NUMINAMATH_GPT_smallest_odd_digit_number_gt_1000_mult_5_l1616_161685

def is_odd_digit (n : ℕ) : Prop := n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def valid_number (n : ℕ) : Prop :=
  n > 1000 ∧ (∃ d1 d2 d3 d4, n = d1 * 1000 + d2 * 100 + d3 * 10 + d4 ∧ 
  is_odd_digit d1 ∧ is_odd_digit d2 ∧ is_odd_digit d3 ∧ is_odd_digit d4 ∧ 
  d4 = 5)

theorem smallest_odd_digit_number_gt_1000_mult_5 : ∃ n : ℕ, valid_number n ∧ 
  ∀ m : ℕ, valid_number m → m ≥ n := 
by
  use 1115
  simp [valid_number, is_odd_digit]
  sorry

end NUMINAMATH_GPT_smallest_odd_digit_number_gt_1000_mult_5_l1616_161685


namespace NUMINAMATH_GPT_equal_cost_at_150_miles_l1616_161683

def cost_Safety (m : ℝ) := 41.95 + 0.29 * m
def cost_City (m : ℝ) := 38.95 + 0.31 * m
def cost_Metro (m : ℝ) := 44.95 + 0.27 * m

theorem equal_cost_at_150_miles (m : ℝ) :
  cost_Safety m = cost_City m ∧ cost_Safety m = cost_Metro m → m = 150 :=
by
  sorry

end NUMINAMATH_GPT_equal_cost_at_150_miles_l1616_161683


namespace NUMINAMATH_GPT_find_insect_stickers_l1616_161653

noncomputable def flower_stickers : ℝ := 15
noncomputable def animal_stickers : ℝ := 2 * flower_stickers - 3.5
noncomputable def space_stickers : ℝ := 1.5 * flower_stickers + 5.5
noncomputable def total_stickers : ℝ := 70
noncomputable def insect_stickers : ℝ := total_stickers - (animal_stickers + space_stickers)

theorem find_insect_stickers : insect_stickers = 15.5 := by
  sorry

end NUMINAMATH_GPT_find_insect_stickers_l1616_161653


namespace NUMINAMATH_GPT_emily_necklaces_l1616_161618

theorem emily_necklaces (total_beads : ℤ) (beads_per_necklace : ℤ) 
(h_total_beads : total_beads = 16) (h_beads_per_necklace : beads_per_necklace = 8) : 
  total_beads / beads_per_necklace = 2 := 
by
  sorry

end NUMINAMATH_GPT_emily_necklaces_l1616_161618


namespace NUMINAMATH_GPT_eighth_graders_ninth_grader_points_l1616_161613

noncomputable def eighth_grader_points (y : ℚ) (x : ℕ) : Prop :=
  x * y + 8 = ((x + 2) * (x + 1)) / 2

theorem eighth_graders (x : ℕ) (y : ℚ) (hx : eighth_grader_points y x) :
  x = 7 ∨ x = 14 :=
sorry

noncomputable def tenth_grader_points (z y : ℚ) (x : ℕ) : Prop :=
  10 * z = 4.5 * y ∧ x * z = y

theorem ninth_grader_points (y : ℚ) (x : ℕ) (z : ℚ)
  (hx : tenth_grader_points z y x) :
  y = 10 :=
sorry

end NUMINAMATH_GPT_eighth_graders_ninth_grader_points_l1616_161613


namespace NUMINAMATH_GPT_total_people_correct_current_people_correct_l1616_161648

-- Define the conditions as constants
def morning_people : ℕ := 473
def noon_left : ℕ := 179
def afternoon_people : ℕ := 268

-- Define the total number of people
def total_people : ℕ := morning_people + afternoon_people

-- Define the current number of people in the amusement park
def current_people : ℕ := morning_people - noon_left + afternoon_people

-- Theorem proofs
theorem total_people_correct : total_people = 741 := by sorry
theorem current_people_correct : current_people = 562 := by sorry

end NUMINAMATH_GPT_total_people_correct_current_people_correct_l1616_161648


namespace NUMINAMATH_GPT_car_speed_first_hour_l1616_161680

theorem car_speed_first_hour (x : ℕ) :
  (x + 60) / 2 = 75 → x = 90 :=
by
  -- To complete the proof in Lean, we would need to solve the equation,
  -- reversing the steps provided in the solution. 
  -- But as per instructions, we don't need the proof, hence we put sorry.
  sorry

end NUMINAMATH_GPT_car_speed_first_hour_l1616_161680


namespace NUMINAMATH_GPT_part_I_part_II_l1616_161614

open Real

variable (a b : ℝ)

theorem part_I (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : (1 / a^2) + (1 / b^2) ≥ 8 := 
sorry

theorem part_II (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 1) : (1 / a) + (1 / b) + (1 / (a * b)) ≥ 8 := 
sorry

end NUMINAMATH_GPT_part_I_part_II_l1616_161614


namespace NUMINAMATH_GPT_IvanPetrovich_daily_lessons_and_charity_l1616_161665

def IvanPetrovichConditions (L k : ℕ) : Prop :=
  24 = 8 + 3*L + k ∧
  3000 * L * 21 + 14000 = 70000 + (7000 * k / 3)

theorem IvanPetrovich_daily_lessons_and_charity
  (L k : ℕ) (h : IvanPetrovichConditions L k) :
  L = 2 ∧ 7000 * k / 3 = 70000 := 
by
  sorry

end NUMINAMATH_GPT_IvanPetrovich_daily_lessons_and_charity_l1616_161665


namespace NUMINAMATH_GPT_spider_total_distance_l1616_161640

theorem spider_total_distance :
  let start := 3
  let mid := -4
  let final := 8
  let dist1 := abs (mid - start)
  let dist2 := abs (final - mid)
  let total_distance := dist1 + dist2
  total_distance = 19 :=
by
  sorry

end NUMINAMATH_GPT_spider_total_distance_l1616_161640


namespace NUMINAMATH_GPT_quiz_answer_key_count_l1616_161664

theorem quiz_answer_key_count :
  let true_false_possibilities := 6  -- Combinations for 3 T/F questions where not all are same
  let multiple_choice_possibilities := 4^3  -- 4 choices for each of 3 multiple-choice questions
  true_false_possibilities * multiple_choice_possibilities = 384 := by
  sorry

end NUMINAMATH_GPT_quiz_answer_key_count_l1616_161664


namespace NUMINAMATH_GPT_parabola_through_point_l1616_161689

-- Define the parabola equation property
def parabola (a x : ℝ) : ℝ := x^2 + (a+1) * x + a

-- Introduce the main problem statement
theorem parabola_through_point (a m : ℝ) (h : parabola a (-1) = m) : m = 0 :=
by
  sorry

end NUMINAMATH_GPT_parabola_through_point_l1616_161689


namespace NUMINAMATH_GPT_Oliver_monster_club_cards_l1616_161681

theorem Oliver_monster_club_cards (BG AB MC : ℕ) 
  (h1 : BG = 48) 
  (h2 : BG = 3 * AB) 
  (h3 : MC = 2 * AB) : 
  MC = 32 :=
by sorry

end NUMINAMATH_GPT_Oliver_monster_club_cards_l1616_161681


namespace NUMINAMATH_GPT_linear_regression_forecast_l1616_161623

variable (x : ℝ) (y : ℝ)
variable (b : ℝ) (a : ℝ) (center_x : ℝ) (center_y : ℝ)

theorem linear_regression_forecast :
  b=-2 → center_x=4 → center_y=50 → (center_y = b * center_x + a) →
  (a = 58) → (x = 6) → y = b * x + a → y = 46 :=
by
  intros hb hcx hcy heq ha hx hy
  sorry

end NUMINAMATH_GPT_linear_regression_forecast_l1616_161623


namespace NUMINAMATH_GPT_exists_five_distinct_natural_numbers_product_eq_1000_l1616_161694

theorem exists_five_distinct_natural_numbers_product_eq_1000 :
  ∃ (a b c d e : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a * b * c * d * e = 1000 := sorry

end NUMINAMATH_GPT_exists_five_distinct_natural_numbers_product_eq_1000_l1616_161694


namespace NUMINAMATH_GPT_max_download_speed_l1616_161676

def download_speed (size_GB : ℕ) (time_hours : ℕ) : ℚ :=
  let size_MB := size_GB * 1024
  let time_seconds := time_hours * 60 * 60
  size_MB / time_seconds

theorem max_download_speed (h₁ : size_GB = 360) (h₂ : time_hours = 2) :
  download_speed size_GB time_hours = 51.2 :=
by
  sorry

end NUMINAMATH_GPT_max_download_speed_l1616_161676


namespace NUMINAMATH_GPT_base_2_representation_of_123_l1616_161693

theorem base_2_representation_of_123 : (123 : ℕ) = 1 * 2^6 + 1 * 2^5 + 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0 :=
by
  sorry

end NUMINAMATH_GPT_base_2_representation_of_123_l1616_161693


namespace NUMINAMATH_GPT_move_point_right_3_units_l1616_161684

theorem move_point_right_3_units (x y : ℤ) (hx : x = 2) (hy : y = -1) :
  (x + 3, y) = (5, -1) :=
by
  sorry

end NUMINAMATH_GPT_move_point_right_3_units_l1616_161684


namespace NUMINAMATH_GPT_units_digit_expression_l1616_161602

theorem units_digit_expression: 
  (8 * 19 * 1981 + 6^3 - 2^5) % 10 = 6 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_expression_l1616_161602


namespace NUMINAMATH_GPT_pet_store_initial_house_cats_l1616_161663

theorem pet_store_initial_house_cats
    (H : ℕ)
    (h1 : 13 + H - 10 = 8) :
    H = 5 :=
by
  sorry

end NUMINAMATH_GPT_pet_store_initial_house_cats_l1616_161663


namespace NUMINAMATH_GPT_area_of_right_square_l1616_161695

theorem area_of_right_square (side_length_left : ℕ) (side_length_left_eq : side_length_left = 10) : ∃ area_right, area_right = 68 := 
by
  sorry

end NUMINAMATH_GPT_area_of_right_square_l1616_161695


namespace NUMINAMATH_GPT_a_share_in_gain_l1616_161667

noncomputable def investment_share (x: ℝ) (total_gain: ℝ): ℝ := 
  let a_interest := x * 0.1
  let b_interest := (2 * x) * (7 / 100) * (1.5)
  let c_interest := (3 * x) * (10 / 100) * (1.33)
  let total_interest := a_interest + b_interest + c_interest
  a_interest

theorem a_share_in_gain (total_gain: ℝ) (a_share: ℝ) (x: ℝ)
  (hx: 0.709 * x = total_gain):
  investment_share x total_gain = a_share :=
sorry

end NUMINAMATH_GPT_a_share_in_gain_l1616_161667


namespace NUMINAMATH_GPT_rectangle_ratio_l1616_161673

theorem rectangle_ratio (s x y : ℝ) 
  (h_outer_area : (2 * s) ^ 2 = 4 * s ^ 2)
  (h_inner_sides : s + 2 * y = 2 * s)
  (h_outer_sides : x + y = 2 * s) :
  x / y = 3 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_ratio_l1616_161673


namespace NUMINAMATH_GPT_jeff_pencils_initial_l1616_161649

def jeff_initial_pencils (J : ℝ) := J
def jeff_remaining_pencils (J : ℝ) := 0.70 * J
def vicki_initial_pencils (J : ℝ) := 2 * J
def vicki_remaining_pencils (J : ℝ) := 0.25 * vicki_initial_pencils J
def remaining_pencils (J : ℝ) := jeff_remaining_pencils J + vicki_remaining_pencils J

theorem jeff_pencils_initial (J : ℝ) (h : remaining_pencils J = 360) : J = 300 :=
by
  sorry

end NUMINAMATH_GPT_jeff_pencils_initial_l1616_161649


namespace NUMINAMATH_GPT_cost_when_q_is_2_l1616_161692

-- Defining the cost function
def cost (q : ℕ) : ℕ := q^3 + q - 1

-- Theorem to prove the cost when q = 2
theorem cost_when_q_is_2 : cost 2 = 9 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_cost_when_q_is_2_l1616_161692


namespace NUMINAMATH_GPT_find_a_plus_c_l1616_161600

theorem find_a_plus_c {a b c d : ℝ} 
  (h1 : ∀ x, -|x - a| + b = |x - c| + d → x = 4 ∧ -|4 - a| + b = 7 ∨ x = 10 ∧ -|10 - a| + b = 3)
  (h2 : b + d = 12): a + c = 14 := by
  sorry

end NUMINAMATH_GPT_find_a_plus_c_l1616_161600


namespace NUMINAMATH_GPT_train_crossing_time_l1616_161642

-- Definitions for the conditions
def speed_kmph : Float := 72
def speed_mps : Float := speed_kmph * (1000 / 3600)
def length_train_m : Float := 240.0416
def length_platform_m : Float := 280
def total_distance_m : Float := length_train_m + length_platform_m

-- The problem statement
theorem train_crossing_time :
  (total_distance_m / speed_mps) = 26.00208 :=
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l1616_161642


namespace NUMINAMATH_GPT_not_unique_y_20_paise_l1616_161652

theorem not_unique_y_20_paise (x y z w : ℕ) : 
  x + y + z + w = 750 → 10 * x + 20 * y + 50 * z + 100 * w = 27500 → ∃ (y₁ y₂ : ℕ), y₁ ≠ y₂ :=
by 
  intro h1 h2
  -- Without additional constraints on x, y, z, w,
  -- suppose that there are at least two different solutions satisfying both equations,
  -- demonstrating the non-uniqueness of y.
  sorry

end NUMINAMATH_GPT_not_unique_y_20_paise_l1616_161652


namespace NUMINAMATH_GPT_abs_eq_two_l1616_161610

theorem abs_eq_two (m : ℤ) (h : |m| = 2) : m = 2 ∨ m = -2 :=
sorry

end NUMINAMATH_GPT_abs_eq_two_l1616_161610


namespace NUMINAMATH_GPT_calculate_change_l1616_161605

theorem calculate_change : 
  let bracelet_cost := 15
  let necklace_cost := 10
  let mug_cost := 20
  let num_bracelets := 3
  let num_necklaces := 2
  let num_mugs := 1
  let discount := 0.10
  let total_cost := (num_bracelets * bracelet_cost) + (num_necklaces * necklace_cost) + (num_mugs * mug_cost)
  let discount_amount := total_cost * discount
  let final_amount := total_cost - discount_amount
  let payment := 100
  let change := payment - final_amount
  change = 23.50 :=
by
  -- Intentionally skipping the proof
  sorry

end NUMINAMATH_GPT_calculate_change_l1616_161605


namespace NUMINAMATH_GPT_smallest_m_satisfying_conditions_l1616_161604

theorem smallest_m_satisfying_conditions :
  ∃ m : ℕ, m = 4 ∧ (∃ k : ℕ, 0 ≤ k ∧ k ≤ m ∧ (m^2 + m) % k ≠ 0) ∧ (∀ k : ℕ, (0 ≤ k ∧ k ≤ m) → (k ≠ 0 → (m^2 + m) % k = 0)) :=
sorry

end NUMINAMATH_GPT_smallest_m_satisfying_conditions_l1616_161604


namespace NUMINAMATH_GPT_triple_solution_exists_and_unique_l1616_161658

theorem triple_solution_exists_and_unique:
  ∀ (x y z : ℝ), (1 + x^4 ≤ 2 * (y - z) ^ 2) ∧ (1 + y^4 ≤ 2 * (z - x) ^ 2) ∧ (1 + z^4 ≤ 2 * (x - y) ^ 2)
  → (x = 1 ∧ y = 0 ∧ z = -1) :=
by
  sorry

end NUMINAMATH_GPT_triple_solution_exists_and_unique_l1616_161658


namespace NUMINAMATH_GPT_probability_defective_units_l1616_161677

theorem probability_defective_units (X : ℝ) (hX : X > 0) :
  let defectA := (14 / 2000) * (0.40 * X)
  let defectB := (9 / 1500) * (0.35 * X)
  let defectC := (7 / 1000) * (0.25 * X)
  let total_defects := defectA + defectB + defectC
  let total_units := X
  let probability := total_defects / total_units
  probability = 0.00665 :=
by
  sorry

end NUMINAMATH_GPT_probability_defective_units_l1616_161677


namespace NUMINAMATH_GPT_training_cost_per_month_correct_l1616_161679

-- Define the conditions
def salary1 : ℕ := 42000
def revenue1 : ℕ := 93000
def training_duration : ℕ := 3
def salary2 : ℕ := 45000
def revenue2 : ℕ := 92000
def bonus2 : ℕ := (45000 / 100) -- 1% of salary2 which is 450
def net_gain_diff : ℕ := 850

-- Define the monthly training cost for the first applicant
def monthly_training_cost : ℕ := 1786667 / 100

-- Prove that the monthly training cost for the first applicant is correct
theorem training_cost_per_month_correct :
  (revenue1 - (salary1 + 3 * monthly_training_cost) = revenue2 - (salary2 + bonus2) + net_gain_diff) :=
by
  sorry

end NUMINAMATH_GPT_training_cost_per_month_correct_l1616_161679


namespace NUMINAMATH_GPT_lowest_score_l1616_161624

theorem lowest_score (max_mark : ℕ) (n_tests : ℕ) (avg_mark : ℕ) (h_avg : n_tests * avg_mark = 352) (h_max : ∀ k, k < n_tests → k ≤ max_mark) :
  ∃ x, (x ≤ max_mark ∧ (3 * max_mark + x) = 352) ∧ x = 52 :=
by
  sorry

end NUMINAMATH_GPT_lowest_score_l1616_161624


namespace NUMINAMATH_GPT_shifted_polynomial_sum_l1616_161631

theorem shifted_polynomial_sum (a b c : ℝ) :
  (∀ x : ℝ, (3 * x^2 + 2 * x + 5) = (a * (x + 5)^2 + b * (x + 5) + c)) →
  a + b + c = 125 :=
by
  sorry

end NUMINAMATH_GPT_shifted_polynomial_sum_l1616_161631


namespace NUMINAMATH_GPT_expected_total_rain_l1616_161630

noncomputable def expected_daily_rain : ℝ :=
  (0.50 * 0) + (0.30 * 3) + (0.20 * 8)

theorem expected_total_rain :
  (5 * expected_daily_rain) = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_expected_total_rain_l1616_161630


namespace NUMINAMATH_GPT_driving_time_l1616_161620

-- Conditions from problem
variable (distance1 : ℕ) (time1 : ℕ) (distance2 : ℕ)
variable (same_speed : distance1 / time1 = distance2 / (5 : ℕ))

-- Statement to prove
theorem driving_time (h1 : distance1 = 120) (h2 : time1 = 3) (h3 : distance2 = 200)
  : distance2 / (40 : ℕ) = (5 : ℕ) := by
  sorry

end NUMINAMATH_GPT_driving_time_l1616_161620


namespace NUMINAMATH_GPT_beth_wins_if_arjun_plays_first_l1616_161632

/-- 
In the game where players take turns removing one, two adjacent, or two non-adjacent bricks from 
walls, given certain configurations, the configuration where Beth has a guaranteed winning 
strategy if Arjun plays first is (7, 3, 1).
-/
theorem beth_wins_if_arjun_plays_first :
  let nim_value_1 := 1
  let nim_value_2 := 2
  let nim_value_3 := 3
  let nim_value_7 := 2 -- computed as explained in the solution
  ∀ config : List ℕ,
    config = [7, 1, 1] ∨ config = [7, 2, 1] ∨ config = [7, 2, 2] ∨ config = [7, 3, 1] ∨ config = [7, 3, 2] →
    match config with
    | [7, 3, 1] => true
    | _ => false :=
by
  sorry

end NUMINAMATH_GPT_beth_wins_if_arjun_plays_first_l1616_161632


namespace NUMINAMATH_GPT_sum_of_digits_power_of_9_gt_9_l1616_161670

def sum_of_digits (n : ℕ) : ℕ :=
  -- function to calculate the sum of digits of n 
  sorry

theorem sum_of_digits_power_of_9_gt_9 (n : ℕ) (h : n ≥ 3) : sum_of_digits (9^n) > 9 :=
  sorry

end NUMINAMATH_GPT_sum_of_digits_power_of_9_gt_9_l1616_161670


namespace NUMINAMATH_GPT_y_intercept_of_parallel_line_l1616_161675

-- Define the conditions for the problem
def line_parallel (m1 m2 : ℝ) : Prop := 
  m1 = m2

def point_on_line (m : ℝ) (b x1 y1 : ℝ) : Prop := 
  y1 = m * x1 + b

-- Define the main problem statement
theorem y_intercept_of_parallel_line (m b1 b2 x1 y1 : ℝ) 
  (h1 : line_parallel m 3) 
  (h2 : point_on_line m b1 x1 y1) 
  (h3 : x1 = 1) 
  (h4 : y1 = 2) 
  : b1 = -1 :=
sorry

end NUMINAMATH_GPT_y_intercept_of_parallel_line_l1616_161675


namespace NUMINAMATH_GPT_bob_final_total_score_l1616_161608

theorem bob_final_total_score 
  (points_per_correct : ℕ := 5)
  (points_per_incorrect : ℕ := 2)
  (correct_answers : ℕ := 18)
  (incorrect_answers : ℕ := 2) :
  (points_per_correct * correct_answers - points_per_incorrect * incorrect_answers) = 86 :=
by 
  sorry

end NUMINAMATH_GPT_bob_final_total_score_l1616_161608


namespace NUMINAMATH_GPT_Gina_makes_30_per_hour_l1616_161628

variable (rose_cups_per_hour lily_cups_per_hour : ℕ)
variable (rose_cup_order lily_cup_order total_payment : ℕ)
variable (total_hours : ℕ)

def Gina_hourly_rate (rose_cups_per_hour: ℕ) (lily_cups_per_hour: ℕ) (rose_cup_order: ℕ) (lily_cup_order: ℕ) (total_payment: ℕ) : Prop :=
    let rose_time := rose_cup_order / rose_cups_per_hour
    let lily_time := lily_cup_order / lily_cups_per_hour
    let total_time := rose_time + lily_time
    total_payment / total_time = total_hours

theorem Gina_makes_30_per_hour :
    let rose_cups_per_hour := 6
    let lily_cups_per_hour := 7
    let rose_cup_order := 6
    let lily_cup_order := 14
    let total_payment := 90
    Gina_hourly_rate rose_cups_per_hour lily_cups_per_hour rose_cup_order lily_cup_order total_payment 30 :=
by
    sorry

end NUMINAMATH_GPT_Gina_makes_30_per_hour_l1616_161628


namespace NUMINAMATH_GPT_terminal_side_in_fourth_quadrant_l1616_161634

theorem terminal_side_in_fourth_quadrant (θ : ℝ) (h1 : Real.cos θ > 0) (h2 : Real.sin (2 * θ) < 0) : 
  (θ ≥ 0 ∧ θ < Real.pi/2) ∨ (θ > 3 * Real.pi / 2 ∧ θ < 2 * Real.pi) :=
sorry

end NUMINAMATH_GPT_terminal_side_in_fourth_quadrant_l1616_161634


namespace NUMINAMATH_GPT_train_speed_l1616_161622

def train_length : ℝ := 110
def bridge_length : ℝ := 265
def crossing_time : ℝ := 30
def conversion_factor : ℝ := 3.6

theorem train_speed (train_length bridge_length crossing_time conversion_factor : ℝ) :
  (train_length + bridge_length) / crossing_time * conversion_factor = 45 :=
by
  sorry

end NUMINAMATH_GPT_train_speed_l1616_161622


namespace NUMINAMATH_GPT_find_number_l1616_161647

theorem find_number (x : ℕ) (h : 23 + x = 34) : x = 11 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1616_161647


namespace NUMINAMATH_GPT_ship_speed_in_still_water_eq_25_l1616_161635

-- Definitions and conditions
variable (x : ℝ) (h1 : 81 / (x + 2) = 69 / (x - 2)) (h2 : x ≠ -2) (h3 : x ≠ 2)

-- Theorem statement
theorem ship_speed_in_still_water_eq_25 : x = 25 :=
by
  sorry

end NUMINAMATH_GPT_ship_speed_in_still_water_eq_25_l1616_161635


namespace NUMINAMATH_GPT_feet_of_pipe_per_bolt_l1616_161697

-- Definition of the initial conditions
def total_pipe_length := 40 -- total feet of pipe
def washers_per_bolt := 2
def initial_washers := 20
def remaining_washers := 4

-- The proof statement
theorem feet_of_pipe_per_bolt :
  ∀ (total_pipe_length washers_per_bolt initial_washers remaining_washers : ℕ),
  initial_washers - remaining_washers = 16 → -- 16 washers used
  16 / washers_per_bolt = 8 → -- 8 bolts used
  total_pipe_length / 8 = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_feet_of_pipe_per_bolt_l1616_161697


namespace NUMINAMATH_GPT_find_tip_percentage_l1616_161672

def original_bill : ℝ := 139.00
def per_person_share : ℝ := 30.58
def number_of_people : ℕ := 5

theorem find_tip_percentage (original_bill : ℝ) (per_person_share : ℝ) (number_of_people : ℕ) 
  (total_paid : ℝ := per_person_share * number_of_people) 
  (tip_amount : ℝ := total_paid - original_bill) : 
  (tip_amount / original_bill) * 100 = 10 :=
by
  sorry

end NUMINAMATH_GPT_find_tip_percentage_l1616_161672


namespace NUMINAMATH_GPT_largest_five_digit_integer_l1616_161696

/-- The product of the digits of the integer 98752 is (7 * 6 * 5 * 4 * 3 * 2 * 1), and
    98752 is the largest five-digit integer with this property. -/
theorem largest_five_digit_integer :
  (∃ (n : ℕ), n = 98752 ∧ (∃ (d1 d2 d3 d4 d5 : ℕ),
    n = d1 * 10^4 + d2 * 10^3 + d3 * 10^2 + d4 * 10 + d5 ∧
    (d1 * d2 * d3 * d4 * d5 = 7 * 6 * 5 * 4 * 3 * 2 * 1) ∧
    (∀ (m : ℕ), m ≠ 98752 → m < 100000 ∧ (∃ (e1 e2 e3 e4 e5 : ℕ),
    m = e1 * 10^4 + e2 * 10^3 + e3 * 10^2 + e4 * 10 + e5 →
    (e1 * e2 * e3 * e4 * e5 = 7 * 6 * 5 * 4 * 3 * 2 * 1) → m < 98752)))) :=
  sorry

end NUMINAMATH_GPT_largest_five_digit_integer_l1616_161696


namespace NUMINAMATH_GPT_fraction_zero_implies_x_eq_one_l1616_161606

theorem fraction_zero_implies_x_eq_one (x : ℝ) (h : (x - 1) / (x + 1) = 0) : x = 1 :=
sorry

end NUMINAMATH_GPT_fraction_zero_implies_x_eq_one_l1616_161606


namespace NUMINAMATH_GPT_triangle_problem_l1616_161611

def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

def has_same_area (a b : ℕ) (area : ℝ) : Prop :=
  let s := (2 * a + b) / 2
  let areaT := Real.sqrt (s * (s - a) * (s - a) * (s - b))
  areaT = area

def has_same_perimeter (a b : ℕ) (perimeter : ℕ) : Prop :=
  2 * a + b = perimeter

def correct_b (b : ℕ) : Prop :=
  b = 5

theorem triangle_problem
  (a1 a2 b1 b2 : ℕ)
  (h1 : is_isosceles_triangle a1 a1 b1)
  (h2 : is_isosceles_triangle a2 a2 b2)
  (h3 : has_same_area a1 b1 (Real.sqrt 275))
  (h4 : has_same_perimeter a1 b1 22)
  (h5 : has_same_area a2 b2 (Real.sqrt 275))
  (h6 : has_same_perimeter a2 b2 22)
  (h7 : ¬(a1 = a2 ∧ b1 = b2)) : correct_b b2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_problem_l1616_161611


namespace NUMINAMATH_GPT_cole_runs_7_miles_l1616_161625

theorem cole_runs_7_miles
  (xavier_miles : ℕ)
  (katie_miles : ℕ)
  (cole_miles : ℕ)
  (h1 : xavier_miles = 3 * katie_miles)
  (h2 : katie_miles = 4 * cole_miles)
  (h3 : xavier_miles = 84)
  (h4 : katie_miles = 28) :
  cole_miles = 7 := 
sorry

end NUMINAMATH_GPT_cole_runs_7_miles_l1616_161625


namespace NUMINAMATH_GPT_line_intersects_x_axis_l1616_161686

theorem line_intersects_x_axis (x y : ℝ) (h : 5 * y - 6 * x = 15) (hy : y = 0) : x = -2.5 ∧ y = 0 := 
by
  sorry

end NUMINAMATH_GPT_line_intersects_x_axis_l1616_161686


namespace NUMINAMATH_GPT_factor_polynomial_l1616_161698

theorem factor_polynomial (x : ℝ) : 54*x^3 - 135*x^5 = 27*x^3*(2 - 5*x^2) := 
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l1616_161698


namespace NUMINAMATH_GPT_min_value_reciprocal_l1616_161629

theorem min_value_reciprocal (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) :
  (1 / a + 1 / b) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_reciprocal_l1616_161629


namespace NUMINAMATH_GPT_circles_intersect_range_l1616_161655

def circle1_radius := 3
def circle2_radius := 5

theorem circles_intersect_range : 2 < d ∧ d < 8 :=
by
  let r1 := circle1_radius
  let r2 := circle2_radius
  have h1 : d > r2 - r1 := sorry
  have h2 : d < r2 + r1 := sorry
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_circles_intersect_range_l1616_161655


namespace NUMINAMATH_GPT_volume_ratio_of_spheres_l1616_161637

theorem volume_ratio_of_spheres 
  (r1 r2 : ℝ) 
  (h_surface_area : (4 * Real.pi * r1^2) / (4 * Real.pi * r2^2) = 1 / 16) : 
  (4 / 3 * Real.pi * r1^3) / (4 / 3 * Real.pi * r2^3) = 1 / 64 :=
by 
  sorry

end NUMINAMATH_GPT_volume_ratio_of_spheres_l1616_161637


namespace NUMINAMATH_GPT_smallest_positive_period_f_max_min_f_on_interval_l1616_161668

noncomputable def f (x : ℝ) : ℝ := 4 * Real.cos x * Real.sin (x - Real.pi / 6) + 1

theorem smallest_positive_period_f : 
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ Real.pi) :=
sorry

theorem max_min_f_on_interval :
  let a := Real.pi / 4
  let b := 2 * Real.pi / 3
  ∃ M m, (∀ x, a ≤ x ∧ x ≤ b → f x ≤ M ∧ f x ≥ m) ∧ (M = 2) ∧ (m = -1) :=
sorry

end NUMINAMATH_GPT_smallest_positive_period_f_max_min_f_on_interval_l1616_161668


namespace NUMINAMATH_GPT_savanna_more_giraffes_l1616_161643

-- Definitions based on conditions
def lions_safari := 100
def snakes_safari := lions_safari / 2
def giraffes_safari := snakes_safari - 10

def lions_savanna := 2 * lions_safari
def snakes_savanna := 3 * snakes_safari

-- Totals given and to calculate giraffes in Savanna
def total_animals_savanna := 410

-- Prove that Savanna has 20 more giraffes than Safari
theorem savanna_more_giraffes :
  ∃ (giraffes_savanna : ℕ), giraffes_savanna = total_animals_savanna - lions_savanna - snakes_savanna ∧
  giraffes_savanna - giraffes_safari = 20 :=
  by
  sorry

end NUMINAMATH_GPT_savanna_more_giraffes_l1616_161643


namespace NUMINAMATH_GPT_system_solution_l1616_161603

theorem system_solution (x y z a : ℝ) (h1 : x + y + z = 1) (h2 : 1/x + 1/y + 1/z = 1) (h3 : x * y * z = a) :
    (x = 1 ∧ y = Real.sqrt (-a) ∧ z = -Real.sqrt (-a)) ∨
    (x = 1 ∧ y = -Real.sqrt (-a) ∧ z = Real.sqrt (-a)) ∨
    (x = Real.sqrt (-a) ∧ y = -Real.sqrt (-a) ∧ z = 1) ∨
    (x = -Real.sqrt (-a) ∧ y = Real.sqrt (-a) ∧ z = 1) ∨
    (x = Real.sqrt (-a) ∧ y = 1 ∧ z = -Real.sqrt (-a)) ∨
    (x = -Real.sqrt (-a) ∧ y = 1 ∧ z = Real.sqrt (-a)) :=
sorry

end NUMINAMATH_GPT_system_solution_l1616_161603


namespace NUMINAMATH_GPT_solve_equation_l1616_161612

theorem solve_equation (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ -2) :
    (3 / (x + 2) - 1 / x = 0) → x = 1 :=
  by sorry

end NUMINAMATH_GPT_solve_equation_l1616_161612


namespace NUMINAMATH_GPT_distance_P_to_AB_l1616_161654

def point_P_condition (P : ℝ) : Prop :=
  P > 0 ∧ P < 1

def parallel_line_property (P : ℝ) (h : ℝ) : Prop :=
  h = 1 - P / 1

theorem distance_P_to_AB (P h : ℝ) (area_total : ℝ) (area_smaller : ℝ) :
  point_P_condition P →
  parallel_line_property P h →
  (area_smaller / area_total) = 1 / 3 →
  h = 2 / 3 :=
by
  intro hP hp hratio
  sorry

end NUMINAMATH_GPT_distance_P_to_AB_l1616_161654


namespace NUMINAMATH_GPT_odd_function_value_l1616_161669

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then x^2 + x else -(x^2 + x)

theorem odd_function_value : f (-3) = -12 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_odd_function_value_l1616_161669


namespace NUMINAMATH_GPT_circumcircle_area_l1616_161674

theorem circumcircle_area (a b c A B C : ℝ) (h : a * Real.cos B + b * Real.cos A = 4 * Real.sin C) :
    π * (2 : ℝ) ^ 2 = 4 * π :=
by
  sorry

end NUMINAMATH_GPT_circumcircle_area_l1616_161674


namespace NUMINAMATH_GPT_find_number_of_pairs_l1616_161609

variable (n : ℕ)
variable (prob_same_color : ℚ := 0.09090909090909091)
variable (total_shoes : ℕ := 12)
variable (pairs_of_shoes : ℕ)

-- The condition on the probability of selecting two shoes of the same color
def condition_probability : Prop :=
  (1 : ℚ) / ((2 * n - 1) : ℚ) = prob_same_color

-- The condition on the total number of shoes
def condition_total_shoes : Prop :=
  2 * n = total_shoes

-- The goal to prove that the number of pairs of shoes is 6 given the conditions
theorem find_number_of_pairs (h1 : condition_probability n) (h2 : condition_total_shoes n) : n = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_pairs_l1616_161609


namespace NUMINAMATH_GPT_library_books_difference_l1616_161641

theorem library_books_difference (total_books : ℕ) (borrowed_percentage : ℕ) 
  (initial_books : total_books = 400) 
  (percentage_borrowed : borrowed_percentage = 30) :
  (total_books - (borrowed_percentage * total_books / 100)) = 280 :=
by
  sorry

end NUMINAMATH_GPT_library_books_difference_l1616_161641


namespace NUMINAMATH_GPT_average_speed_lila_l1616_161636

-- Definitions
def distance1 : ℝ := 50 -- miles
def speed1 : ℝ := 20 -- miles per hour
def distance2 : ℝ := 20 -- miles
def speed2 : ℝ := 40 -- miles per hour
def break_time : ℝ := 0.5 -- hours

-- Question to prove: Lila's average speed for the entire ride is 20 miles per hour
theorem average_speed_lila (d1 d2 s1 s2 bt : ℝ) 
  (h1 : d1 = distance1) (h2 : s1 = speed1) (h3 : d2 = distance2) (h4 : s2 = speed2) (h5 : bt = break_time) :
  (d1 + d2) / (d1 / s1 + d2 / s2 + bt) = 20 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_lila_l1616_161636


namespace NUMINAMATH_GPT_percentage_cut_second_week_l1616_161659

noncomputable def calculate_final_weight (initial_weight : ℝ) (percentage1 : ℝ) (percentage2 : ℝ) (percentage3 : ℝ) : ℝ :=
  let weight_after_first_week := (1 - percentage1 / 100) * initial_weight
  let weight_after_second_week := (1 - percentage2 / 100) * weight_after_first_week
  let final_weight := (1 - percentage3 / 100) * weight_after_second_week
  final_weight

theorem percentage_cut_second_week : 
  ∀ (initial_weight : ℝ) (final_weight : ℝ), (initial_weight = 250) → (final_weight = 105) →
    (calculate_final_weight initial_weight 30 x 25 = final_weight) → 
    x = 20 := 
by 
  intros initial_weight final_weight h1 h2 h3
  sorry

end NUMINAMATH_GPT_percentage_cut_second_week_l1616_161659


namespace NUMINAMATH_GPT_product_of_roots_l1616_161607

noncomputable def is_root (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

theorem product_of_roots :
  ∀ (x1 x2 : ℝ), is_root 1 (-4) 3 x1 ∧ is_root 1 (-4) 3 x2 ∧ x1 ≠ x2 → x1 * x2 = 3 :=
by
  intros x1 x2 h
  sorry

end NUMINAMATH_GPT_product_of_roots_l1616_161607
