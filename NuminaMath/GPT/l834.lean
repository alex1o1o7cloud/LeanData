import Mathlib

namespace NUMINAMATH_GPT_quadratic_transformation_l834_83445

noncomputable def transform_roots (p q r : ℚ) (u v : ℚ) 
    (h1 : u + v = -q / p) 
    (h2 : u * v = r / p) : Prop :=
  ∃ y : ℚ, y^2 - q^2 + 4 * p * r = 0

theorem quadratic_transformation (p q r u v : ℚ) 
    (h1 : u + v = -q / p) 
    (h2 : u * v = r / p) :
  ∃ y : ℚ, (y - (2 * p * u + q)) * (y - (2 * p * v + q)) = y^2 - q^2 + 4 * p * r :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_transformation_l834_83445


namespace NUMINAMATH_GPT_possible_values_of_ABCD_l834_83447

noncomputable def discriminant (a b c : ℕ) : ℕ :=
  b^2 - 4*a*c

theorem possible_values_of_ABCD 
  (A B C D : ℕ)
  (AB BC CD : ℕ)
  (hAB : AB = 10*A + B)
  (hBC : BC = 10*B + C)
  (hCD : CD = 10*C + D)
  (h_no_9 : A ≠ 9 ∧ B ≠ 9 ∧ C ≠ 9 ∧ D ≠ 9)
  (h_leading_nonzero : A ≠ 0)
  (h_quad1 : discriminant A B CD ≥ 0)
  (h_quad2 : discriminant A BC D ≥ 0)
  (h_quad3 : discriminant AB C D ≥ 0) :
  ABCD = 1710 ∨ ABCD = 1810 :=
sorry

end NUMINAMATH_GPT_possible_values_of_ABCD_l834_83447


namespace NUMINAMATH_GPT_range_of_a_l834_83410

variable (a : ℝ)
def f (x : ℝ) := x^2 + 2 * (a - 1) * x + 2
def f_deriv (x : ℝ) := 2 * x + 2 * (a - 1)

theorem range_of_a (h : ∀ x ≥ -4, f_deriv a x ≥ 0) : a ≥ 5 :=
sorry

end NUMINAMATH_GPT_range_of_a_l834_83410


namespace NUMINAMATH_GPT_smallest_number_with_sum_32_and_distinct_digits_l834_83493

def is_distinct (l : List Nat) : Prop := l.Nodup

def sum_of_digits (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

theorem smallest_number_with_sum_32_and_distinct_digits :
  ∀ n : Nat, is_distinct (Nat.digits 10 n) ∧ sum_of_digits n = 32 → 26789 ≤ n :=
by
  intro n
  intro h
  sorry

end NUMINAMATH_GPT_smallest_number_with_sum_32_and_distinct_digits_l834_83493


namespace NUMINAMATH_GPT_solve_for_b_l834_83427

theorem solve_for_b (b : ℚ) (h : b + b / 4 = 5 / 2) : b = 2 := 
sorry

end NUMINAMATH_GPT_solve_for_b_l834_83427


namespace NUMINAMATH_GPT_max_product_of_two_integers_sum_2000_l834_83468

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end NUMINAMATH_GPT_max_product_of_two_integers_sum_2000_l834_83468


namespace NUMINAMATH_GPT_green_ball_probability_l834_83437

/-
  There are four containers:
  - Container A holds 5 red balls and 7 green balls.
  - Container B holds 7 red balls and 3 green balls.
  - Container C holds 8 red balls and 2 green balls.
  - Container D holds 4 red balls and 6 green balls.
  The probability of choosing containers A, B, C, and D is 1/4 each.
-/

def prob_A : ℚ := 1 / 4
def prob_B : ℚ := 1 / 4
def prob_C : ℚ := 1 / 4
def prob_D : ℚ := 1 / 4

def prob_Given_A : ℚ := 7 / 12
def prob_Given_B : ℚ := 3 / 10
def prob_Given_C : ℚ := 1 / 5
def prob_Given_D : ℚ := 3 / 5

def total_prob_green : ℚ :=
  prob_A * prob_Given_A + prob_B * prob_Given_B +
  prob_C * prob_Given_C + prob_D * prob_Given_D

theorem green_ball_probability : total_prob_green = 101 / 240 := 
by
  -- here would normally be the proof steps, but we use sorry to skip it.
  sorry

end NUMINAMATH_GPT_green_ball_probability_l834_83437


namespace NUMINAMATH_GPT_fourth_person_height_l834_83412

-- Definitions based on conditions
def h1 : ℕ := 73  -- height of first person
def h2 : ℕ := h1 + 2  -- height of second person
def h3 : ℕ := h2 + 2  -- height of third person
def h4 : ℕ := h3 + 6  -- height of fourth person

theorem fourth_person_height : h4 = 83 :=
by
  -- calculation to check the average height and arriving at h1
  -- (all detailed calculations are skipped using "sorry")
  sorry

end NUMINAMATH_GPT_fourth_person_height_l834_83412


namespace NUMINAMATH_GPT_probability_of_c_between_l834_83449

noncomputable def probability_c_between (a b : ℝ) (hab : 0 < a ∧ a ≤ 1 ∧ 0 < b ∧ b ≤ 1) : ℝ :=
  let c := a / (a + b)
  if (1 / 4 : ℝ) ≤ c ∧ c ≤ (3 / 4 : ℝ) then sorry else sorry
  
theorem probability_of_c_between (a b : ℝ) (hab : 0 < a ∧ a ≤ 1 ∧ 0 < b ∧ b ≤ 1) : 
  probability_c_between a b hab = (2 / 3 : ℝ) :=
sorry

end NUMINAMATH_GPT_probability_of_c_between_l834_83449


namespace NUMINAMATH_GPT_expected_value_of_third_flip_l834_83416

-- Definitions for the conditions
def prob_heads : ℚ := 2/5
def prob_tails : ℚ := 3/5
def win_amount : ℚ := 4
def base_loss : ℚ := 3
def doubled_loss : ℚ := 2 * base_loss
def first_two_flips_were_tails : Prop := true 

-- The main statement: Proving the expected value of the third flip
theorem expected_value_of_third_flip (h : first_two_flips_were_tails) : 
  (prob_heads * win_amount + prob_tails * -doubled_loss) = -2 := by
  sorry

end NUMINAMATH_GPT_expected_value_of_third_flip_l834_83416


namespace NUMINAMATH_GPT_minimum_distance_on_line_l834_83401

-- Define the line as a predicate
def on_line (P : ℝ × ℝ) : Prop := P.1 - P.2 = 1

-- Define the expression to be minimized
def distance_squared (P : ℝ × ℝ) : ℝ := (P.1 - 2)^2 + (P.2 - 2)^2

theorem minimum_distance_on_line :
  ∃ P : ℝ × ℝ, on_line P ∧ distance_squared P = 1 / 2 :=
sorry

end NUMINAMATH_GPT_minimum_distance_on_line_l834_83401


namespace NUMINAMATH_GPT_distance_on_third_day_is_36_difference_between_longest_and_shortest_is_57_average_daily_distance_is_50_l834_83452

-- Definitions for each day's recorded distance deviation
def day_1_distance := -8
def day_2_distance := -11
def day_3_distance := -14
def day_4_distance := 0
def day_5_distance := 8
def day_6_distance := 41
def day_7_distance := -16

-- Parameters and conditions
def actual_distance (recorded: Int) : Int := 50 + recorded

noncomputable def distance_3rd_day : Int := actual_distance day_3_distance
noncomputable def longest_distance : Int :=
    max (max (max (day_1_distance) (day_2_distance)) (max (day_3_distance) (day_4_distance)))
        (max (max (day_5_distance) (day_6_distance)) (day_7_distance))
noncomputable def shortest_distance : Int :=
    min (min (min (day_1_distance) (day_2_distance)) (min (day_3_distance) (day_4_distance)))
        (min (min (day_5_distance) (day_6_distance)) (day_7_distance))
noncomputable def average_distance : Int :=
    50 + (day_1_distance + day_2_distance + day_3_distance + day_4_distance +
          day_5_distance + day_6_distance + day_7_distance) / 7

-- Theorems to prove each part of the problem
theorem distance_on_third_day_is_36 : distance_3rd_day = 36 := by
  sorry

theorem difference_between_longest_and_shortest_is_57 : 
  (actual_distance longest_distance - actual_distance shortest_distance) = 57 := by
  sorry

theorem average_daily_distance_is_50 : average_distance = 50 := by
  sorry

end NUMINAMATH_GPT_distance_on_third_day_is_36_difference_between_longest_and_shortest_is_57_average_daily_distance_is_50_l834_83452


namespace NUMINAMATH_GPT_solve_inequality_system_l834_83474

theorem solve_inequality_system
  (x : ℝ)
  (h1 : 3 * (x - 1) < 5 * x + 11)
  (h2 : 2 * x > (9 - x) / 4) :
  x > 1 :=
sorry

end NUMINAMATH_GPT_solve_inequality_system_l834_83474


namespace NUMINAMATH_GPT_total_amount_paid_l834_83428

-- Define the conditions
def chicken_nuggets_ordered : ℕ := 100
def nuggets_per_box : ℕ := 20
def cost_per_box : ℕ := 4

-- Define the hypothesis on the amount of money paid for the chicken nuggets
theorem total_amount_paid :
  (chicken_nuggets_ordered / nuggets_per_box) * cost_per_box = 20 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_paid_l834_83428


namespace NUMINAMATH_GPT_find_a_plus_b_l834_83488

theorem find_a_plus_b (x a b : ℝ) (ha : x = a + Real.sqrt b)
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : x^2 + 5 * x + 4/x + 1/(x^2) = 34) : a + b = 5 :=
sorry

end NUMINAMATH_GPT_find_a_plus_b_l834_83488


namespace NUMINAMATH_GPT_positive_real_number_solution_l834_83438

theorem positive_real_number_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 11) (h3 : (x - 6) / 11 = 6 / (x - 11)) : x = 17 :=
sorry

end NUMINAMATH_GPT_positive_real_number_solution_l834_83438


namespace NUMINAMATH_GPT_find_unknown_value_l834_83454

theorem find_unknown_value (x : ℝ) (h : (3 + 5 + 6 + 8 + x) / 5 = 7) : x = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_unknown_value_l834_83454


namespace NUMINAMATH_GPT_girls_with_brown_eyes_and_light_brown_skin_l834_83495

theorem girls_with_brown_eyes_and_light_brown_skin 
  (total_girls : ℕ)
  (light_brown_skin_girls : ℕ)
  (blue_eyes_fair_skin_girls : ℕ)
  (brown_eyes_total : ℕ)
  (total_girls_50 : total_girls = 50)
  (light_brown_skin_31 : light_brown_skin_girls = 31)
  (blue_eyes_fair_skin_14 : blue_eyes_fair_skin_girls = 14)
  (brown_eyes_18 : brown_eyes_total = 18) :
  ∃ (brown_eyes_light_brown_skin_girls : ℕ), brown_eyes_light_brown_skin_girls = 13 :=
by sorry

end NUMINAMATH_GPT_girls_with_brown_eyes_and_light_brown_skin_l834_83495


namespace NUMINAMATH_GPT_sean_whistles_l834_83489

def charles_whistles : ℕ := 128
def sean_more_whistles : ℕ := 95

theorem sean_whistles : charles_whistles + sean_more_whistles = 223 :=
by {
  sorry
}

end NUMINAMATH_GPT_sean_whistles_l834_83489


namespace NUMINAMATH_GPT_odd_power_sum_divisible_l834_83458

theorem odd_power_sum_divisible (x y : ℤ) (n : ℕ) (h_odd : ∃ k : ℕ, n = 2 * k + 1) :
  (x ^ n + y ^ n) % (x + y) = 0 := 
sorry

end NUMINAMATH_GPT_odd_power_sum_divisible_l834_83458


namespace NUMINAMATH_GPT_sum_of_two_digit_integers_l834_83462

theorem sum_of_two_digit_integers :
  let a := 10
  let l := 99
  let d := 1
  let n := (l - a) / d + 1
  let S := n * (a + l) / 2
  S = 4905 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_two_digit_integers_l834_83462


namespace NUMINAMATH_GPT_weight_of_B_l834_83455

theorem weight_of_B (A B C : ℝ) (h1 : (A + B + C) / 3 = 45) (h2 : (A + B) / 2 = 40) (h3 : (B + C) / 2 = 46) : B = 37 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_B_l834_83455


namespace NUMINAMATH_GPT_work_completion_days_l834_83487

theorem work_completion_days (Dx : ℕ) (Dy : ℕ) (days_y_worked : ℕ) (days_x_finished_remaining : ℕ)
  (work_rate_y : ℝ) (work_rate_x : ℝ) 
  (h1 : Dy = 24)
  (h2 : days_y_worked = 12)
  (h3 : days_x_finished_remaining = 18)
  (h4 : work_rate_y = 1 / Dy)
  (h5 : 12 * work_rate_y = 1 / 2)
  (h6 : work_rate_x = 1 / (2 * days_x_finished_remaining))
  (h7 : Dx * work_rate_x = 1) : Dx = 36 := sorry

end NUMINAMATH_GPT_work_completion_days_l834_83487


namespace NUMINAMATH_GPT_jason_games_planned_last_month_l834_83469

-- Define the conditions
variable (games_planned_this_month : Nat) (games_missed : Nat) (games_attended : Nat)

-- Define what we want to prove
theorem jason_games_planned_last_month (h1 : games_planned_this_month = 11)
                                        (h2 : games_missed = 16)
                                        (h3 : games_attended = 12) :
                                        (games_attended + games_missed - games_planned_this_month = 17) := 
by
  sorry

end NUMINAMATH_GPT_jason_games_planned_last_month_l834_83469


namespace NUMINAMATH_GPT_train_pass_man_time_l834_83441

/--
Prove that the train, moving at 120 kmph, passes a man running at 10 kmph in the opposite direction in approximately 13.85 seconds, given the train is 500 meters long.
-/
theorem train_pass_man_time (length_of_train : ℝ) (speed_of_train : ℝ) (speed_of_man : ℝ) : 
  length_of_train = 500 →
  speed_of_train = 120 →
  speed_of_man = 10 →
  abs ((500 / ((speed_of_train + speed_of_man) * 1000 / 3600)) - 13.85) < 0.01 :=
by
  intro h1 h2 h3
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_train_pass_man_time_l834_83441


namespace NUMINAMATH_GPT_mail_sorting_time_l834_83413

theorem mail_sorting_time :
  (1 / (1 / 3 + 1 / 6) = 2) :=
by
  sorry

end NUMINAMATH_GPT_mail_sorting_time_l834_83413


namespace NUMINAMATH_GPT_people_in_room_l834_83442

theorem people_in_room (P C : ℚ) (H1 : (3 / 5) * P = (2 / 3) * C) (H2 : C / 3 = 5) : 
  P = 50 / 3 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_people_in_room_l834_83442


namespace NUMINAMATH_GPT_shanille_probability_l834_83448

-- Defining the probability function according to the problem's conditions.
def hit_probability (n k : ℕ) : ℚ :=
  if n = 100 ∧ k = 50 then 1 / 99 else 0

-- Prove that the probability Shanille hits exactly 50 of her first 100 shots is 1/99.
theorem shanille_probability :
  hit_probability 100 50 = 1 / 99 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_shanille_probability_l834_83448


namespace NUMINAMATH_GPT_tan_neg_585_eq_neg_1_l834_83432

theorem tan_neg_585_eq_neg_1 : Real.tan (-585 * Real.pi / 180) = -1 := by
  sorry

end NUMINAMATH_GPT_tan_neg_585_eq_neg_1_l834_83432


namespace NUMINAMATH_GPT_dubblefud_red_balls_l834_83497

theorem dubblefud_red_balls (R B G : ℕ) 
  (h1 : 3^R * 7^B * 11^G = 5764801)
  (h2 : B = G) :
  R = 7 :=
by
  sorry

end NUMINAMATH_GPT_dubblefud_red_balls_l834_83497


namespace NUMINAMATH_GPT_abs_neg_four_minus_six_l834_83409

theorem abs_neg_four_minus_six : abs (-4 - 6) = 10 := 
by
  sorry

end NUMINAMATH_GPT_abs_neg_four_minus_six_l834_83409


namespace NUMINAMATH_GPT_books_before_grant_l834_83418

-- Define the conditions 
def books_purchased_with_grant : ℕ := 2647
def total_books_now : ℕ := 8582

-- Prove the number of books before the grant
theorem books_before_grant : 
  (total_books_now - books_purchased_with_grant = 5935) := 
by
  sorry

end NUMINAMATH_GPT_books_before_grant_l834_83418


namespace NUMINAMATH_GPT_fuel_remaining_l834_83457

-- Definitions given in the conditions of the original problem
def initial_fuel : ℕ := 48
def fuel_consumption_rate : ℕ := 8

-- Lean 4 statement of the mathematical proof problem
theorem fuel_remaining (x : ℕ) : 
  ∃ y : ℕ, y = initial_fuel - fuel_consumption_rate * x :=
sorry

end NUMINAMATH_GPT_fuel_remaining_l834_83457


namespace NUMINAMATH_GPT_giant_kite_area_72_l834_83496

-- Definition of the vertices of the medium kite
def vertices_medium_kite : List (ℕ × ℕ) := [(1,6), (4,9), (7,6), (4,1)]

-- Given condition function to check if the giant kite is created by doubling the height and width
def double_coordinates (c : (ℕ × ℕ)) : (ℕ × ℕ) := (2 * c.1, 2 * c.2)

def vertices_giant_kite : List (ℕ × ℕ) := vertices_medium_kite.map double_coordinates

-- Function to calculate the area of the kite based on its vertices
def kite_area (vertices : List (ℕ × ℕ)) : ℕ := sorry -- The way to calculate the kite area can be complex

-- Theorem to prove the area of the giant kite
theorem giant_kite_area_72 :
  kite_area vertices_giant_kite = 72 := 
sorry

end NUMINAMATH_GPT_giant_kite_area_72_l834_83496


namespace NUMINAMATH_GPT_smallest_nat_satisfies_conditions_l834_83407

theorem smallest_nat_satisfies_conditions : 
  ∃ x : ℕ, (∃ m : ℤ, x + 13 = 5 * m) ∧ (∃ n : ℤ, x - 13 = 6 * n) ∧ x = 37 := by
  sorry

end NUMINAMATH_GPT_smallest_nat_satisfies_conditions_l834_83407


namespace NUMINAMATH_GPT_sodium_acetate_formed_is_3_l834_83490

-- Definitions for chemicals involved in the reaction
def AceticAcid : Type := ℕ -- Number of moles of acetic acid
def SodiumHydroxide : Type := ℕ -- Number of moles of sodium hydroxide
def SodiumAcetate : Type := ℕ -- Number of moles of sodium acetate

-- Given conditions as definitions
def reaction (acetic_acid naoh : ℕ) : ℕ :=
  if acetic_acid = naoh then acetic_acid else min acetic_acid naoh

-- Lean theorem statement
theorem sodium_acetate_formed_is_3 
  (acetic_acid naoh : ℕ) 
  (h1 : acetic_acid = 3) 
  (h2 : naoh = 3) :
  reaction acetic_acid naoh = 3 :=
by
  -- Proof body (to be completed)
  sorry

end NUMINAMATH_GPT_sodium_acetate_formed_is_3_l834_83490


namespace NUMINAMATH_GPT_find_m_l834_83461

theorem find_m (x y m : ℝ) (h₁ : x - 2 * y = m) (h₂ : x = 2) (h₃ : y = 1) : m = 0 :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_find_m_l834_83461


namespace NUMINAMATH_GPT_xiaoli_estimate_larger_l834_83450

variable (x y z w : ℝ)
variable (hxy : x > y) (hy0 : y > 0) (hz1 : z > 1) (hw0 : w > 0)

theorem xiaoli_estimate_larger : (x + w) - (y - w) * z > x - y * z :=
by sorry

end NUMINAMATH_GPT_xiaoli_estimate_larger_l834_83450


namespace NUMINAMATH_GPT_time_to_finish_typing_l834_83426

-- Definitions
def words_per_minute : ℕ := 38
def total_words : ℕ := 4560

-- Theorem to prove
theorem time_to_finish_typing : (total_words / words_per_minute) / 60 = 2 := by
  sorry

end NUMINAMATH_GPT_time_to_finish_typing_l834_83426


namespace NUMINAMATH_GPT_Nadine_pebbles_l834_83446

theorem Nadine_pebbles :
  ∀ (white red blue green x : ℕ),
    white = 20 →
    red = white / 2 →
    blue = red / 3 →
    green = blue + 5 →
    red = (1/5) * x →
    x = 50 :=
by
  intros white red blue green x h_white h_red h_blue h_green h_percentage
  sorry

end NUMINAMATH_GPT_Nadine_pebbles_l834_83446


namespace NUMINAMATH_GPT_solution_set_eq_l834_83456

noncomputable def f (x : ℝ) : ℝ := x^6 + x^2
noncomputable def g (x : ℝ) : ℝ := (2*x + 3)^3 + 2*x + 3

theorem solution_set_eq : {x : ℝ | f x = g x} = {-1, 3} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_eq_l834_83456


namespace NUMINAMATH_GPT_geometric_sequence_solution_l834_83414

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) :=
  ∀ n, a n = a1 * q ^ (n - 1)

theorem geometric_sequence_solution {a : ℕ → ℝ} {q a1 : ℝ}
  (h1 : geometric_sequence a q a1)
  (h2 : a 3 + a 5 = 20)
  (h3 : a 4 = 8) :
  a 2 + a 6 = 34 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_solution_l834_83414


namespace NUMINAMATH_GPT_bob_grade_is_35_l834_83466

-- Define the conditions
def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

-- State the theorem
theorem bob_grade_is_35 : bob_grade = 35 := by
  sorry

end NUMINAMATH_GPT_bob_grade_is_35_l834_83466


namespace NUMINAMATH_GPT_factorization_of_x12_minus_4096_l834_83420

variable (x : ℝ)

theorem factorization_of_x12_minus_4096 : 
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end NUMINAMATH_GPT_factorization_of_x12_minus_4096_l834_83420


namespace NUMINAMATH_GPT_unit_digit_23_pow_100000_l834_83403

theorem unit_digit_23_pow_100000 : (23^100000) % 10 = 1 := 
by
  -- Import necessary submodules and definitions

sorry

end NUMINAMATH_GPT_unit_digit_23_pow_100000_l834_83403


namespace NUMINAMATH_GPT_compare_abc_l834_83480

/-- Define the constants a, b, and c as given in the problem -/
noncomputable def a : ℝ := -5 / 4 * Real.log (4 / 5)
noncomputable def b : ℝ := Real.exp (1 / 4) / 4
noncomputable def c : ℝ := 1 / 3

/-- The theorem to be proved: a < b < c -/
theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end NUMINAMATH_GPT_compare_abc_l834_83480


namespace NUMINAMATH_GPT_farmer_field_l834_83492

theorem farmer_field (m : ℤ) : 
  (3 * m + 8) * (m - 3) = 85 → m = 6 :=
by
  sorry

end NUMINAMATH_GPT_farmer_field_l834_83492


namespace NUMINAMATH_GPT_percent_of_games_lost_l834_83453

theorem percent_of_games_lost (w l : ℕ) (h1 : w / l = 8 / 5) (h2 : w + l = 65) :
  (l * 100 / 65 : ℕ) = 38 :=
sorry

end NUMINAMATH_GPT_percent_of_games_lost_l834_83453


namespace NUMINAMATH_GPT_movies_in_first_box_l834_83405

theorem movies_in_first_box (x : ℕ) 
  (cost_first : ℕ) (cost_second : ℕ) 
  (num_second : ℕ) (avg_price : ℕ)
  (h_cost_first : cost_first = 2)
  (h_cost_second : cost_second = 5)
  (h_num_second : num_second = 5)
  (h_avg_price : avg_price = 3)
  (h_total_eq : cost_first * x + cost_second * num_second = avg_price * (x + num_second)) :
  x = 5 :=
by
  sorry

end NUMINAMATH_GPT_movies_in_first_box_l834_83405


namespace NUMINAMATH_GPT_find_multiple_of_larger_integer_l834_83408

/--
The sum of two integers is 30. A certain multiple of the larger integer is 10 less than 5 times
the smaller integer. The smaller integer is 10. What is the multiple of the larger integer?
-/
theorem find_multiple_of_larger_integer
  (S L M : ℤ)
  (h1 : S + L = 30)
  (h2 : S = 10)
  (h3 : M * L = 5 * S - 10) :
  M = 2 :=
sorry

end NUMINAMATH_GPT_find_multiple_of_larger_integer_l834_83408


namespace NUMINAMATH_GPT_factorize_expression_l834_83460

variable (x y : ℝ)

theorem factorize_expression : 
  (y - 2 * x * y + x^2 * y) = y * (1 - x)^2 := 
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l834_83460


namespace NUMINAMATH_GPT_correct_operation_l834_83430

theorem correct_operation : (3 * a^2 * b^3 - 2 * a^2 * b^3 = a^2 * b^3) ∧ 
                            ¬(a^2 * a^3 = a^6) ∧ 
                            ¬(a^6 / a^2 = a^3) ∧ 
                            ¬((a^2)^3 = a^5) :=
by
  sorry

end NUMINAMATH_GPT_correct_operation_l834_83430


namespace NUMINAMATH_GPT_hannah_probability_12_flips_l834_83440

/-!
We need to prove that the probability of getting fewer than 4 heads when flipping 12 coins is 299/4096.
-/

def probability_fewer_than_4_heads (flips : ℕ) : ℚ :=
  let total_outcomes := 2^flips
  let favorable_outcomes := (Nat.choose flips 0) + (Nat.choose flips 1) + (Nat.choose flips 2) + (Nat.choose flips 3)
  favorable_outcomes / total_outcomes

theorem hannah_probability_12_flips : probability_fewer_than_4_heads 12 = 299 / 4096 := by
  sorry

end NUMINAMATH_GPT_hannah_probability_12_flips_l834_83440


namespace NUMINAMATH_GPT_annes_initial_bottle_caps_l834_83479

-- Define the conditions
def albert_bottle_caps : ℕ := 9
def annes_added_bottle_caps : ℕ := 5
def annes_total_bottle_caps : ℕ := 15

-- Question (to prove)
theorem annes_initial_bottle_caps :
  annes_total_bottle_caps - annes_added_bottle_caps = 10 :=
by sorry

end NUMINAMATH_GPT_annes_initial_bottle_caps_l834_83479


namespace NUMINAMATH_GPT_moving_circle_passes_through_fixed_point_l834_83486
-- We will start by importing the necessary libraries and setting up the problem conditions.

-- Define the parabola y^2 = 8x.
def parabola (p : ℝ × ℝ) : Prop :=
  p.2 ^ 2 = 8 * p.1

-- Define the line x + 2 = 0.
def tangent_line (p : ℝ × ℝ) : Prop :=
  p.1 = -2

-- Define the fixed point.
def fixed_point : ℝ × ℝ :=
  (2, 0)

-- Define the moving circle passing through the fixed point.
def moving_circle (p : ℝ × ℝ) (c : ℝ × ℝ) :=
  p = fixed_point

-- Bring it all together in the theorem.
theorem moving_circle_passes_through_fixed_point (c : ℝ × ℝ) (p : ℝ × ℝ)
  (h_parabola : parabola c)
  (h_tangent : tangent_line p) :
  moving_circle p c :=
sorry

end NUMINAMATH_GPT_moving_circle_passes_through_fixed_point_l834_83486


namespace NUMINAMATH_GPT_max_area_rectangle_shorter_side_l834_83419

theorem max_area_rectangle_shorter_side (side_length : ℕ) (n : ℕ)
  (hsq : side_length = 40) (hn : n = 5) :
  ∃ (shorter_side : ℕ), shorter_side = 8 := by
  sorry

end NUMINAMATH_GPT_max_area_rectangle_shorter_side_l834_83419


namespace NUMINAMATH_GPT_inequality_proof_l834_83470

theorem inequality_proof (x : ℝ) (hx : x ≥ 1) : x^5 - 1 / x^4 ≥ 9 * (x - 1) := 
by sorry

end NUMINAMATH_GPT_inequality_proof_l834_83470


namespace NUMINAMATH_GPT_singleBase12Digit_l834_83425

theorem singleBase12Digit (n : ℕ) : 
  (7 ^ 6 ^ 5 ^ 3 ^ 2 ^ 1) % 11 = 4 :=
sorry

end NUMINAMATH_GPT_singleBase12Digit_l834_83425


namespace NUMINAMATH_GPT_constant_term_g_eq_l834_83400

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := f * g

theorem constant_term_g_eq : 
  (h.coeff 0 = 2) ∧ (f.coeff 0 = -6) →  g.coeff 0 = -1/3 := by
  sorry

end NUMINAMATH_GPT_constant_term_g_eq_l834_83400


namespace NUMINAMATH_GPT_solve_for_x_l834_83422

theorem solve_for_x (x : ℝ) (h : 3 - (1 / (2 - x)) = (1 / (2 - x))) : x = 4 / 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l834_83422


namespace NUMINAMATH_GPT_ratio_small_to_large_is_one_to_one_l834_83451

theorem ratio_small_to_large_is_one_to_one
  (total_beads : ℕ)
  (large_beads_per_bracelet : ℕ)
  (bracelets_count : ℕ)
  (small_beads : ℕ)
  (large_beads : ℕ)
  (small_beads_per_bracelet : ℕ) :
  total_beads = 528 →
  large_beads_per_bracelet = 12 →
  bracelets_count = 11 →
  large_beads = total_beads / 2 →
  large_beads >= bracelets_count * large_beads_per_bracelet →
  small_beads = total_beads / 2 →
  small_beads_per_bracelet = small_beads / bracelets_count →
  small_beads_per_bracelet / large_beads_per_bracelet = 1 :=
by sorry

end NUMINAMATH_GPT_ratio_small_to_large_is_one_to_one_l834_83451


namespace NUMINAMATH_GPT_samuel_teacups_left_l834_83421

-- Define the initial conditions
def total_boxes := 60
def pans_boxes := 12
def decoration_fraction := 1 / 4
def decoration_trade := 3
def trade_gain := 1
def teacups_per_box := 6 * 4 * 2
def broken_per_pickup := 4

-- Calculate the number of boxes initially containing teacups
def remaining_boxes := total_boxes - pans_boxes
def decoration_boxes := decoration_fraction * remaining_boxes
def initial_teacup_boxes := remaining_boxes - decoration_boxes

-- Adjust the number of teacup boxes after the trade
def teacup_boxes := initial_teacup_boxes + trade_gain

-- Calculate total number of teacups and the number of teacups broken
def total_teacups := teacup_boxes * teacups_per_box
def total_broken := teacup_boxes * broken_per_pickup

-- Calculate the number of teacups left
def teacups_left := total_teacups - total_broken

-- State the theorem
theorem samuel_teacups_left : teacups_left = 1628 := by
  sorry

end NUMINAMATH_GPT_samuel_teacups_left_l834_83421


namespace NUMINAMATH_GPT_total_fish_is_22_l834_83424

def gold_fish : ℕ := 15
def blue_fish : ℕ := 7
def total_fish : ℕ := gold_fish + blue_fish

theorem total_fish_is_22 : total_fish = 22 :=
by
  -- the proof should be written here
  sorry

end NUMINAMATH_GPT_total_fish_is_22_l834_83424


namespace NUMINAMATH_GPT_find_a_n_geo_b_find_S_2n_l834_83429
noncomputable def S : ℕ → ℚ
| n => (n^2 + n + 1) / 2

def a (n : ℕ) : ℚ :=
  if n = 1 then 3/2
  else n

theorem find_a_n (n : ℕ) : a n = if n = 1 then 3/2 else n :=
by
  sorry

def b (n : ℕ) : ℚ :=
  a (2 * n - 1) + a (2 * n)

theorem geo_b (n : ℕ) : b (n + 1) = 3 * b n :=
by
  sorry

theorem find_S_2n (n : ℕ) : S (2 * n) = 3/2 * (3^n - 1) :=
by
  sorry

end NUMINAMATH_GPT_find_a_n_geo_b_find_S_2n_l834_83429


namespace NUMINAMATH_GPT_discount_amount_correct_l834_83498

noncomputable def cost_price : ℕ := 180
noncomputable def markup_percentage : ℝ := 0.45
noncomputable def profit_percentage : ℝ := 0.20

theorem discount_amount_correct : 
  let markup := cost_price * markup_percentage
  let mp := cost_price + markup
  let profit := cost_price * profit_percentage
  let sp := cost_price + profit
  let discount_amount := mp - sp
  discount_amount = 45 :=
by
  sorry

end NUMINAMATH_GPT_discount_amount_correct_l834_83498


namespace NUMINAMATH_GPT_sticker_count_l834_83499

def stickers_per_page : ℕ := 25
def num_pages : ℕ := 35
def total_stickers : ℕ := 875

theorem sticker_count : num_pages * stickers_per_page = total_stickers :=
by {
  sorry
}

end NUMINAMATH_GPT_sticker_count_l834_83499


namespace NUMINAMATH_GPT_geometric_sequence_at_t_l834_83444

theorem geometric_sequence_at_t (a : ℕ → ℕ) (S : ℕ → ℕ) (t : ℕ) :
  (∀ n, a n = a 1 * (3 ^ (n - 1))) →
  a 1 = 1 →
  S t = (a 1 * (1 - 3 ^ t)) / (1 - 3) →
  S t = 364 →
  a t = 243 :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_at_t_l834_83444


namespace NUMINAMATH_GPT_halfway_fraction_l834_83465

-- Assume a definition for the two fractions
def fracA : ℚ := 1 / 4
def fracB : ℚ := 1 / 7

-- Define the target property we want to prove
theorem halfway_fraction : (fracA + fracB) / 2 = 11 / 56 := 
by 
  -- Proof will happen here, adding sorry to indicate it's skipped for now
  sorry

end NUMINAMATH_GPT_halfway_fraction_l834_83465


namespace NUMINAMATH_GPT_exists_unique_inverse_l834_83417

theorem exists_unique_inverse (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (h_gcd : Nat.gcd p a = 1) : 
  ∃! (b : ℕ), b ∈ Finset.range p ∧ (a * b) % p = 1 := 
sorry

end NUMINAMATH_GPT_exists_unique_inverse_l834_83417


namespace NUMINAMATH_GPT_f_inequality_l834_83406

noncomputable def f (a x : ℝ) : ℝ := a * (Real.exp x + a) - x

theorem f_inequality (a : ℝ) (h : a > 0) (x : ℝ) : 
  f a x > 2 * Real.log a + 3 / 2 := 
sorry 

end NUMINAMATH_GPT_f_inequality_l834_83406


namespace NUMINAMATH_GPT_mn_sum_eq_neg_one_l834_83415

theorem mn_sum_eq_neg_one (m n : ℤ) (h : (∀ x : ℤ, (x + 2) * (x - 1) = x^2 + m * x + n)) :
  m + n = -1 :=
sorry

end NUMINAMATH_GPT_mn_sum_eq_neg_one_l834_83415


namespace NUMINAMATH_GPT_proof_b_lt_a_lt_c_l834_83475

noncomputable def a : ℝ := 2^(4/5)
noncomputable def b : ℝ := 4^(2/7)
noncomputable def c : ℝ := 25^(1/5)

theorem proof_b_lt_a_lt_c : b < a ∧ a < c := by
  sorry

end NUMINAMATH_GPT_proof_b_lt_a_lt_c_l834_83475


namespace NUMINAMATH_GPT_higher_concentration_acid_solution_l834_83485

theorem higher_concentration_acid_solution (x : ℝ) (h1 : 2 * (8 / 100 : ℝ) = 1.2 * (x / 100) + 0.8 * (5 / 100)) : x = 10 :=
sorry

end NUMINAMATH_GPT_higher_concentration_acid_solution_l834_83485


namespace NUMINAMATH_GPT_susie_pizza_sales_l834_83436

theorem susie_pizza_sales :
  ∃ x : ℕ, 
    (24 * 3 + 15 * x = 117) ∧ 
    x = 3 := 
by
  sorry

end NUMINAMATH_GPT_susie_pizza_sales_l834_83436


namespace NUMINAMATH_GPT_sum_of_eight_digits_l834_83435

open Nat

theorem sum_of_eight_digits {a b c d e f g h : ℕ} 
  (h_distinct : ∀ i j, i ∈ [a, b, c, d, e, f, g, h] → j ∈ [a, b, c, d, e, f, g, h] → i ≠ j → i ≠ j)
  (h_vertical_sum : a + b + c + d + e = 25)
  (h_horizontal_sum : f + g + h + b = 15) 
  (h_digits_set : ∀ x ∈ [a, b, c, d, e, f, g, h], x ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ)) : 
  a + b + c + d + e + f + g + h - b = 39 := 
sorry

end NUMINAMATH_GPT_sum_of_eight_digits_l834_83435


namespace NUMINAMATH_GPT_sin_value_of_arithmetic_sequence_l834_83464

open Real

def arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d, ∀ n, a (n + 1) = a n + d

theorem sin_value_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith_seq : arithmetic_sequence a) 
  (h_cond : a 1 + a 5 + a 9 = 5 * π) : 
  sin (a 2 + a 8) = - (sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_sin_value_of_arithmetic_sequence_l834_83464


namespace NUMINAMATH_GPT_man_l834_83482

/-- A man can row downstream at the rate of 45 kmph.
    A man can row upstream at the rate of 23 kmph.
    The rate of current is 11 kmph.
    The man's rate in still water is 34 kmph. -/
theorem man's_rate_in_still_water
  (v c : ℕ)
  (h1 : v + c = 45)
  (h2 : v - c = 23)
  (h3 : c = 11) : v = 34 := by
  sorry

end NUMINAMATH_GPT_man_l834_83482


namespace NUMINAMATH_GPT_find_a_l834_83476

theorem find_a (f : ℝ → ℝ) (h1 : ∀ x, f (2^x) = x + 3) (h2 : f a = 5) : a = 4 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l834_83476


namespace NUMINAMATH_GPT_john_subtraction_number_l834_83491

theorem john_subtraction_number (a b : ℕ) (h1 : a = 40) (h2 : b = 1) :
  40^2 - ((2 * 40 * 1) - 1^2) = 39^2 :=
by
  -- sorry indicates the proof is skipped
  sorry

end NUMINAMATH_GPT_john_subtraction_number_l834_83491


namespace NUMINAMATH_GPT_expected_value_correct_l834_83483

-- Define the problem conditions
def num_balls : ℕ := 5

def prob_swapped_twice : ℚ := (2 / 25)
def prob_never_swapped : ℚ := (9 / 25)
def prob_original_position : ℚ := prob_swapped_twice + prob_never_swapped

-- Define the expected value calculation
def expected_num_in_original_position : ℚ :=
  num_balls * prob_original_position

-- Claim: The expected number of balls that occupy their original positions after two successive transpositions is 2.2.
theorem expected_value_correct :
  expected_num_in_original_position = 2.2 :=
sorry

end NUMINAMATH_GPT_expected_value_correct_l834_83483


namespace NUMINAMATH_GPT_g_inv_zero_solution_l834_83434

noncomputable def g (a b x : ℝ) : ℝ := 1 / (2 * a * x + b)

theorem g_inv_zero_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  g a b (g a b 0) = 0 ↔ g a b 0 = 1 / b :=
by
  sorry

end NUMINAMATH_GPT_g_inv_zero_solution_l834_83434


namespace NUMINAMATH_GPT_cells_remain_illuminated_l834_83467

-- The rect grid screen of size m × n with more than (m - 1)(n - 1) cells illuminated 
-- with the condition that in any 2 × 2 square if three cells are not illuminated, 
-- then the fourth cell also turns off eventually.
theorem cells_remain_illuminated 
  {m n : ℕ} 
  (h1 : ∃ k : ℕ, k > (m - 1) * (n - 1) ∧ k ≤ m * n) 
  (h2 : ∀ (i j : ℕ) (hiv : i < m - 1) (hjv : j < n - 1), 
    (∃ c1 c2 c3 c4 : ℕ, 
      c1 + c2 + c3 + c4 = 4 ∧ 
      (c1 = 1 ∨ c2 = 1 ∨ c3 = 1 ∨ c4 = 1) → 
      (c1 = 0 ∧ c2 = 0 ∧ c3 = 0 ∧ c4 = 0))) :
  ∃ (i j : ℕ) (hil : i < m) (hjl : j < n), true := sorry

end NUMINAMATH_GPT_cells_remain_illuminated_l834_83467


namespace NUMINAMATH_GPT_triangle_shape_l834_83463

open Real

noncomputable def triangle (a b c A B C S : ℝ) :=
  ∃ (a b c A B C S : ℝ),
    a = 2 * sqrt 3 ∧
    A = π / 3 ∧
    S = 2 * sqrt 3 ∧
    (S = (1 / 2) * b * c * sin A) ∧
    (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * cos A) ∧
    (b = 2 ∧ c = 4 ∨ b = 4 ∧ c = 2)

theorem triangle_shape (A B C : ℝ) (h : sin (C - B) = sin (2 * B) - sin A):
    (B = π / 2 ∨ C = B) :=
sorry

end NUMINAMATH_GPT_triangle_shape_l834_83463


namespace NUMINAMATH_GPT_Connor_spends_36_dollars_l834_83472

def tickets_cost := 10.00
def combo_meal_cost := 11.00
def candy_cost := 2.50
def total_cost := tickets_cost * 2 + combo_meal_cost + candy_cost * 2

theorem Connor_spends_36_dollars : total_cost = 36.00 := 
by 
  sorry

end NUMINAMATH_GPT_Connor_spends_36_dollars_l834_83472


namespace NUMINAMATH_GPT_rectangle_area_l834_83471

theorem rectangle_area (x : ℝ) (w : ℝ) (h : w^2 + (2 * w)^2 = x^2) : 
  2 * (w^2) = (2 / 5) * x^2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l834_83471


namespace NUMINAMATH_GPT_part1_part2_l834_83439

-- Definitions and conditions
variables (A B C : ℝ) (a b c : ℝ)
-- Given conditions
variable (h1 : sin C * sin (A - B) = sin B * sin (C - A))

-- Proof for part (1)
theorem part1 : 2 * a^2 = b^2 + c^2 :=
  sorry

-- Given further conditions for part (2)
variables (a_eq : a = 5) (cosA_eq : cos A = 25 / 31)

-- Proof for part (2)
theorem part2 : a + b + c = 14 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l834_83439


namespace NUMINAMATH_GPT_solution_set_of_abs_inequality_l834_83477

theorem solution_set_of_abs_inequality :
  {x : ℝ // |2 * x - 1| < 3} = {x : ℝ // -1 < x ∧ x < 2} :=
by sorry

end NUMINAMATH_GPT_solution_set_of_abs_inequality_l834_83477


namespace NUMINAMATH_GPT_april_total_earned_l834_83478

variable (r_price t_price d_price : ℕ)
variable (r_sold t_sold d_sold : ℕ)
variable (r_total t_total d_total : ℕ)

-- Define prices
def rose_price : ℕ := 4
def tulip_price : ℕ := 3
def daisy_price : ℕ := 2

-- Define quantities sold
def roses_sold : ℕ := 9
def tulips_sold : ℕ := 6
def daisies_sold : ℕ := 12

-- Define total money earned for each type of flower
def rose_total := roses_sold * rose_price
def tulip_total := tulips_sold * tulip_price
def daisy_total := daisies_sold * daisy_price

-- Define total money earned
def total_earned := rose_total + tulip_total + daisy_total

-- Statement to prove
theorem april_total_earned : total_earned = 78 :=
by sorry

end NUMINAMATH_GPT_april_total_earned_l834_83478


namespace NUMINAMATH_GPT_simplify_expression_l834_83459

noncomputable def a : ℝ := Real.sqrt 3 - 1

theorem simplify_expression : 
  ( (a - 1) / (a^2 - 2 * a + 1) / ( (a^2 + a) / (a^2 - 1) + 1 / (a - 1) ) = Real.sqrt 3 / 3 ) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l834_83459


namespace NUMINAMATH_GPT_sum_a1_a5_l834_83481

theorem sum_a1_a5 (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = n^2 + 1)
  (ha : ∀ n, a (n + 1) = S (n + 1) - S n) :
  a 1 + a 5 = 11 :=
sorry

end NUMINAMATH_GPT_sum_a1_a5_l834_83481


namespace NUMINAMATH_GPT_cost_of_pencils_and_notebooks_l834_83431

variable (P N : ℝ)

theorem cost_of_pencils_and_notebooks
  (h1 : 4 * P + 3 * N = 9600)
  (h2 : 2 * P + 2 * N = 5400) :
  8 * P + 7 * N = 20400 := by
  sorry

end NUMINAMATH_GPT_cost_of_pencils_and_notebooks_l834_83431


namespace NUMINAMATH_GPT_solution_set_inequality_l834_83423

theorem solution_set_inequality (x : ℝ) : (0 < x ∧ x < 1) ↔ (1 / (x - 1) < -1) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_inequality_l834_83423


namespace NUMINAMATH_GPT_nth_term_correct_l834_83402

noncomputable def nth_term (a b : ℝ) (n : ℕ) : ℝ :=
  (-1 : ℝ)^n * (2 * n - 1) * b / a^n

theorem nth_term_correct (a b : ℝ) (n : ℕ) (h : 0 < a) : 
  nth_term a b n = (-1 : ℝ)^↑n * (2 * n - 1) * b / a^n :=
by sorry

end NUMINAMATH_GPT_nth_term_correct_l834_83402


namespace NUMINAMATH_GPT_problem_solution_l834_83484

noncomputable def solveSystem : Prop :=
  ∃ (x1 x2 x3 x4 x5 x6 x7 x8 : ℝ),
    (x1 + x2 + x3 = 6) ∧
    (x2 + x3 + x4 = 9) ∧
    (x3 + x4 + x5 = 3) ∧
    (x4 + x5 + x6 = -3) ∧
    (x5 + x6 + x7 = -9) ∧
    (x6 + x7 + x8 = -6) ∧
    (x7 + x8 + x1 = -2) ∧
    (x8 + x1 + x2 = 2) ∧
    (x1 = 1) ∧
    (x2 = 2) ∧
    (x3 = 3) ∧
    (x4 = 4) ∧
    (x5 = -4) ∧
    (x6 = -3) ∧
    (x7 = -2) ∧
    (x8 = -1)

theorem problem_solution : solveSystem :=
by
  -- Skip the proof for now
  sorry

end NUMINAMATH_GPT_problem_solution_l834_83484


namespace NUMINAMATH_GPT_find_real_a_l834_83443

open Complex

noncomputable def pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem find_real_a (a : ℝ) (i : ℂ) (h_i : i = Complex.I) :
  pure_imaginary ((2 + i) * (a - (2 * i))) ↔ a = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_real_a_l834_83443


namespace NUMINAMATH_GPT_length_HD_is_3_l834_83433

noncomputable def square_side : ℝ := 8

noncomputable def midpoint_AD : ℝ := square_side / 2

noncomputable def length_FD : ℝ := midpoint_AD

theorem length_HD_is_3 :
  ∃ (x : ℝ), 0 < x ∧ x < square_side ∧ (8 - x) ^ 2 = x ^ 2 + length_FD ^ 2 ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_length_HD_is_3_l834_83433


namespace NUMINAMATH_GPT_has_minimum_value_iff_l834_83411

noncomputable def f (a x : ℝ) : ℝ :=
if x < a then -a * x + 4 else (x - 2) ^ 2

theorem has_minimum_value_iff (a : ℝ) : (∃ m, ∀ x, f a x ≥ m) ↔ 0 ≤ a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_has_minimum_value_iff_l834_83411


namespace NUMINAMATH_GPT_greatest_integer_x_l834_83494

theorem greatest_integer_x (x : ℤ) (h : 7 - 3 * x + 2 > 23) : x ≤ -5 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_integer_x_l834_83494


namespace NUMINAMATH_GPT_find_k_l834_83404

theorem find_k
  (k : ℝ)
  (AB : ℝ × ℝ := (3, 1))
  (AC : ℝ × ℝ := (2, k))
  (BC : ℝ × ℝ := (2 - 3, k - 1))
  (h_perpendicular : AB.1 * BC.1 + AB.2 * BC.2 = 0)
  : k = 4 :=
sorry

end NUMINAMATH_GPT_find_k_l834_83404


namespace NUMINAMATH_GPT_find_k_l834_83473

noncomputable def arithmetic_sequence_sum (a₁ d : ℕ) (n : ℕ) : ℕ :=
  n * a₁ + (n * (n-1)) / 2 * d

theorem find_k (a₁ d : ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h₁ : a₁ = 1) (h₂ : d = 2) (h₃ : ∀ n, S (n+2) = 28 + S n) :
  k = 6 := by
  sorry

end NUMINAMATH_GPT_find_k_l834_83473
