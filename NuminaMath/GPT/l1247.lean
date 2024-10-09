import Mathlib

namespace ted_age_l1247_124784

variable (t s : ℕ)

theorem ted_age (h1 : t = 3 * s - 10) (h2 : t + s = 65) : t = 46 := by
  sorry

end ted_age_l1247_124784


namespace triangles_sticks_not_proportional_l1247_124770

theorem triangles_sticks_not_proportional :
  ∀ (n_triangles n_sticks : ℕ), 
  (∃ k : ℕ, n_triangles = k * n_sticks) 
  ∨ 
  (∃ k : ℕ, n_triangles * n_sticks = k) 
  → False :=
by
  sorry

end triangles_sticks_not_proportional_l1247_124770


namespace joe_anne_bill_difference_l1247_124713

theorem joe_anne_bill_difference (m j a : ℝ) 
  (hm : (15 / 100) * m = 3) 
  (hj : (10 / 100) * j = 2) 
  (ha : (20 / 100) * a = 3) : 
  j - a = 5 := 
by {
  sorry
}

end joe_anne_bill_difference_l1247_124713


namespace slope_and_intercept_of_given_function_l1247_124738

-- Defining the form of a linear function
def linear_function (m b : ℝ) (x : ℝ) : ℝ := m * x + b

-- The given linear function
def given_function (x : ℝ) : ℝ := 3 * x + 2

-- Stating the problem as a theorem
theorem slope_and_intercept_of_given_function :
  (∀ x : ℝ, given_function x = linear_function 3 2 x) :=
by
  intro x
  sorry

end slope_and_intercept_of_given_function_l1247_124738


namespace area_of_figure_l1247_124712

theorem area_of_figure : 
  ∀ (x y : ℝ), |3 * x + 4| + |4 * y - 3| ≤ 12 → area_of_rhombus = 24 := 
by
  sorry

end area_of_figure_l1247_124712


namespace paige_finished_problems_l1247_124714

-- Define the conditions
def initial_problems : ℕ := 110
def problems_per_page : ℕ := 9
def remaining_pages : ℕ := 7

-- Define the statement we want to prove
theorem paige_finished_problems :
  initial_problems - (remaining_pages * problems_per_page) = 47 :=
by sorry

end paige_finished_problems_l1247_124714


namespace distance_between_centers_l1247_124735

theorem distance_between_centers (r1 r2 d x : ℝ) (h1 : r1 = 10) (h2 : r2 = 6) (h3 : d = 30) :
  x = 2 * Real.sqrt 229 := 
sorry

end distance_between_centers_l1247_124735


namespace shaded_region_area_l1247_124715

structure Point where
  x : ℝ
  y : ℝ

def W : Point := ⟨0, 0⟩
def X : Point := ⟨5, 0⟩
def Y : Point := ⟨5, 2⟩
def Z : Point := ⟨0, 2⟩
def Q : Point := ⟨1, 0⟩
def S : Point := ⟨5, 0.5⟩
def R : Point := ⟨0, 1⟩
def D : Point := ⟨1, 2⟩

def triangle_area (A B C : Point) : ℝ :=
  0.5 * |(A.x * B.y + B.x * C.y + C.x * A.y) - (B.x * A.y + C.x * B.y + A.x * C.y)|

theorem shaded_region_area : triangle_area R D Y = 1 := by
  sorry

end shaded_region_area_l1247_124715


namespace Harriett_total_money_l1247_124796

open Real

theorem Harriett_total_money :
    let quarters := 14 * 0.25
    let dimes := 7 * 0.10
    let nickels := 9 * 0.05
    let pennies := 13 * 0.01
    let half_dollars := 4 * 0.50
    quarters + dimes + nickels + pennies + half_dollars = 6.78 :=
by
    sorry

end Harriett_total_money_l1247_124796


namespace black_square_area_l1247_124744

-- Define the edge length of the cube
def edge_length := 12

-- Define the total amount of yellow paint available
def yellow_paint_area := 432

-- Define the total surface area of the cube
def total_surface_area := 6 * (edge_length * edge_length)

-- Define the area covered by yellow paint per face
def yellow_per_face := yellow_paint_area / 6

-- Define the area of one face of the cube
def face_area := edge_length * edge_length

-- State the theorem: the area of the black square on each face
theorem black_square_area : (face_area - yellow_per_face) = 72 := by
  sorry

end black_square_area_l1247_124744


namespace petya_friends_l1247_124766

variable (friends stickers : Nat)

-- Condition where giving 5 stickers to each friend leaves Petya with 8 stickers.
def condition1 := stickers - friends * 5 = 8

-- Condition where giving 6 stickers to each friend makes Petya short of 11 stickers.
def condition2 := stickers = friends * 6 - 11

-- The theorem that states Petya has 19 friends given the above conditions
theorem petya_friends : ∀ {friends stickers : Nat}, 
  (stickers - friends * 5 = 8) →
  (stickers = friends * 6 - 11) →
  friends = 19 := 
by
  intros friends stickers cond1 cond2
  have proof : friends = 19 := sorry
  exact proof

end petya_friends_l1247_124766


namespace waiting_time_probability_l1247_124726

theorem waiting_time_probability :
  (∀ (t : ℝ), 0 ≤ t ∧ t < 30 → (1 / 30) * (if t < 25 then 5 else 5 - (t - 25)) = 1 / 6) :=
by
  sorry

end waiting_time_probability_l1247_124726


namespace problem_statement_l1247_124758

variable {x y z : ℝ}

theorem problem_statement (h : x^3 + y^3 + z^3 - 3 * x * y * z - 3 * (x^2 + y^2 + z^2 - x * y - y * z - z * x) = 0)
  (hne : ¬(x = y ∧ y = z)) (hpos : x > 0 ∧ y > 0 ∧ z > 0) :
  (x + y + z = 3) ∧ (x^2 * (1 + y) + y^2 * (1 + z) + z^2 * (1 + x) > 6) :=
sorry

end problem_statement_l1247_124758


namespace solve_for_A_l1247_124760

variable (x y : ℝ)

theorem solve_for_A (A : ℝ) : (2 * x - y) ^ 2 + A = (2 * x + y) ^ 2 → A = 8 * x * y :=
by
  intro h
  sorry

end solve_for_A_l1247_124760


namespace find_second_divisor_l1247_124776

theorem find_second_divisor
  (N D : ℕ)
  (h1 : ∃ k : ℕ, N = 35 * k + 25)
  (h2 : ∃ m : ℕ, N = D * m + 4) :
  D = 21 :=
sorry

end find_second_divisor_l1247_124776


namespace time_to_cross_is_30_seconds_l1247_124703

variable (length_train : ℕ) (speed_km_per_hr : ℕ) (length_bridge : ℕ)

def total_distance := length_train + length_bridge

def speed_m_per_s := (speed_km_per_hr * 1000 : ℕ) / 3600

def time_to_cross_bridge := total_distance length_train length_bridge / speed_m_per_s speed_km_per_hr

theorem time_to_cross_is_30_seconds 
  (h_train_length : length_train = 140)
  (h_train_speed : speed_km_per_hr = 45)
  (h_bridge_length : length_bridge = 235) :
  time_to_cross_bridge length_train speed_km_per_hr length_bridge = 30 :=
by
  sorry

end time_to_cross_is_30_seconds_l1247_124703


namespace complex_number_second_quadrant_l1247_124725

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- Define the complex number z
def z : ℂ := i * (1 + i)

-- Define a predicate to determine if a complex number is in the second quadrant
def is_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- The main statement
theorem complex_number_second_quadrant : is_second_quadrant z := by
  sorry

end complex_number_second_quadrant_l1247_124725


namespace required_height_for_roller_coaster_l1247_124707

-- Definitions based on conditions from the problem
def initial_height : ℕ := 48
def natural_growth_rate_per_month : ℚ := 1 / 3
def upside_down_growth_rate_per_hour : ℚ := 1 / 12
def hours_per_month_hanging_upside_down : ℕ := 2
def months_in_a_year : ℕ := 12

-- Calculations needed for the proof
def annual_natural_growth := natural_growth_rate_per_month * months_in_a_year
def annual_upside_down_growth := (upside_down_growth_rate_per_hour * hours_per_month_hanging_upside_down) * months_in_a_year
def total_annual_growth := annual_natural_growth + annual_upside_down_growth
def height_next_year := initial_height + total_annual_growth

-- Statement of the required height for the roller coaster
theorem required_height_for_roller_coaster : height_next_year = 54 :=
by
  sorry

end required_height_for_roller_coaster_l1247_124707


namespace total_stickers_at_end_of_week_l1247_124782

-- Defining the initial and earned stickers as constants
def initial_stickers : ℕ := 39
def earned_stickers : ℕ := 22

-- Defining the goal as a proof statement
theorem total_stickers_at_end_of_week : initial_stickers + earned_stickers = 61 := 
by {
  sorry
}

end total_stickers_at_end_of_week_l1247_124782


namespace ratio_is_one_half_l1247_124790

namespace CupRice

-- Define the grains of rice in one cup
def grains_in_one_cup : ℕ := 480

-- Define the grains of rice in the portion of the cup
def grains_in_portion : ℕ := 8 * 3 * 10

-- Define the ratio of the portion of the cup to the whole cup
def portion_to_cup_ratio := grains_in_portion / grains_in_one_cup

-- Prove that the ratio of the portion of the cup to the whole cup is 1:2
theorem ratio_is_one_half : portion_to_cup_ratio = 1 / 2 := by
  -- Proof goes here, but we skip it as required
  sorry
end CupRice

end ratio_is_one_half_l1247_124790


namespace calc_expression_l1247_124779

theorem calc_expression :
  (2014 * 2014 + 2012) - 2013 * 2013 = 6039 :=
by
  -- Let 2014 = 2013 + 1 and 2012 = 2013 - 1
  have h2014 : 2014 = 2013 + 1 := by sorry
  have h2012 : 2012 = 2013 - 1 := by sorry
  -- Start the main proof
  sorry

end calc_expression_l1247_124779


namespace max_crosses_4x10_proof_l1247_124775

def max_crosses_4x10 (table : Matrix ℕ ℕ Bool) : ℕ :=
  sorry -- Placeholder for actual function implementation

theorem max_crosses_4x10_proof (table : Matrix ℕ ℕ Bool) (h : ∀ i < 4, ∃ j < 10, table i j = tt) :
  max_crosses_4x10 table = 30 :=
sorry

end max_crosses_4x10_proof_l1247_124775


namespace strategy_classification_l1247_124701

inductive Player
| A
| B

def A_winning_strategy (n0 : Nat) : Prop :=
  n0 >= 8

def B_winning_strategy (n0 : Nat) : Prop :=
  n0 <= 5

def neither_winning_strategy (n0 : Nat) : Prop :=
  n0 = 6 ∨ n0 = 7

theorem strategy_classification (n0 : Nat) : 
  (A_winning_strategy n0 ∨ B_winning_strategy n0 ∨ neither_winning_strategy n0) := by
    sorry

end strategy_classification_l1247_124701


namespace find_m_n_find_a_l1247_124733

def quadratic_roots (x : ℝ) (m n : ℝ) : Prop := 
  x^2 + m * x - 3 = 0

theorem find_m_n {m n : ℝ} : 
  quadratic_roots (-1) m n ∧ quadratic_roots n m n → 
  m = -2 ∧ n = 3 := 
sorry

def f (x m : ℝ) : ℝ := 
  x^2 + m * x - 3

theorem find_a {a m : ℝ} (h : m = -2) : 
  f 3 m = f (2 * a - 3) m → 
  a = 1 ∨ a = 3 := 
sorry

end find_m_n_find_a_l1247_124733


namespace fraction_of_students_who_say_dislike_but_actually_like_l1247_124728

-- Define the conditions
def total_students : ℕ := 100
def like_dancing : ℕ := total_students / 2
def dislike_dancing : ℕ := total_students / 2

def like_dancing_honest : ℕ := (7 * like_dancing) / 10
def like_dancing_dishonest : ℕ := (3 * like_dancing) / 10

def dislike_dancing_honest : ℕ := (4 * dislike_dancing) / 5
def dislike_dancing_dishonest : ℕ := dislike_dancing / 5

-- Define the proof objective
theorem fraction_of_students_who_say_dislike_but_actually_like :
  (like_dancing_dishonest : ℚ) / (total_students - like_dancing_honest - dislike_dancing_dishonest) = 3 / 11 :=
by
  sorry

end fraction_of_students_who_say_dislike_but_actually_like_l1247_124728


namespace quadratic_no_real_roots_l1247_124786

theorem quadratic_no_real_roots 
  (a b c m : ℝ) 
  (h1 : c > 0) 
  (h2 : c = a * m^2) 
  (h3 : c = b * m)
  : ∀ x : ℝ, ¬ (a * x^2 + b * x + c = 0) :=
by 
  sorry

end quadratic_no_real_roots_l1247_124786


namespace machines_remain_closed_l1247_124746

open Real

/-- A techno company has 14 machines of equal efficiency in its factory.
The annual manufacturing costs are Rs 42000 and establishment charges are Rs 12000.
The annual output of the company is Rs 70000. The annual output and manufacturing
costs are directly proportional to the number of machines. The shareholders get
12.5% profit, which is directly proportional to the annual output of the company.
If some machines remain closed throughout the year, then the percentage decrease
in the amount of profit of the shareholders is 12.5%. Prove that 2 machines remain
closed throughout the year. -/
theorem machines_remain_closed (machines total_cost est_charges output : ℝ)
    (shareholders_profit : ℝ)
    (machines_closed percentage_decrease : ℝ) :
  machines = 14 →
  total_cost = 42000 →
  est_charges = 12000 →
  output = 70000 →
  shareholders_profit = 0.125 →
  percentage_decrease = 0.125 →
  machines_closed = 2 :=
by
  sorry

end machines_remain_closed_l1247_124746


namespace A_gt_B_and_C_lt_A_l1247_124783

structure Box where
  x : ℕ
  y : ℕ
  z : ℕ

def canBePlacedInside (K P : Box) :=
  (K.x ≤ P.x ∧ K.y ≤ P.y ∧ K.z ≤ P.z) ∨
  (K.x ≤ P.x ∧ K.y ≤ P.z ∧ K.z ≤ P.y) ∨
  (K.x ≤ P.y ∧ K.y ≤ P.x ∧ K.z ≤ P.z) ∨
  (K.x ≤ P.y ∧ K.y ≤ P.z ∧ K.z ≤ P.x) ∨
  (K.x ≤ P.z ∧ K.y ≤ P.x ∧ K.z ≤ P.y) ∨
  (K.x ≤ P.z ∧ K.y ≤ P.y ∧ K.z ≤ P.x)

theorem A_gt_B_and_C_lt_A :
  let A := Box.mk 6 5 3
  let B := Box.mk 5 4 1
  let C := Box.mk 3 2 2
  (canBePlacedInside B A ∧ ¬ canBePlacedInside A B) ∧
  (canBePlacedInside C A ∧ ¬ canBePlacedInside A C) :=
by
  sorry -- Proof goes here

end A_gt_B_and_C_lt_A_l1247_124783


namespace water_segment_length_l1247_124709

theorem water_segment_length 
  (total_distance : ℝ)
  (find_probability : ℝ)
  (lose_probability : ℝ)
  (probability_equation : total_distance * lose_probability = 750) :
  total_distance = 2500 → 
  find_probability = 7 / 10 →
  lose_probability = 3 / 10 →
  x = 750 :=
by
  intros h1 h2 h3
  sorry

end water_segment_length_l1247_124709


namespace distribute_paper_clips_l1247_124731

theorem distribute_paper_clips (total_paper_clips boxes : ℕ) (h_total : total_paper_clips = 81) (h_boxes : boxes = 9) : total_paper_clips / boxes = 9 := by
  sorry

end distribute_paper_clips_l1247_124731


namespace equal_real_roots_value_l1247_124742

theorem equal_real_roots_value (a c : ℝ) (ha : a ≠ 0) (h : 4 - 4 * a * (2 - c) = 0) : (1 / a) + c = 2 := 
by
  sorry

end equal_real_roots_value_l1247_124742


namespace babysitting_earnings_l1247_124740

theorem babysitting_earnings
  (cost_video_game : ℕ)
  (cost_candy : ℕ)
  (hours_worked : ℕ)
  (amount_left : ℕ)
  (total_earned : ℕ)
  (earnings_per_hour : ℕ) :
  cost_video_game = 60 →
  cost_candy = 5 →
  hours_worked = 9 →
  amount_left = 7 →
  total_earned = cost_video_game + cost_candy + amount_left →
  earnings_per_hour = total_earned / hours_worked →
  earnings_per_hour = 8 :=
by
  intros h_game h_candy h_hours h_left h_total_earned h_earn_per_hour
  rw [h_game, h_candy] at h_total_earned
  simp at h_total_earned
  have h_total_earned : total_earned = 72 := by linarith
  rw [h_total_earned, h_hours] at h_earn_per_hour
  simp at h_earn_per_hour
  assumption

end babysitting_earnings_l1247_124740


namespace num_consecutive_sets_summing_to_90_l1247_124706

-- Define the arithmetic sequence sum properties
theorem num_consecutive_sets_summing_to_90 : 
  ∃ n : ℕ, n ≥ 2 ∧
    ∃ (a : ℕ), 2 * a + n - 1 = 180 / n ∧
      (∃ k : ℕ, 
         k ≥ 2 ∧
         ∃ b : ℕ, 2 * b + k - 1 = 180 / k) ∧
      (∃ m : ℕ, 
         m ≥ 2 ∧ 
         ∃ c : ℕ, 2 * c + m - 1 = 180 / m) ∧
      (n = 3 ∨ n = 5 ∨ n = 9) :=
sorry

end num_consecutive_sets_summing_to_90_l1247_124706


namespace exist_m_n_l1247_124710

theorem exist_m_n (p : ℕ) [hp : Fact (Nat.Prime p)] (h : 5 < p) :
  ∃ m n : ℕ, (m + n < p ∧ p ∣ (2^m * 3^n - 1)) := sorry

end exist_m_n_l1247_124710


namespace n_squared_plus_n_plus_1_is_odd_l1247_124755

theorem n_squared_plus_n_plus_1_is_odd (n : ℤ) : Odd (n^2 + n + 1) :=
sorry

end n_squared_plus_n_plus_1_is_odd_l1247_124755


namespace time_to_cover_escalator_l1247_124732

noncomputable def escalator_speed : ℝ := 8
noncomputable def person_speed : ℝ := 2
noncomputable def escalator_length : ℝ := 160
noncomputable def combined_speed : ℝ := escalator_speed + person_speed

theorem time_to_cover_escalator :
  escalator_length / combined_speed = 16 := by
  sorry

end time_to_cover_escalator_l1247_124732


namespace max_discount_l1247_124748

-- Definitions:
def cost_price : ℝ := 400
def sale_price : ℝ := 600
def desired_profit_margin : ℝ := 0.05

-- Statement:
theorem max_discount 
  (x : ℝ) 
  (hx : sale_price * (1 - x / 100) ≥ cost_price * (1 + desired_profit_margin)) :
  x ≤ 90 := 
sorry

end max_discount_l1247_124748


namespace max_value_of_a_l1247_124736

variable {a : ℝ}

theorem max_value_of_a (h : a > 0) : 
  (∀ x : ℝ, x > 0 → (2 * x^2 - a * x + a > 0)) ↔ a ≤ 8 := 
sorry

end max_value_of_a_l1247_124736


namespace inequality_division_by_positive_l1247_124787

theorem inequality_division_by_positive (x y : ℝ) (h : x > y) : (x / 5 > y / 5) :=
by
  sorry

end inequality_division_by_positive_l1247_124787


namespace remainder_of_3_pow_45_mod_17_l1247_124704

theorem remainder_of_3_pow_45_mod_17 : 3^45 % 17 = 15 := 
by {
  sorry
}

end remainder_of_3_pow_45_mod_17_l1247_124704


namespace solve_inequality_l1247_124788

theorem solve_inequality (x : ℝ) : |x - 1| + |x - 2| > 5 ↔ (x < -1 ∨ x > 4) :=
by
  sorry

end solve_inequality_l1247_124788


namespace no_eleven_points_achieve_any_score_l1247_124789

theorem no_eleven_points (x y : ℕ) : 3 * x + 7 * y ≠ 11 := 
sorry

theorem achieve_any_score (S : ℕ) (h : S ≥ 12) : ∃ (x y : ℕ), 3 * x + 7 * y = S :=
sorry

end no_eleven_points_achieve_any_score_l1247_124789


namespace smallest_yummy_is_minus_2013_l1247_124759

-- Define a yummy integer
def is_yummy (A : ℤ) : Prop :=
  ∃ (k : ℕ), ∃ (a : ℤ), (a <= A) ∧ (a + k = A) ∧ ((k + 1) * A - k*(k + 1)/2 = 2014)

-- Define the smallest yummy integer
def smallest_yummy : ℤ :=
  -2013

-- The Lean theorem to state the proof problem
theorem smallest_yummy_is_minus_2013 : ∀ A : ℤ, is_yummy A → (-2013 ≤ A) :=
by
  sorry

end smallest_yummy_is_minus_2013_l1247_124759


namespace video_game_cost_l1247_124749

theorem video_game_cost
  (weekly_allowance1 : ℕ)
  (weeks1 : ℕ)
  (weekly_allowance2 : ℕ)
  (weeks2 : ℕ)
  (money_spent_on_clothes_fraction : ℚ)
  (remaining_money : ℕ)
  (allowance1 : weekly_allowance1 = 5)
  (duration1 : weeks1 = 8)
  (allowance2 : weekly_allowance2 = 6)
  (duration2 : weeks2 = 6)
  (money_spent_fraction : money_spent_on_clothes_fraction = 1/2)
  (remaining_money_condition : remaining_money = 3) :
  (weekly_allowance1 * weeks1 + weekly_allowance2 * weeks2) * (1 - money_spent_on_clothes_fraction) - remaining_money = 35 :=
by
  rw [allowance1, duration1, allowance2, duration2, money_spent_fraction, remaining_money_condition]
  -- Calculation steps are omitted; they can be filled in here.
  exact sorry

end video_game_cost_l1247_124749


namespace question_eq_answer_l1247_124753

theorem question_eq_answer (n : ℝ) (h : 0.25 * 0.1 * n = 15) :
  0.1 * 0.25 * n = 15 :=
by
  sorry

end question_eq_answer_l1247_124753


namespace find_smaller_part_l1247_124799

noncomputable def smaller_part (x y : ℕ) : ℕ :=
  if x ≤ y then x else y

theorem find_smaller_part (x y : ℕ) (h1 : x + y = 24) (h2 : 7 * x + 5 * y = 146) : smaller_part x y = 11 :=
  sorry

end find_smaller_part_l1247_124799


namespace totalNameLengths_l1247_124716

-- Definitions of the lengths of names
def JonathanNameLength := 8 + 10
def YoungerSisterNameLength := 5 + 10
def OlderBrotherNameLength := 6 + 10
def YoungestSisterNameLength := 4 + 15

-- Statement to prove
theorem totalNameLengths :
  JonathanNameLength + YoungerSisterNameLength + OlderBrotherNameLength + YoungestSisterNameLength = 68 :=
by
  sorry -- no proof required

end totalNameLengths_l1247_124716


namespace find_x4_l1247_124764

open Real

theorem find_x4 (x : ℝ) (h₁ : 0 < x) (h₂ : sqrt (1 - x^2) + sqrt (1 + x^2) = 2) : x^4 = 0 :=
by
  sorry

end find_x4_l1247_124764


namespace find_prices_max_basketballs_l1247_124751

-- Definition of given conditions
def conditions1 (x y : ℝ) : Prop := 
  (x - y = 50) ∧ (6 * x + 8 * y = 1700)

-- Definitions of questions:
-- Question 1: Find the price of one basketball and one soccer ball
theorem find_prices (x y : ℝ) (h: conditions1 x y) : x = 150 ∧ y = 100 := sorry

-- Definition of given conditions for Question 2
def conditions2 (x y : ℝ) (a : ℕ) : Prop :=
  (x = 150) ∧ (y = 100) ∧ 
  (0.9 * x * a + 0.85 * y * (10 - a) ≤ 1150)

-- Question 2: The school plans to purchase 10 items with given discounts
theorem max_basketballs (x y : ℝ) (a : ℕ) (h1: x = 150) (h2: y = 100) (h3: a ≤ 10) (h4: conditions2 x y a) : a ≤ 6 := sorry

end find_prices_max_basketballs_l1247_124751


namespace mart_income_percentage_j_l1247_124720

variables (J T M : ℝ)

-- condition: Tim's income is 40 percent less than Juan's income
def tims_income := T = 0.60 * J

-- condition: Mart's income is 40 percent more than Tim's income
def marts_income := M = 1.40 * T

-- goal: Prove that Mart's income is 84 percent of Juan's income
theorem mart_income_percentage_j (J : ℝ) (T : ℝ) (M : ℝ)
  (h1 : T = 0.60 * J) 
  (h2 : M = 1.40 * T) : 
  M = 0.84 * J := 
sorry

end mart_income_percentage_j_l1247_124720


namespace total_payment_360_l1247_124785

noncomputable def q : ℝ := 12
noncomputable def p_wage : ℝ := 1.5 * q
noncomputable def p_hourly_rate : ℝ := q + 6
noncomputable def h : ℝ := 20
noncomputable def total_payment_p : ℝ := p_wage * h -- The total payment when candidate p is hired
noncomputable def total_payment_q : ℝ := q * (h + 10) -- The total payment when candidate q is hired

theorem total_payment_360 : 
  p_wage = p_hourly_rate ∧ 
  total_payment_p = total_payment_q ∧ 
  total_payment_p = 360 := by
  sorry

end total_payment_360_l1247_124785


namespace dave_pieces_l1247_124730

theorem dave_pieces (boxes_bought : ℕ) (boxes_given : ℕ) (pieces_per_box : ℕ) 
  (h₁ : boxes_bought = 12) (h₂ : boxes_given = 5) (h₃ : pieces_per_box = 3) : 
  boxes_bought - boxes_given * pieces_per_box = 21 :=
by
  sorry

end dave_pieces_l1247_124730


namespace circle_tangent_to_x_axis_l1247_124780

theorem circle_tangent_to_x_axis (b : ℝ) :
  (∃ c : ℝ, ∀ x y : ℝ,
    (x^2 + y^2 + 4 * x + 2 * b * y + c = 0) ∧ (∃ r : ℝ, r > 0 ∧ ∀ y : ℝ, y = -b ↔ y = 2)) ↔ (b = 2 ∨ b = -2) :=
sorry

end circle_tangent_to_x_axis_l1247_124780


namespace percentage_increase_l1247_124722

-- Conditions
variables (S_final S_initial : ℝ) (P : ℝ)
def conditions := (S_final = 3135) ∧ (S_initial = 3000) ∧
  (S_final = (S_initial + (P/100) * S_initial) - 0.05 * (S_initial + (P/100) * S_initial))

-- Statement of the problem
theorem percentage_increase (S_final S_initial : ℝ) 
  (cond : conditions S_final S_initial P) : P = 10 := by
  sorry

end percentage_increase_l1247_124722


namespace parallel_lines_iff_a_eq_3_l1247_124757

theorem parallel_lines_iff_a_eq_3 (a : ℝ) :
  (∀ x y : ℝ, (6 * x - 4 * y + 1 = 0) ↔ (a * x - 2 * y - 1 = 0)) ↔ (a = 3) := 
sorry

end parallel_lines_iff_a_eq_3_l1247_124757


namespace real_and_equal_roots_of_quadratic_l1247_124797

theorem real_and_equal_roots_of_quadratic (k: ℝ) :
  (-(k+2))^2 - 4 * 3 * 12 = 0 ↔ k = 10 ∨ k = -14 :=
by
  sorry

end real_and_equal_roots_of_quadratic_l1247_124797


namespace inequality_am_gm_l1247_124765

theorem inequality_am_gm 
  (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := 
by 
  sorry

end inequality_am_gm_l1247_124765


namespace production_difference_l1247_124723

theorem production_difference (w t : ℕ) (h1 : w = 3 * t) :
  (w * t) - ((w + 6) * (t - 3)) = 3 * t + 18 :=
by
  sorry

end production_difference_l1247_124723


namespace count_two_digit_numbers_with_unit_7_lt_50_l1247_124769

def is_two_digit_nat (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def has_unit_digit_7 (n : ℕ) : Prop := n % 10 = 7
def less_than_50 (n : ℕ) : Prop := n < 50

theorem count_two_digit_numbers_with_unit_7_lt_50 : 
  ∃ (s : Finset ℕ), 
    (∀ n ∈ s, is_two_digit_nat n ∧ has_unit_digit_7 n ∧ less_than_50 n) ∧ s.card = 4 := 
by
  sorry

end count_two_digit_numbers_with_unit_7_lt_50_l1247_124769


namespace side_length_percentage_error_l1247_124729

variable (s s' : Real)
-- Conditions
-- s' = s * 1.06 (measured side length is 6% more than actual side length)
-- (s'^2 - s^2) / s^2 * 100% = 12.36% (percentage error in area)

theorem side_length_percentage_error 
    (h1 : s' = s * 1.06)
    (h2 : (s'^2 - s^2) / s^2 * 100 = 12.36) :
    ((s' - s) / s) * 100 = 6 := 
sorry

end side_length_percentage_error_l1247_124729


namespace lines_perpendicular_iff_l1247_124763

/-- Given two lines y = k₁ x + l₁ and y = k₂ x + l₂, 
    which are not parallel to the coordinate axes,
    these lines are perpendicular if and only if k₁ * k₂ = -1. -/
theorem lines_perpendicular_iff 
  (k₁ k₂ l₁ l₂ : ℝ) (h1 : k₁ ≠ 0) (h2 : k₂ ≠ 0) :
  (∀ x, k₁ * x + l₁ = k₂ * x + l₂) <-> k₁ * k₂ = -1 :=
sorry

end lines_perpendicular_iff_l1247_124763


namespace factor_expression_value_l1247_124721

theorem factor_expression_value :
  ∃ (k m n : ℕ), 
    k > 1 ∧ m > 1 ∧ n > 1 ∧ 
    k ≤ 60 ∧ m ≤ 35 ∧ n ≤ 20 ∧ 
    (2^k + 3^m + k^3 * m^n - n = 43) :=
by
  sorry

end factor_expression_value_l1247_124721


namespace students_contribution_l1247_124743

theorem students_contribution (n x : ℕ) 
  (h₁ : ∃ (k : ℕ), k * 9 = 22725)
  (h₂ : n * x = k / 9)
  : (n = 5 ∧ x = 505) ∨ (n = 25 ∧ x = 101) :=
sorry

end students_contribution_l1247_124743


namespace total_fast_food_order_cost_l1247_124761

def burger_cost : ℕ := 5
def sandwich_cost : ℕ := 4
def smoothie_cost : ℕ := 4
def smoothies_quantity : ℕ := 2

theorem total_fast_food_order_cost : burger_cost + sandwich_cost + smoothies_quantity * smoothie_cost = 17 := 
by
  sorry

end total_fast_food_order_cost_l1247_124761


namespace find_solns_to_eqn_l1247_124737

theorem find_solns_to_eqn (x y z w : ℕ) :
  2^x * 3^y - 5^z * 7^w = 1 ↔ (x, y, z, w) = (1, 0, 0, 0) ∨ 
                                        (x, y, z, w) = (3, 0, 0, 1) ∨ 
                                        (x, y, z, w) = (1, 1, 1, 0) ∨ 
                                        (x, y, z, w) = (2, 2, 1, 1) := 
sorry -- Placeholder for the actual proof

end find_solns_to_eqn_l1247_124737


namespace Q_has_negative_and_potentially_positive_roots_l1247_124734

def Q (x : ℝ) : ℝ := x^7 - 4 * x^6 + 2 * x^5 - 9 * x^3 + 2 * x + 16

theorem Q_has_negative_and_potentially_positive_roots :
  (∃ x : ℝ, x < 0 ∧ Q x = 0) ∧ (∃ y : ℝ, y > 0 ∧ Q y = 0 ∨ ∀ z : ℝ, Q z > 0) :=
by
  sorry

end Q_has_negative_and_potentially_positive_roots_l1247_124734


namespace problem1_l1247_124767

def f (x : ℝ) := (1 - 3 * x) * (1 + x) ^ 5

theorem problem1 :
  let a : ℝ := f (1 / 3)
  a = 0 :=
by
  let a := f (1 / 3)
  sorry

end problem1_l1247_124767


namespace pieces_eaten_first_l1247_124793

variable (initial_candy : ℕ) (remaining_candy : ℕ) (candy_eaten_second : ℕ)

theorem pieces_eaten_first 
    (initial_candy := 21) 
    (remaining_candy := 7)
    (candy_eaten_second := 9) :
    (initial_candy - remaining_candy - candy_eaten_second = 5) :=
sorry

end pieces_eaten_first_l1247_124793


namespace alex_distribution_ways_l1247_124792

theorem alex_distribution_ways : (15^5 = 759375) := by {
  sorry
}

end alex_distribution_ways_l1247_124792


namespace two_integer_solutions_iff_l1247_124772

theorem two_integer_solutions_iff (a : ℝ) :
  (∃ (n m : ℤ), n ≠ m ∧ |n - 1| < a * n ∧ |m - 1| < a * m ∧
    ∀ (k : ℤ), |k - 1| < a * k → k = n ∨ k = m) ↔
  (1/2 : ℝ) < a ∧ a ≤ (2/3 : ℝ) :=
by
  sorry

end two_integer_solutions_iff_l1247_124772


namespace value_of_b_l1247_124705

theorem value_of_b (b : ℝ) :
  (∀ x : ℝ, 3 * (5 + b * x) = 18 * x + 15) → b = 6 :=
by
  intro h
  -- Proving that b = 6
  sorry

end value_of_b_l1247_124705


namespace initial_observations_l1247_124768

theorem initial_observations (n : ℕ) (S : ℕ) (new_obs : ℕ) :
  (S = 12 * n) → (new_obs = 5) → (S + new_obs = 11 * (n + 1)) → n = 6 :=
by
  intro h1 h2 h3
  sorry

end initial_observations_l1247_124768


namespace decimal_equivalence_l1247_124739

theorem decimal_equivalence : 4 + 3 / 10 + 9 / 1000 = 4.309 := 
by
  sorry

end decimal_equivalence_l1247_124739


namespace incorrect_calculation_l1247_124781

theorem incorrect_calculation : ¬ (3 + 2 * Real.sqrt 2 = 5 * Real.sqrt 2) :=
by sorry

end incorrect_calculation_l1247_124781


namespace m_lt_n_l1247_124773

theorem m_lt_n (a t : ℝ) (h : 0 < t ∧ t < 1) : 
  abs (Real.log (1 + t) / Real.log a) < abs (Real.log (1 - t) / Real.log a) :=
sorry

end m_lt_n_l1247_124773


namespace man_age_difference_l1247_124745

theorem man_age_difference (S M : ℕ) (h1 : S = 24) (h2 : M + 2 = 2 * (S + 2)) : M - S = 26 := by
  sorry

end man_age_difference_l1247_124745


namespace perpendicular_lines_implies_m_values_l1247_124727

-- Define the equations of the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x - (m - 2) * y + 2 = 0
def l2 (m : ℝ) (x y : ℝ) : Prop := 3 * x + m * y - 1 = 0

-- Define the condition of perpendicularity between lines l1 and l2
def perpendicular (m : ℝ) : Prop :=
  let a1 := (m + 2) / (m - 2)
  let a2 := -3 / m
  a1 * a2 = -1

-- The statement to be proved
theorem perpendicular_lines_implies_m_values (m : ℝ) :
  (∀ x y : ℝ, l1 m x y ∧ l2 m x y → perpendicular m) → (m = -1 ∨ m = 6) :=
sorry

end perpendicular_lines_implies_m_values_l1247_124727


namespace problem_statement_l1247_124700

variable (f : ℕ → ℝ)

theorem problem_statement (hf : ∀ k : ℕ, f k ≥ k^2 → f (k + 1) ≥ (k + 1)^2)
  (h : f 4 = 25) : ∀ k : ℕ, k ≥ 4 → f k ≥ k^2 := 
by
  sorry

end problem_statement_l1247_124700


namespace minimum_value_of_weighted_sum_l1247_124717

theorem minimum_value_of_weighted_sum 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 27) :
  3 * a + 6 * b + 9 * c ≥ 54 :=
sorry

end minimum_value_of_weighted_sum_l1247_124717


namespace division_by_fraction_l1247_124702

theorem division_by_fraction :
  5 / (8 / 13) = 65 / 8 :=
sorry

end division_by_fraction_l1247_124702


namespace smallest_term_index_l1247_124771

theorem smallest_term_index (a_n : ℕ → ℤ) (h : ∀ n, a_n n = 3 * n^2 - 38 * n + 12) : ∃ n, a_n n = a_n 6 ∧ ∀ m, a_n m ≥ a_n 6 :=
by
  sorry

end smallest_term_index_l1247_124771


namespace cos_equiv_l1247_124774

theorem cos_equiv (n : ℤ) (hn : 0 ≤ n ∧ n ≤ 180) (hcos : Real.cos (n * Real.pi / 180) = Real.cos (1018 * Real.pi / 180)) : n = 62 := 
sorry

end cos_equiv_l1247_124774


namespace relation_between_m_and_n_l1247_124795

variable {A x y z a b c d e n m : ℝ}
variable {p r : ℝ}
variable (s : finset ℝ) (hset : s = {x, y, z, a, b, c, d, e})
variable (hsorted : x < y ∧ y < z ∧ z < a ∧ a < b ∧ b < c ∧ c < d ∧ d < e)
variable (hne : n ∉ s)
variable (hme : m ∉ s)

theorem relation_between_m_and_n 
  (h_avg_n : (s.sum + n) / 9 = (s.sum / 8) * (1 + p / 100)) 
  (h_avg_m : (s.sum + m) / 9 = (s.sum / 8) * (1 + r / 100)) 
  : m = n + 9 * (s.sum / 8) * (r / 100 - p / 100) :=
sorry

end relation_between_m_and_n_l1247_124795


namespace population_ratio_l1247_124724

theorem population_ratio
  (P_A P_B P_C P_D P_E P_F : ℕ)
  (h1 : P_A = 8 * P_B)
  (h2 : P_B = 5 * P_C)
  (h3 : P_D = 3 * P_C)
  (h4 : P_D = P_E / 2)
  (h5 : P_F = P_A / 4) :
  P_E / P_B = 6 / 5 := by
    sorry

end population_ratio_l1247_124724


namespace b_spends_85_percent_l1247_124718

-- Definitions based on the given conditions
def combined_salary (a_salary b_salary : ℤ) : Prop := a_salary + b_salary = 3000
def a_salary : ℤ := 2250
def a_spending_ratio : ℝ := 0.95
def a_savings : ℝ := a_salary - a_salary * a_spending_ratio
def b_savings : ℝ := a_savings

-- The goal is to prove that B spends 85% of his salary
theorem b_spends_85_percent (b_salary : ℤ) (b_spending_ratio : ℝ) :
  combined_salary a_salary b_salary →
  b_spending_ratio * b_salary = 0.85 * b_salary :=
  sorry

end b_spends_85_percent_l1247_124718


namespace ratio_proof_l1247_124719

theorem ratio_proof (X: ℕ) (h: 150 * 2 = 300 * X) : X = 1 := by
  sorry

end ratio_proof_l1247_124719


namespace max_int_value_of_a_real_roots_l1247_124778

-- Definitions and theorem statement based on the above conditions
theorem max_int_value_of_a_real_roots (a : ℤ) :
  (∃ x : ℝ, (a-1) * x^2 - 2 * x + 3 = 0) ↔ a ≠ 1 ∧ a ≤ 0 := by
  sorry

end max_int_value_of_a_real_roots_l1247_124778


namespace roof_length_width_difference_l1247_124752

theorem roof_length_width_difference
  {w l : ℝ} 
  (h_area : l * w = 576) 
  (h_length : l = 4 * w) 
  (hw_pos : w > 0) :
  l - w = 36 :=
by 
  sorry

end roof_length_width_difference_l1247_124752


namespace no_solutions_sinx_eq_sin_sinx_l1247_124791

open Real

theorem no_solutions_sinx_eq_sin_sinx (x : ℝ) (h : 0 ≤ x ∧ x ≤ arcsin 0.9) : ¬ (sin x = sin (sin x)) :=
by
  sorry

end no_solutions_sinx_eq_sin_sinx_l1247_124791


namespace total_amount_spent_correct_l1247_124756

-- Definitions based on conditions
def price_of_food_before_tax_and_tip : ℝ := 140
def sales_tax_rate : ℝ := 0.10
def tip_rate : ℝ := 0.20

-- Definitions of intermediate steps
def sales_tax : ℝ := sales_tax_rate * price_of_food_before_tax_and_tip
def total_before_tip : ℝ := price_of_food_before_tax_and_tip + sales_tax
def tip : ℝ := tip_rate * total_before_tip
def total_amount_spent : ℝ := total_before_tip + tip

-- Theorem statement to be proved
theorem total_amount_spent_correct : total_amount_spent = 184.80 :=
by
  sorry -- Proof is skipped

end total_amount_spent_correct_l1247_124756


namespace cos_585_eq_neg_sqrt2_div_2_l1247_124794

theorem cos_585_eq_neg_sqrt2_div_2 :
  Real.cos (585 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end cos_585_eq_neg_sqrt2_div_2_l1247_124794


namespace a2_a4_a6_a8_a10_a12_sum_l1247_124747

theorem a2_a4_a6_a8_a10_a12_sum :
  ∀ (x : ℝ), 
    (1 + x + x^2)^6 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7 + a8 * x^8 + a9 * x^9 + a10 * x^10 + a11 * x^11 + a12 * x^12 →
    a2 + a4 + a6 + a8 + a10 + a12 = 364 :=
sorry

end a2_a4_a6_a8_a10_a12_sum_l1247_124747


namespace Ursula_hot_dogs_l1247_124750

theorem Ursula_hot_dogs 
  (H : ℕ) 
  (cost_hot_dog : ℚ := 1.50) 
  (cost_salad : ℚ := 2.50) 
  (num_salads : ℕ := 3) 
  (total_money : ℚ := 20) 
  (change : ℚ := 5) :
  (cost_hot_dog * H + cost_salad * num_salads = total_money - change) → H = 5 :=
by
  sorry

end Ursula_hot_dogs_l1247_124750


namespace probability_of_hitting_exactly_twice_l1247_124708

def P_hit_first : ℝ := 0.4
def P_hit_second : ℝ := 0.5
def P_hit_third : ℝ := 0.7

def P_hit_exactly_twice_in_three_shots : ℝ :=
  P_hit_first * P_hit_second * (1 - P_hit_third) +
  (1 - P_hit_first) * P_hit_second * P_hit_third +
  P_hit_first * (1 - P_hit_second) * P_hit_third

theorem probability_of_hitting_exactly_twice :
  P_hit_exactly_twice_in_three_shots = 0.41 := 
by
  sorry

end probability_of_hitting_exactly_twice_l1247_124708


namespace rectangle_perimeter_is_3y_l1247_124741

noncomputable def congruent_rectangle_perimeter (y : ℝ) (h1 : y > 0) : ℝ :=
  let side_length := 2 * y
  let center_square_side := y
  let width := (side_length - center_square_side) / 2
  let length := center_square_side
  2 * (length + width)

theorem rectangle_perimeter_is_3y (y : ℝ) (h1 : y > 0) :
  congruent_rectangle_perimeter y h1 = 3 * y :=
sorry

end rectangle_perimeter_is_3y_l1247_124741


namespace tangent_line_perpendicular_l1247_124711

theorem tangent_line_perpendicular (m : ℝ) :
  (∀ x : ℝ, y = 2 * x^2) →
  (∀ x : ℝ, (4 * x - y + m = 0) ∧ (x + 4 * y - 8 = 0) → 
  (16 + 8 * m = 0)) →
  m = -2 :=
by
  sorry

end tangent_line_perpendicular_l1247_124711


namespace fraction_simplification_l1247_124798

theorem fraction_simplification (x : ℚ) : 
  (3 / 4) * 60 - x * 60 + 63 = 12 → 
  x = (8 / 5) :=
by
  sorry

end fraction_simplification_l1247_124798


namespace max_eccentricity_of_ellipse_l1247_124762

theorem max_eccentricity_of_ellipse 
  (R_large : ℝ)
  (r_cylinder : ℝ)
  (R_small : ℝ)
  (D_centers : ℝ)
  (a : ℝ)
  (b : ℝ)
  (e : ℝ) :
  R_large = 1 → 
  r_cylinder = 1 → 
  R_small = 1/4 → 
  D_centers = 10/3 → 
  a = 5/3 → 
  b = 1 → 
  e = Real.sqrt (1 - (b / a) ^ 2) → 
  e = 4/5 := by 
  sorry

end max_eccentricity_of_ellipse_l1247_124762


namespace min_area_triangle_l1247_124777

-- Conditions
def point_on_curve (x y : ℝ) : Prop :=
  y^2 = 2 * x

def incircle (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1

-- Theorem statement
theorem min_area_triangle (x₀ y₀ b c : ℝ) (h_curve : point_on_curve x₀ y₀) 
  (h_bc_yaxis : b ≠ c) (h_incircle : incircle x₀ y₀) :
  ∃ P : ℝ × ℝ, 
    ∃ B C : ℝ × ℝ, 
    ∃ S : ℝ,
    point_on_curve P.1 P.2 ∧
    B = (0, b) ∧
    C = (0, c) ∧
    incircle P.1 P.2 ∧
    S = (x₀ - 2) + (4 / (x₀ - 2)) + 4 ∧
    S = 8 :=
sorry

end min_area_triangle_l1247_124777


namespace solve_eq1_solve_eq2_l1247_124754

-- Prove the solution of the first equation
theorem solve_eq1 (x : ℝ) : 3 * x - (x - 1) = 7 ↔ x = 3 :=
by
  sorry

-- Prove the solution of the second equation
theorem solve_eq2 (x : ℝ) : (2 * x - 1) / 3 - (x - 3) / 6 = 1 ↔ x = (5 : ℝ) / 3 :=
by
  sorry

end solve_eq1_solve_eq2_l1247_124754
