import Mathlib

namespace gcd_of_squares_sum_l1967_196796

theorem gcd_of_squares_sum : Nat.gcd (100^2 + 221^2 + 320^2) (101^2 + 220^2 + 321^2) = 1 := by
  sorry

end gcd_of_squares_sum_l1967_196796


namespace ratio_equality_l1967_196752

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc : a^2 + b^2 + c^2 = 49)
  (sum_xyz : x^2 + y^2 + z^2 = 16)
  (dot_product : a*x + b*y + c*z = 28) :
  (a + b + c) / (x + y + z) = 7/4 := by
  sorry

end ratio_equality_l1967_196752


namespace solve_a_and_m_solve_inequality_l1967_196718

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Theorem 1
theorem solve_a_and_m (a m : ℝ) : 
  (∀ x, f a x ≤ m ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 ∧ m = 3 :=
sorry

-- Theorem 2
theorem solve_inequality (t : ℝ) (h : 0 ≤ t ∧ t < 2) :
  {x : ℝ | f 2 x + t ≥ f 2 (x + 2)} = Set.Iic ((t + 2) / 2) :=
sorry

end solve_a_and_m_solve_inequality_l1967_196718


namespace min_value_of_x_l1967_196773

theorem min_value_of_x (x : ℝ) (h1 : x > 0) 
  (h2 : Real.log x / Real.log 3 ≥ Real.log 9 / Real.log 3 - (1/3) * (Real.log x / Real.log 3)) :
  x ≥ Real.sqrt 27 := by
  sorry

end min_value_of_x_l1967_196773


namespace angle_measure_theorem_l1967_196770

theorem angle_measure_theorem (x : ℝ) : x = 2 * (90 - x) - 60 ↔ x = 40 := by sorry

end angle_measure_theorem_l1967_196770


namespace stream_speed_l1967_196794

/-- Given a canoe's upstream and downstream speeds, calculate the stream speed -/
theorem stream_speed (upstream_speed downstream_speed : ℝ) 
  (h_upstream : upstream_speed = 3)
  (h_downstream : downstream_speed = 12) :
  ∃ (stream_speed : ℝ), stream_speed = 4.5 ∧
    upstream_speed = (downstream_speed - upstream_speed) / 2 - stream_speed ∧
    downstream_speed = (downstream_speed - upstream_speed) / 2 + stream_speed :=
by sorry

end stream_speed_l1967_196794


namespace xy_equals_three_l1967_196780

theorem xy_equals_three (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x ≠ y) 
  (h4 : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end xy_equals_three_l1967_196780


namespace music_school_tuition_cost_l1967_196767

/-- The cost calculation for music school tuition with sibling discounts -/
theorem music_school_tuition_cost : 
  let base_tuition : ℕ := 45
  let first_sibling_discount : ℕ := 15
  let additional_sibling_discount : ℕ := 10
  let num_children : ℕ := 4
  
  base_tuition + 
  (base_tuition - first_sibling_discount) + 
  (base_tuition - additional_sibling_discount) + 
  (base_tuition - additional_sibling_discount) = 145 :=
by sorry

end music_school_tuition_cost_l1967_196767


namespace triangle_lines_l1967_196763

-- Define the triangle ABC
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 3)
def C : ℝ × ℝ := (3, 4)

-- Define the altitude line l₁
def l₁ (x y : ℝ) : Prop := 4 * x + y - 5 = 0

-- Define the two possible equations for line l₂
def l₂_1 (x y : ℝ) : Prop := x + y - 7 = 0
def l₂_2 (x y : ℝ) : Prop := 2 * x - 3 * y + 6 = 0

-- Theorem statement
theorem triangle_lines :
  -- l₁ is the altitude from A to BC
  (∀ x y : ℝ, l₁ x y ↔ (y - A.2 = -(B.2 - C.2)/(B.1 - C.1) * (x - A.1))) ∧
  -- l₂ passes through C
  ((∀ x y : ℝ, l₂_1 x y → x = C.1 ∧ y = C.2) ∨
   (∀ x y : ℝ, l₂_2 x y → x = C.1 ∧ y = C.2)) ∧
  -- Distances from A and B to l₂ are equal
  ((∀ x y : ℝ, l₂_1 x y → 
    (|x + y - (A.1 + A.2)|/Real.sqrt 2 = |x + y - (B.1 + B.2)|/Real.sqrt 2)) ∨
   (∀ x y : ℝ, l₂_2 x y → 
    (|2*x - 3*y + 6 - (2*A.1 - 3*A.2 + 6)|/Real.sqrt 13 = 
     |2*x - 3*y + 6 - (2*B.1 - 3*B.2 + 6)|/Real.sqrt 13))) :=
by sorry


end triangle_lines_l1967_196763


namespace jessica_purchase_cost_l1967_196783

def chocolate_bars : ℕ := 10
def gummy_bears : ℕ := 10
def chocolate_chips : ℕ := 20

def price_chocolate_bar : ℕ := 3
def price_gummy_bears : ℕ := 2
def price_chocolate_chips : ℕ := 5

def total_cost : ℕ := chocolate_bars * price_chocolate_bar +
                      gummy_bears * price_gummy_bears +
                      chocolate_chips * price_chocolate_chips

theorem jessica_purchase_cost : total_cost = 150 := by
  sorry

end jessica_purchase_cost_l1967_196783


namespace multiplicative_inverse_207_mod_397_l1967_196778

theorem multiplicative_inverse_207_mod_397 :
  ∃ a : ℕ, a < 397 ∧ (207 * a) % 397 = 1 :=
by
  use 66
  sorry

end multiplicative_inverse_207_mod_397_l1967_196778


namespace water_tank_emptying_time_water_tank_empties_in_12_minutes_l1967_196736

/-- Represents the time it takes to empty a water tank given specific conditions. -/
theorem water_tank_emptying_time 
  (initial_fill : ℚ) 
  (pipe_a_fill_rate : ℚ) 
  (pipe_b_empty_rate : ℚ) : ℚ :=
  let net_rate := pipe_a_fill_rate - pipe_b_empty_rate
  let time_to_empty := initial_fill / (-net_rate)
  by
    -- Assuming initial_fill = 4/5
    -- pipe_a_fill_rate = 1/10
    -- pipe_b_empty_rate = 1/6
    sorry

/-- The main theorem stating it takes 12 minutes to empty the tank. -/
theorem water_tank_empties_in_12_minutes : 
  water_tank_emptying_time (4/5) (1/10) (1/6) = 12 :=
by sorry

end water_tank_emptying_time_water_tank_empties_in_12_minutes_l1967_196736


namespace defective_product_selection_l1967_196710

theorem defective_product_selection (n m k : ℕ) (hn : n = 100) (hm : m = 98) (hk : k = 3) :
  let total := n
  let qualified := m
  let defective := n - m
  let select := k
  Nat.choose n k - Nat.choose m k = 
    Nat.choose defective 1 * Nat.choose qualified 2 + 
    Nat.choose defective 2 * Nat.choose qualified 1 :=
by sorry

end defective_product_selection_l1967_196710


namespace sum_of_digits_product_76_eights_76_fives_l1967_196781

/-- Represents a number consisting of n repetitions of a single digit -/
def repeatedDigitNumber (digit : Nat) (n : Nat) : Nat :=
  digit * (10^n - 1) / 9

/-- Calculates the sum of digits in a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem to be proved -/
theorem sum_of_digits_product_76_eights_76_fives : 
  sumOfDigits (repeatedDigitNumber 8 76 * repeatedDigitNumber 5 76) = 304 := by
  sorry


end sum_of_digits_product_76_eights_76_fives_l1967_196781


namespace area_of_triangle_MOI_l1967_196786

/-- Triangle ABC with given side lengths -/
structure Triangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ

/-- Circumcenter of a triangle -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Point M such that a circle centered at M is tangent to AC, BC, and the circumcircle -/
def tangentPoint (t : Triangle) : ℝ × ℝ := sorry

/-- Area of a triangle given three points -/
def triangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

/-- Main theorem -/
theorem area_of_triangle_MOI (t : Triangle) 
  (h1 : t.AB = 15) (h2 : t.AC = 8) (h3 : t.BC = 7) : 
  triangleArea (tangentPoint t) (circumcenter t) (incenter t) = 1.765 := by
  sorry

end area_of_triangle_MOI_l1967_196786


namespace least_number_for_divisibility_l1967_196731

theorem least_number_for_divisibility : ∃ (n : ℕ), n = 11 ∧
  (∀ (m : ℕ), m < n → ¬((1789 + m) % 5 = 0 ∧ (1789 + m) % 6 = 0 ∧ (1789 + m) % 4 = 0 ∧ (1789 + m) % 3 = 0)) ∧
  ((1789 + n) % 5 = 0 ∧ (1789 + n) % 6 = 0 ∧ (1789 + n) % 4 = 0 ∧ (1789 + n) % 3 = 0) :=
by sorry

end least_number_for_divisibility_l1967_196731


namespace k_increasing_on_neg_reals_l1967_196788

/-- The function k(x) = 3 - x is increasing on the interval (-∞, 0). -/
theorem k_increasing_on_neg_reals :
  StrictMonoOn (fun x : ℝ => 3 - x) (Set.Iio 0) := by
  sorry

end k_increasing_on_neg_reals_l1967_196788


namespace square_root_positive_l1967_196744

theorem square_root_positive (x : ℝ) (h : x > 0) : Real.sqrt x > 0 := by
  sorry

end square_root_positive_l1967_196744


namespace pink_flowers_in_bag_B_l1967_196704

theorem pink_flowers_in_bag_B : 
  let bag_A_red : ℕ := 6
  let bag_A_pink : ℕ := 3
  let bag_B_red : ℕ := 2
  let bag_B_pink : ℕ := 7
  let total_flowers_A : ℕ := bag_A_red + bag_A_pink
  let total_flowers_B : ℕ := bag_B_red + bag_B_pink
  let prob_pink_A : ℚ := bag_A_pink / total_flowers_A
  let prob_pink_B : ℚ := bag_B_pink / total_flowers_B
  let overall_prob_pink : ℚ := (prob_pink_A + prob_pink_B) / 2
  overall_prob_pink = 5555555555555556 / 10000000000000000 →
  bag_B_pink = 7 :=
by sorry

end pink_flowers_in_bag_B_l1967_196704


namespace val_money_value_l1967_196701

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 0.05

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The number of nickels Val initially has -/
def initial_nickels : ℕ := 20

/-- The number of dimes Val has -/
def dimes : ℕ := 3 * initial_nickels

/-- The number of additional nickels Val finds -/
def additional_nickels : ℕ := 2 * initial_nickels

/-- The total value of money Val has after taking the additional nickels -/
def total_value : ℚ := 
  (initial_nickels : ℚ) * nickel_value + 
  (dimes : ℚ) * dime_value + 
  (additional_nickels : ℚ) * nickel_value

theorem val_money_value : total_value = 9 := by
  sorry

end val_money_value_l1967_196701


namespace ellipse_ratio_l1967_196755

/-- Given an ellipse with semi-major axis a, semi-minor axis b, and semi-focal length c,
    if a² + b² - 3c² = 0, then (a+c)/(a-c) = 3 + 2√2 -/
theorem ellipse_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
    (h4 : a^2 + b^2 - 3*c^2 = 0) : (a + c) / (a - c) = 3 + 2 * Real.sqrt 2 := by
  sorry


end ellipse_ratio_l1967_196755


namespace fourth_root_inequality_l1967_196768

theorem fourth_root_inequality (x : ℝ) :
  (x ^ (1/4) - 3 / (x ^ (1/4) + 4) ≥ 0) ↔ (0 ≤ x ∧ x ≤ 81) :=
by sorry

end fourth_root_inequality_l1967_196768


namespace tuesday_to_monday_ratio_l1967_196700

/-- Proves the ratio of square feet painted on Tuesday to Monday is 2:1 -/
theorem tuesday_to_monday_ratio (monday : ℝ) (wednesday : ℝ) (total : ℝ) : 
  monday = 30 →
  wednesday = monday / 2 →
  total = monday + wednesday + (total - monday - wednesday) →
  (total - monday - wednesday) / monday = 2 := by
  sorry

end tuesday_to_monday_ratio_l1967_196700


namespace tony_pills_l1967_196720

/-- The number of pills left in Tony's bottle after his treatment --/
def pills_left : ℕ :=
  let initial_pills : ℕ := 50
  let first_two_days : ℕ := 2 * 3 * 2
  let next_three_days : ℕ := 1 * 3 * 3
  let last_day : ℕ := 2
  initial_pills - (first_two_days + next_three_days + last_day)

theorem tony_pills : pills_left = 27 := by
  sorry

end tony_pills_l1967_196720


namespace solution_set_when_a_is_one_range_of_a_given_condition_l1967_196771

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |2*x - 1|

-- Theorem 1
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≥ 2} = {x : ℝ | x ≤ 0 ∨ x ≥ 2/3} := by sorry

-- Theorem 2
theorem range_of_a_given_condition (a : ℝ) :
  (∀ x ∈ Set.Icc (1/2 : ℝ) 1, f a x ≤ 2*x) →
  a ∈ Set.Icc (-3/2 : ℝ) 0 := by sorry

end solution_set_when_a_is_one_range_of_a_given_condition_l1967_196771


namespace tom_initial_balloons_l1967_196750

/-- The number of balloons Tom gave to Fred -/
def balloons_given : ℕ := 16

/-- The number of balloons Tom has left -/
def balloons_left : ℕ := 14

/-- The initial number of balloons Tom had -/
def initial_balloons : ℕ := balloons_given + balloons_left

theorem tom_initial_balloons : initial_balloons = 30 := by
  sorry

end tom_initial_balloons_l1967_196750


namespace survey_sample_is_opinions_of_selected_parents_l1967_196790

/-- Represents a parent of a student -/
structure Parent : Type :=
  (id : ℕ)

/-- Represents an opinion on the school rule -/
structure Opinion : Type :=
  (value : Bool)

/-- Represents a school survey -/
structure Survey : Type :=
  (participants : Finset Parent)
  (opinions : Parent → Option Opinion)

/-- Definition of a sample in the context of this survey -/
def sample (s : Survey) : Set Opinion :=
  {o | ∃ p ∈ s.participants, s.opinions p = some o}

theorem survey_sample_is_opinions_of_selected_parents 
  (s : Survey) 
  (h_size : s.participants.card = 100) :
  sample s = {o | ∃ p ∈ s.participants, s.opinions p = some o} :=
by
  sorry

#check survey_sample_is_opinions_of_selected_parents

end survey_sample_is_opinions_of_selected_parents_l1967_196790


namespace thousands_digit_of_common_remainder_l1967_196725

theorem thousands_digit_of_common_remainder (n : ℕ) 
  (h1 : n > 1000000)
  (h2 : n % 40 = n % 625) : 
  (n / 1000) % 10 = 0 ∨ (n / 1000) % 10 = 5 := by
sorry

end thousands_digit_of_common_remainder_l1967_196725


namespace train_cars_distribution_l1967_196784

theorem train_cars_distribution (soldiers_train1 soldiers_train2 soldiers_train3 : ℕ) 
  (h1 : soldiers_train1 = 462)
  (h2 : soldiers_train2 = 546)
  (h3 : soldiers_train3 = 630) :
  let max_soldiers_per_car := Nat.gcd soldiers_train1 (Nat.gcd soldiers_train2 soldiers_train3)
  (cars_train1, cars_train2, cars_train3) = (soldiers_train1 / max_soldiers_per_car,
                                             soldiers_train2 / max_soldiers_per_car,
                                             soldiers_train3 / max_soldiers_per_car) →
  (cars_train1, cars_train2, cars_train3) = (11, 13, 15) := by
sorry

end train_cars_distribution_l1967_196784


namespace intersection_points_l1967_196751

/-- The set of possible values for k such that the graph of |z - 3| = 3|z + 3| 
    intersects the graph of |z| = k in exactly one point -/
def possible_k_values : Set ℝ :=
  {k : ℝ | k = 1.5 ∨ k = 6}

/-- The condition that |z - 3| = 3|z + 3| -/
def condition (z : ℂ) : Prop :=
  Complex.abs (z - 3) = 3 * Complex.abs (z + 3)

/-- The theorem stating that the only values of k for which the graph of |z - 3| = 3|z + 3| 
    intersects the graph of |z| = k in exactly one point are 1.5 and 6 -/
theorem intersection_points (k : ℝ) :
  (∃! z : ℂ, condition z ∧ Complex.abs z = k) ↔ k ∈ possible_k_values := by
  sorry

end intersection_points_l1967_196751


namespace angle_sum_pi_half_l1967_196787

theorem angle_sum_pi_half (α β : Real) (h1 : 0 < α) (h2 : α < π/2) (h3 : 0 < β) (h4 : β < π/2)
  (h5 : 3 * Real.sin α ^ 2 + 2 * Real.sin β ^ 2 = 1)
  (h6 : 3 * Real.sin (2 * α) - 2 * Real.sin (2 * β) = 0) :
  α + 2 * β = π/2 := by
sorry

end angle_sum_pi_half_l1967_196787


namespace square_root_theorem_l1967_196785

theorem square_root_theorem (x : ℝ) :
  Real.sqrt (x + 3) = 3 → (x + 3)^2 = 81 := by
  sorry

end square_root_theorem_l1967_196785


namespace complement_M_intersect_N_l1967_196730

def U : Set Int := {-2, -1, 0, 1, 2}

def M : Set Int := {y | ∃ x, y = 2^x}

def N : Set Int := {x | x^2 - x - 2 = 0}

theorem complement_M_intersect_N :
  (U \ M) ∩ N = {-1} := by sorry

end complement_M_intersect_N_l1967_196730


namespace tysons_races_l1967_196791

/-- Tyson's swimming races problem -/
theorem tysons_races (lake_speed ocean_speed race_distance total_time : ℝ) 
  (h1 : lake_speed = 3)
  (h2 : ocean_speed = 2.5)
  (h3 : race_distance = 3)
  (h4 : total_time = 11) : 
  ∃ (num_races : ℕ), 
    (num_races : ℝ) / 2 * (race_distance / lake_speed) + 
    (num_races : ℝ) / 2 * (race_distance / ocean_speed) = total_time ∧ 
    num_races = 10 := by
  sorry

#check tysons_races

end tysons_races_l1967_196791


namespace base6_subtraction_l1967_196759

-- Define a function to convert a list of digits in base 6 to a natural number
def base6ToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 6 * acc + d) 0

-- Define a function to convert a natural number to a list of digits in base 6
def natToBase6 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 6) ((m % 6) :: acc)
    aux n []

-- State the theorem
theorem base6_subtraction :
  let a := base6ToNat [5, 5, 5]
  let b := base6ToNat [5, 5]
  let c := base6ToNat [2, 0, 2]
  let result := base6ToNat [6, 1, 4]
  (a + b) - c = result := by sorry

end base6_subtraction_l1967_196759


namespace johns_paintball_expense_l1967_196712

/-- The amount John spends on paintballs per month -/
def monthly_paintball_expense (plays_per_month : ℕ) (boxes_per_play : ℕ) (cost_per_box : ℕ) : ℕ :=
  plays_per_month * boxes_per_play * cost_per_box

/-- Theorem stating John's monthly paintball expense -/
theorem johns_paintball_expense :
  monthly_paintball_expense 3 3 25 = 225 := by
  sorry

end johns_paintball_expense_l1967_196712


namespace ferry_position_after_202_trips_l1967_196740

/-- Represents the shore where the ferry can be docked --/
inductive Shore : Type
  | South : Shore
  | North : Shore

/-- Determines the shore where the ferry is docked after a given number of trips --/
def ferry_position (start : Shore) (trips : ℕ) : Shore :=
  if trips % 2 = 0 then start else
    match start with
    | Shore.South => Shore.North
    | Shore.North => Shore.South

/-- Theorem stating that after 202 trips starting from the south shore, the ferry ends up on the south shore --/
theorem ferry_position_after_202_trips :
  ferry_position Shore.South 202 = Shore.South := by
  sorry

end ferry_position_after_202_trips_l1967_196740


namespace distance_to_focus_is_two_l1967_196742

/-- Parabola E defined by y² = 4x -/
def parabola_E (x y : ℝ) : Prop := y^2 = 4*x

/-- Point P with coordinates (x₀, 2) -/
structure Point_P where
  x₀ : ℝ

/-- P lies on parabola E -/
def P_on_E (P : Point_P) : Prop := parabola_E P.x₀ 2

/-- Distance from a point to the focus of parabola E -/
def distance_to_focus (P : Point_P) : ℝ := sorry

theorem distance_to_focus_is_two (P : Point_P) (h : P_on_E P) : 
  distance_to_focus P = 2 := by sorry

end distance_to_focus_is_two_l1967_196742


namespace domain_sqrt_one_minus_x_squared_l1967_196747

theorem domain_sqrt_one_minus_x_squared (x : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) 1 ↔ 1 - x^2 ≥ 0 :=
sorry

end domain_sqrt_one_minus_x_squared_l1967_196747


namespace pyramid_volume_l1967_196733

theorem pyramid_volume (base_length : ℝ) (base_width : ℝ) (height : ℝ) :
  base_length = 1/2 →
  base_width = 1/4 →
  height = 1 →
  (1/3) * base_length * base_width * height = 1/24 := by
sorry

end pyramid_volume_l1967_196733


namespace bobby_candy_problem_l1967_196709

/-- Proves that Bobby ate 9 pieces of candy at the start -/
theorem bobby_candy_problem (initial : ℕ) (eaten_start : ℕ) (eaten_more : ℕ) (left : ℕ)
  (h1 : initial = 22)
  (h2 : eaten_more = 5)
  (h3 : left = 8)
  (h4 : initial = eaten_start + eaten_more + left) :
  eaten_start = 9 := by
  sorry

end bobby_candy_problem_l1967_196709


namespace prime_sum_1998_l1967_196702

theorem prime_sum_1998 (p q r : ℕ) (s t u : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h_eq : 1998 = p^s * q^t * r^u) : p + q + r = 42 := by
  sorry

end prime_sum_1998_l1967_196702


namespace intersection_complement_theorem_l1967_196713

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

theorem intersection_complement_theorem :
  N ∩ Mᶜ = {x : ℝ | 3 < x ∧ x ≤ 4} := by sorry

end intersection_complement_theorem_l1967_196713


namespace perimeter_MNO_value_l1967_196728

/-- A right prism with regular hexagonal bases -/
structure HexagonalPrism where
  height : ℝ
  base_side_length : ℝ

/-- A point on an edge of the prism -/
structure EdgePoint where
  fraction : ℝ  -- Fraction of the edge length from the base

/-- Triangle MNO formed by three points on different edges of the prism -/
structure TriangleMNO where
  prism : HexagonalPrism
  m : EdgePoint
  n : EdgePoint
  o : EdgePoint

/-- Calculate the perimeter of triangle MNO -/
def perimeter_MNO (t : TriangleMNO) : ℝ :=
  sorry

theorem perimeter_MNO_value (t : TriangleMNO) 
  (h1 : t.prism.height = 20)
  (h2 : t.prism.base_side_length = 10)
  (h3 : t.m.fraction = 1/3)
  (h4 : t.n.fraction = 1/4)
  (h5 : t.o.fraction = 1/2) :
  perimeter_MNO t = 10 + Real.sqrt (925/9) + 5 * Real.sqrt 5 :=
sorry

end perimeter_MNO_value_l1967_196728


namespace largest_convex_polygon_on_grid_l1967_196721

/-- Represents a point on the grid -/
structure GridPoint where
  x : ℕ
  y : ℕ
  hx : x < 2004
  hy : y < 2004

/-- Represents a convex polygon on the grid -/
structure ConvexPolygon where
  vertices : List GridPoint
  is_convex : Bool  -- We assume there's a function to check convexity

/-- The main theorem stating the largest possible n-gon on the grid -/
theorem largest_convex_polygon_on_grid :
  ∃ (p : ConvexPolygon), p.vertices.length = 561 ∧
  ∀ (q : ConvexPolygon), q.vertices.length ≤ 561 :=
sorry

end largest_convex_polygon_on_grid_l1967_196721


namespace log_equation_solution_l1967_196757

theorem log_equation_solution (x : ℝ) :
  x > 0 → ((Real.log x / Real.log 2) * (Real.log x / Real.log 5) = Real.log 10 / Real.log 2) ↔
  x = 2 ^ Real.sqrt (6 * Real.log 2) :=
sorry

end log_equation_solution_l1967_196757


namespace one_friend_no_meat_l1967_196707

/-- Represents the cookout scenario with given conditions -/
structure Cookout where
  total_friends : ℕ
  burgers_per_guest : ℕ
  buns_per_pack : ℕ
  packs_bought : ℕ
  friends_no_bread : ℕ

/-- Calculates the number of friends who don't eat meat -/
def friends_no_meat (c : Cookout) : ℕ :=
  c.total_friends - (c.packs_bought * c.buns_per_pack / c.burgers_per_guest + c.friends_no_bread)

/-- Theorem stating that exactly one friend doesn't eat meat -/
theorem one_friend_no_meat (c : Cookout) 
  (h1 : c.total_friends = 10)
  (h2 : c.burgers_per_guest = 3)
  (h3 : c.buns_per_pack = 8)
  (h4 : c.packs_bought = 3)
  (h5 : c.friends_no_bread = 1) :
  friends_no_meat c = 1 := by
  sorry

#eval friends_no_meat { 
  total_friends := 10, 
  burgers_per_guest := 3, 
  buns_per_pack := 8, 
  packs_bought := 3, 
  friends_no_bread := 1 
}

end one_friend_no_meat_l1967_196707


namespace simple_interest_rate_calculation_l1967_196782

/-- Given an initial sum of money that amounts to 9800 after 5 years
    and 12005 after 8 years at the same rate of simple interest,
    prove that the rate of interest per annum is 7.5% -/
theorem simple_interest_rate_calculation (P : ℝ) (R : ℝ) :
  P * (1 + 5 * R / 100) = 9800 →
  P * (1 + 8 * R / 100) = 12005 →
  R = 7.5 :=
by sorry

end simple_interest_rate_calculation_l1967_196782


namespace polynomial_expansion_l1967_196734

theorem polynomial_expansion :
  ∀ x : ℝ, (2 * x^2 - 3 * x + 1) * (x^2 + x + 3) = 2 * x^4 - x^3 + 4 * x^2 - 8 * x + 3 := by
  sorry

end polynomial_expansion_l1967_196734


namespace triangle_angle_measure_l1967_196772

theorem triangle_angle_measure (a b : ℝ) (A B : Real) :
  0 < a ∧ 0 < b ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π →
  b = 2 * a * Real.sin B →
  Real.sin A = 1 / 2 :=
sorry

end triangle_angle_measure_l1967_196772


namespace bus_children_count_l1967_196789

/-- The total number of children on a bus after more children got on is equal to the sum of the initial number of children and the additional children who got on. -/
theorem bus_children_count (initial_children additional_children : ℕ) :
  initial_children + additional_children = initial_children + additional_children :=
by sorry

end bus_children_count_l1967_196789


namespace inequality_proof_l1967_196764

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 ∧
  ((x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2*y ∧ y = z) :=
sorry

end inequality_proof_l1967_196764


namespace bake_sale_cookies_l1967_196743

/-- The number of lemon cookies Marcus brought to the bake sale -/
def lemon_cookies : ℕ := 20

/-- The number of peanut butter cookies Jenny brought -/
def jenny_pb_cookies : ℕ := 40

/-- The number of chocolate chip cookies Jenny brought -/
def jenny_cc_cookies : ℕ := 50

/-- The number of peanut butter cookies Marcus brought -/
def marcus_pb_cookies : ℕ := 30

/-- The total number of cookies at the bake sale -/
def total_cookies : ℕ := jenny_pb_cookies + jenny_cc_cookies + marcus_pb_cookies + lemon_cookies

/-- The probability of picking a peanut butter cookie -/
def pb_probability : ℚ := 1/2

theorem bake_sale_cookies : 
  (jenny_pb_cookies + marcus_pb_cookies : ℚ) / total_cookies = pb_probability := by sorry

end bake_sale_cookies_l1967_196743


namespace carols_age_ratio_l1967_196756

theorem carols_age_ratio (carol alice betty : ℕ) : 
  carol = 5 * alice →
  alice = carol - 12 →
  betty = 6 →
  (carol : ℚ) / betty = 5 / 2 := by
  sorry

end carols_age_ratio_l1967_196756


namespace inequality_system_solution_l1967_196711

theorem inequality_system_solution (x : ℝ) :
  3 * x > x - 4 ∧ (4 + x) / 3 > x + 2 → -2 < x ∧ x < -1 := by
  sorry

end inequality_system_solution_l1967_196711


namespace value_of_a_l1967_196705

theorem value_of_a (a b c d : ℤ) 
  (eq1 : 2 * a + 2 = b)
  (eq2 : 2 * b + 2 = c)
  (eq3 : 2 * c + 2 = d)
  (eq4 : 2 * d + 2 = 62) : 
  a = 2 := by
sorry

end value_of_a_l1967_196705


namespace total_spending_theorem_l1967_196761

/-- Represents the spending of Terry, Maria, and Raj over a week -/
structure WeeklySpending where
  terry : List Float
  maria : List Float
  raj : List Float

/-- Calculates the total spending of all three people over the week -/
def totalSpending (ws : WeeklySpending) : Float :=
  (ws.terry.sum + ws.maria.sum + ws.raj.sum)

/-- Theorem stating that the total spending equals $752.50 -/
theorem total_spending_theorem (ws : WeeklySpending) : 
  ws.terry = [6, 12, 36, 18, 14, 21, 33] ∧ 
  ws.maria = [3, 10, 72, 8, 14, 12, 33] ∧ 
  ws.raj = [7.5, 10, 216, 108, 14, 21, 84] → 
  totalSpending ws = 752.5 := by
  sorry

#eval totalSpending {
  terry := [6, 12, 36, 18, 14, 21, 33],
  maria := [3, 10, 72, 8, 14, 12, 33],
  raj := [7.5, 10, 216, 108, 14, 21, 84]
}

end total_spending_theorem_l1967_196761


namespace cube_volume_surface_area_l1967_196706

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s > 0 ∧ s^3 = 8*x ∧ 6*s^2 = 2*x) → x = 1728 :=
by sorry

end cube_volume_surface_area_l1967_196706


namespace andys_hourly_wage_l1967_196703

/-- Calculates Andy's hourly wage based on his shift earnings and activities. -/
theorem andys_hourly_wage (shift_hours : ℕ) (racquets_strung : ℕ) (grommets_changed : ℕ) (stencils_painted : ℕ)
  (restring_pay : ℕ) (grommet_pay : ℕ) (stencil_pay : ℕ) (total_earnings : ℕ) :
  shift_hours = 8 →
  racquets_strung = 7 →
  grommets_changed = 2 →
  stencils_painted = 5 →
  restring_pay = 15 →
  grommet_pay = 10 →
  stencil_pay = 1 →
  total_earnings = 202 →
  (total_earnings - (racquets_strung * restring_pay + grommets_changed * grommet_pay + stencils_painted * stencil_pay)) / shift_hours = 9 :=
by sorry

end andys_hourly_wage_l1967_196703


namespace vector_calculation_l1967_196714

def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![1, -1]

theorem vector_calculation :
  (1/3 : ℝ) • a - (4/3 : ℝ) • b = ![(-1 : ℝ), 2] := by sorry

end vector_calculation_l1967_196714


namespace smallest_n_divisible_by_1419_l1967_196732

def product_of_evens (n : Nat) : Nat :=
  (List.range ((n / 2) - 1)).foldl (fun acc i => acc * (2 * (i + 2))) 2

theorem smallest_n_divisible_by_1419 :
  (∀ m : Nat, m < 106 → m % 2 = 0 → ¬(product_of_evens m % 1419 = 0)) ∧
  (106 % 2 = 0 ∧ product_of_evens 106 % 1419 = 0) := by
  sorry

end smallest_n_divisible_by_1419_l1967_196732


namespace inequality_solution_l1967_196792

theorem inequality_solution (x : ℝ) :
  x ≠ -4 ∧ x ≠ -10/3 →
  ((2*x + 3) / (x + 4) > (4*x + 5) / (3*x + 10) ↔ 
   x < -5/2 ∨ x > -2) :=
by sorry

end inequality_solution_l1967_196792


namespace temperature_conversion_l1967_196739

theorem temperature_conversion (t k : ℝ) : 
  t = 5 / 9 * (k - 32) → t = 105 → k = 221 := by
sorry

end temperature_conversion_l1967_196739


namespace tom_read_70_books_l1967_196769

/-- The number of books Tom read each month -/
def books_per_month : List Nat := [2, 6, 12, 20, 30]

/-- The total number of books Tom read over five months -/
def total_books : Nat := books_per_month.sum

theorem tom_read_70_books : total_books = 70 := by
  sorry

end tom_read_70_books_l1967_196769


namespace cost_price_of_article_l1967_196777

/-- 
Given an article where the profit when selling it for Rs. 57 is equal to the loss 
when selling it for Rs. 43, prove that the cost price of the article is Rs. 50.
-/
theorem cost_price_of_article (cost_price : ℕ) : cost_price = 50 := by
  sorry

/--
Helper function to calculate profit
-/
def profit (selling_price cost_price : ℕ) : ℤ :=
  (selling_price : ℤ) - (cost_price : ℤ)

/--
Helper function to calculate loss
-/
def loss (cost_price selling_price : ℕ) : ℤ :=
  (cost_price : ℤ) - (selling_price : ℤ)

/--
Assumption that profit when selling at Rs. 57 equals loss when selling at Rs. 43
-/
axiom profit_loss_equality (cost_price : ℕ) :
  profit 57 cost_price = loss cost_price 43

end cost_price_of_article_l1967_196777


namespace largest_solution_of_equation_l1967_196738

theorem largest_solution_of_equation :
  let f (x : ℚ) := 7 * (9 * x^2 + 11 * x + 12) - x * (9 * x - 46)
  ∃ (x : ℚ), f x = 0 ∧ (∀ (y : ℚ), f y = 0 → y ≤ x) ∧ x = -7/6 := by
  sorry

end largest_solution_of_equation_l1967_196738


namespace license_plate_theorem_l1967_196779

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The number of letter positions on the license plate -/
def letter_positions : ℕ := 6

/-- The number of digit positions on the license plate -/
def digit_positions : ℕ := 3

/-- The number of possible license plate combinations with 6 letters 
(where exactly two different letters are repeated once) followed by 3 non-repeating digits -/
def license_plate_combinations : ℕ :=
  (Nat.choose alphabet_size 2) *
  (Nat.choose letter_positions 2) *
  (Nat.choose (letter_positions - 2) 2) *
  (Nat.choose (alphabet_size - 2) 2) *
  (10 * 9 * 8)

theorem license_plate_theorem : license_plate_combinations = 84563400000 := by
  sorry

end license_plate_theorem_l1967_196779


namespace integral_one_plus_sin_x_l1967_196754

theorem integral_one_plus_sin_x : ∫ x in (0)..(π/2), (1 + Real.sin x) = π/2 + 1 := by
  sorry

end integral_one_plus_sin_x_l1967_196754


namespace michael_twice_jacob_age_l1967_196765

/-- 
Given that Jacob is 13 years younger than Michael and Jacob will be 8 years old in 4 years,
this theorem proves that Michael will be twice as old as Jacob in 9 years.
-/
theorem michael_twice_jacob_age (jacob_current_age : ℕ) (michael_current_age : ℕ) :
  jacob_current_age + 4 = 8 →
  michael_current_age = jacob_current_age + 13 →
  ∃ (years : ℕ), years = 9 ∧ michael_current_age + years = 2 * (jacob_current_age + years) :=
by sorry

end michael_twice_jacob_age_l1967_196765


namespace num_common_tangents_for_given_circles_l1967_196760

/-- Circle represented by its equation in the form (x - h)² + (y - k)² = r² --/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- Number of common tangents between two circles --/
def num_common_tangents (c1 c2 : Circle) : ℕ :=
  sorry

/-- Convert from x² + y² - 2ax = 0 form to Circle structure --/
def circle_from_equation (a : ℝ) : Circle :=
  { h := a, k := 0, r := a }

theorem num_common_tangents_for_given_circles :
  let c1 := circle_from_equation 1
  let c2 := circle_from_equation 2
  num_common_tangents c1 c2 = 1 := by
  sorry

end num_common_tangents_for_given_circles_l1967_196760


namespace nut_is_composed_of_prism_and_cylinder_l1967_196758

-- Define the types for geometric bodies
inductive GeometricBody
| RegularHexagonalPrism
| Cylinder
| Nut

-- Define the composition of a nut
def nut_composition : List GeometricBody := [GeometricBody.RegularHexagonalPrism, GeometricBody.Cylinder]

-- Theorem statement
theorem nut_is_composed_of_prism_and_cylinder :
  (GeometricBody.Nut ∈ [GeometricBody.RegularHexagonalPrism, GeometricBody.Cylinder]) →
  (nut_composition.length = 2) →
  (nut_composition = [GeometricBody.RegularHexagonalPrism, GeometricBody.Cylinder]) :=
by sorry

end nut_is_composed_of_prism_and_cylinder_l1967_196758


namespace divisible_by_nine_l1967_196735

theorem divisible_by_nine (h : ℕ) (h_single_digit : h < 10) :
  (7600 + 100 * h + 4) % 9 = 0 ↔ h = 1 := by
  sorry

end divisible_by_nine_l1967_196735


namespace smallest_drama_club_size_l1967_196795

theorem smallest_drama_club_size : ∃ n : ℕ, n > 0 ∧ 
  (∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = n ∧
    (22 * n < 100 * a) ∧ (100 * a < 27 * n) ∧
    (25 * n < 100 * b) ∧ (100 * b < 35 * n) ∧
    (35 * n < 100 * c) ∧ (100 * c < 45 * n)) ∧
  (∀ m : ℕ, m > 0 → m < n →
    ¬(∃ a b c : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧
      a + b + c = m ∧
      (22 * m < 100 * a) ∧ (100 * a < 27 * m) ∧
      (25 * m < 100 * b) ∧ (100 * b < 35 * m) ∧
      (35 * m < 100 * c) ∧ (100 * c < 45 * m))) ∧
  n = 9 :=
by sorry

end smallest_drama_club_size_l1967_196795


namespace sum_of_powers_of_five_squares_l1967_196793

theorem sum_of_powers_of_five_squares (m n : ℕ+) :
  (∃ a b : ℤ, (5 : ℤ)^(n : ℕ) + (5 : ℤ)^(m : ℕ) = a^2 + b^2) ↔ Even (n - m) := by
  sorry

end sum_of_powers_of_five_squares_l1967_196793


namespace ratio_evaluation_and_closest_integer_l1967_196715

theorem ratio_evaluation_and_closest_integer : 
  let r := (2^3000 + 2^3003) / (2^3001 + 2^3002)
  r = 3/2 ∧ ∀ n : ℤ, |r - 2| ≤ |r - n| :=
by
  sorry

end ratio_evaluation_and_closest_integer_l1967_196715


namespace work_completion_time_l1967_196719

/-- Given workers A, B, and C with their individual work rates, 
    prove that B and C together can complete the work in 3 hours. -/
theorem work_completion_time 
  (rate_A : ℝ) 
  (rate_B : ℝ) 
  (rate_C : ℝ) 
  (h1 : rate_A = 1 / 4) 
  (h2 : rate_A + rate_C = 1 / 2) 
  (h3 : rate_B = 1 / 12) : 
  1 / (rate_B + rate_C) = 3 := by
sorry

end work_completion_time_l1967_196719


namespace smallest_b_in_arithmetic_progression_l1967_196749

theorem smallest_b_in_arithmetic_progression (a b c : ℝ) : 
  0 < a ∧ 0 < b ∧ 0 < c →
  (∃ d : ℝ, a = b - d ∧ c = b + d) →
  a * b * c = 125 →
  b ≥ 5 := by
sorry

end smallest_b_in_arithmetic_progression_l1967_196749


namespace persimmons_in_jungkooks_house_l1967_196722

theorem persimmons_in_jungkooks_house : 
  let num_boxes : ℕ := 4
  let persimmons_per_box : ℕ := 5
  num_boxes * persimmons_per_box = 20 := by
  sorry

end persimmons_in_jungkooks_house_l1967_196722


namespace cube_three_minus_seven_equals_square_four_plus_four_l1967_196741

theorem cube_three_minus_seven_equals_square_four_plus_four :
  3^3 - 7 = 4^2 + 4 := by
  sorry

end cube_three_minus_seven_equals_square_four_plus_four_l1967_196741


namespace largest_solution_quadratic_l1967_196716

theorem largest_solution_quadratic : 
  let f : ℝ → ℝ := fun y ↦ 6 * y^2 - 31 * y + 35
  ∃ y : ℝ, f y = 0 ∧ ∀ z : ℝ, f z = 0 → z ≤ y ∧ y = (5 : ℝ) / 2 := by sorry

end largest_solution_quadratic_l1967_196716


namespace circle_symmetry_l1967_196797

-- Define the given circle
def given_circle (x y : ℝ) : Prop :=
  (x + 2)^2 + y^2 = 5

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop :=
  x - y + 1 = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x + 1)^2 + (y + 1)^2 = 5

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y x' y' : ℝ),
    given_circle x y →
    symmetry_line ((x + x') / 2) ((y + y') / 2) →
    symmetric_circle x' y' :=
by sorry

end circle_symmetry_l1967_196797


namespace smallest_perfect_square_divisible_by_5_and_6_l1967_196729

theorem smallest_perfect_square_divisible_by_5_and_6 :
  ∀ n : ℕ, n > 0 → n * n < 900 → ¬(5 ∣ (n * n) ∧ 6 ∣ (n * n)) :=
by sorry

end smallest_perfect_square_divisible_by_5_and_6_l1967_196729


namespace plate_arrangement_theorem_l1967_196775

/-- The number of ways to arrange plates around a circular table. -/
def circularArrangements (total : ℕ) (blue red green orange : ℕ) : ℕ :=
  (Nat.factorial total) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial orange) / total

/-- The number of ways to arrange plates around a circular table with one pair adjacent. -/
def circularArrangementsWithPairAdjacent (total : ℕ) (blue red green orange : ℕ) : ℕ :=
  (Nat.factorial (total - 1)) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial orange) / (total - 1)

/-- The number of ways to arrange plates around a circular table with both pairs adjacent. -/
def circularArrangementsWithBothPairsAdjacent (total : ℕ) (blue red green orange : ℕ) : ℕ :=
  (Nat.factorial (total - 2)) / (Nat.factorial blue * Nat.factorial red * Nat.factorial green * Nat.factorial orange) / (total - 2)

theorem plate_arrangement_theorem :
  let total := 11
  let blue := 6
  let red := 2
  let green := 2
  let orange := 1
  circularArrangements total blue red green orange -
  circularArrangementsWithPairAdjacent total blue red green orange -
  circularArrangementsWithPairAdjacent total blue red green orange +
  circularArrangementsWithBothPairsAdjacent total blue red green orange = 1568 := by
  sorry

end plate_arrangement_theorem_l1967_196775


namespace sock_combination_count_l1967_196708

/-- The number of ways to choose a pair of socks of different colors with at least one blue sock. -/
def sock_combinations (white brown blue : ℕ) : ℕ :=
  (blue * white) + (blue * brown)

/-- Theorem: Given 5 white socks, 3 brown socks, and 4 blue socks, there are 32 ways to choose
    a pair of socks of different colors with at least one blue sock. -/
theorem sock_combination_count :
  sock_combinations 5 3 4 = 32 := by
  sorry

end sock_combination_count_l1967_196708


namespace characterize_inequality_l1967_196774

theorem characterize_inequality (x y : ℝ) :
  x^2 * y - y ≥ 0 ↔ (y ≥ 0 ∧ abs x ≥ 1) ∨ (y ≤ 0 ∧ abs x ≤ 1) := by
  sorry

end characterize_inequality_l1967_196774


namespace distance_y_to_earth_l1967_196724

-- Define the distances
def distance_earth_to_x : ℝ := 0.5
def distance_x_to_y : ℝ := 0.1
def total_distance : ℝ := 0.7

-- Theorem to prove
theorem distance_y_to_earth : 
  total_distance - (distance_earth_to_x + distance_x_to_y) = 0.1 := by
  sorry

end distance_y_to_earth_l1967_196724


namespace sin_sum_of_zero_points_l1967_196723

/-- Given that x₁ and x₂ are two zero points of f(x) = 2sin(2x) + cos(2x) - m
    within the interval [0, π/2], prove that sin(x₁ + x₂) = 2√5/5 -/
theorem sin_sum_of_zero_points (x₁ x₂ m : ℝ) : 
  x₁ ∈ Set.Icc 0 (π/2) →
  x₂ ∈ Set.Icc 0 (π/2) →
  2 * Real.sin (2 * x₁) + Real.cos (2 * x₁) - m = 0 →
  2 * Real.sin (2 * x₂) + Real.cos (2 * x₂) - m = 0 →
  Real.sin (x₁ + x₂) = 2 * Real.sqrt 5 / 5 :=
by sorry

end sin_sum_of_zero_points_l1967_196723


namespace f_satisfies_conditions_f_formula_correct_l1967_196753

/-- Given two distinct real numbers α and β, we define a function f on natural numbers. -/
noncomputable def f (α β : ℝ) (n : ℕ) : ℝ :=
  (α^(n+1) - β^(n+1)) / (α - β)

/-- The main theorem stating that f satisfies the given recurrence relation and initial conditions. -/
theorem f_satisfies_conditions (α β : ℝ) (h : α ≠ β) :
  (f α β 1 = (α^2 - β^2) / (α - β)) ∧
  (f α β 2 = (α^3 - β^3) / (α - β)) ∧
  (∀ n : ℕ, f α β (n+2) = (α + β) * f α β (n+1) - α * β * f α β n) :=
by sorry

/-- The main theorem proving that the given formula for f is correct for all natural numbers. -/
theorem f_formula_correct (α β : ℝ) (h : α ≠ β) (n : ℕ) :
  f α β n = (α^(n+1) - β^(n+1)) / (α - β) :=
by sorry

end f_satisfies_conditions_f_formula_correct_l1967_196753


namespace grain_output_scientific_notation_l1967_196717

/-- Represents the total grain output of China in 2021 in tons -/
def china_grain_output : ℝ := 682.85e6

/-- The scientific notation representation of China's grain output -/
def scientific_notation : ℝ := 6.8285e8

/-- Theorem stating that the grain output is equal to its scientific notation representation -/
theorem grain_output_scientific_notation : china_grain_output = scientific_notation := by
  sorry

end grain_output_scientific_notation_l1967_196717


namespace symmetry_implies_periodicity_l1967_196766

/-- A function f: ℝ → ℝ is symmetric with respect to the point (a, y₀) -/
def SymmetricPoint (f : ℝ → ℝ) (a y₀ : ℝ) : Prop :=
  ∀ x, f (a + x) - y₀ = y₀ - f (a - x)

/-- A function f: ℝ → ℝ is symmetric with respect to the line x = b -/
def SymmetricLine (f : ℝ → ℝ) (b : ℝ) : Prop :=
  ∀ x, f (b + x) = f (b - x)

/-- A function f: ℝ → ℝ is periodic with period p -/
def Periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem symmetry_implies_periodicity (f : ℝ → ℝ) (a b y₀ : ℝ) (hb : b > a) 
    (h1 : SymmetricPoint f a y₀) (h2 : SymmetricLine f b) :
    Periodic f (4 * (b - a)) := by
  sorry

end symmetry_implies_periodicity_l1967_196766


namespace angle_BAD_measure_l1967_196737

-- Define the geometric configuration
structure GeometricConfiguration where
  -- We don't need to explicitly define points, just the angles
  angleABC : ℝ
  angleBDE : ℝ
  angleDBE : ℝ
  -- We'll define angleABD in terms of angleABC

-- Define the theorem
theorem angle_BAD_measure (config : GeometricConfiguration) 
  (h1 : config.angleABC = 132)
  (h2 : config.angleBDE = 31)
  (h3 : config.angleDBE = 30)
  : 180 - (180 - config.angleABC) - config.angleBDE - config.angleDBE = 119 := by
  sorry


end angle_BAD_measure_l1967_196737


namespace red_balls_count_l1967_196799

/-- Given a bag of 16 balls with red and blue balls, if the probability of drawing
    exactly 2 red balls when 3 are drawn at random is 1/10, then there are 7 red balls. -/
theorem red_balls_count (total : ℕ) (red : ℕ) (blue : ℕ) :
  total = 16 ∧
  total = red + blue ∧
  (Nat.choose red 2 * blue : ℚ) / Nat.choose total 3 = 1 / 10 →
  red = 7 := by
  sorry

end red_balls_count_l1967_196799


namespace expression_evaluation_l1967_196776

theorem expression_evaluation (a b : ℤ) (ha : a = -1) (hb : b = 1) :
  (a - 2*b) * (a^2 + 2*a*b + 4*b^2) - a * (a - 5*b) * (a + 3*b) = -21 :=
by sorry

end expression_evaluation_l1967_196776


namespace count_nines_in_subtraction_l1967_196748

/-- The number of occurrences of a digit in a natural number -/
def countDigit (n : ℕ) (d : ℕ) : ℕ := sorry

/-- The result of subtracting 101011 from 10000000000 -/
def subtraction_result : ℕ := 10000000000 - 101011

/-- Theorem stating that the number of 9's in the subtraction result is 8 -/
theorem count_nines_in_subtraction : countDigit subtraction_result 9 = 8 := by sorry

end count_nines_in_subtraction_l1967_196748


namespace local_election_vote_count_l1967_196746

theorem local_election_vote_count (candidate1_percent : ℝ) (candidate2_percent : ℝ) 
  (candidate3_percent : ℝ) (candidate4_percent : ℝ) (candidate2_votes : ℕ) :
  candidate1_percent = 0.45 →
  candidate2_percent = 0.25 →
  candidate3_percent = 0.20 →
  candidate4_percent = 0.10 →
  candidate2_votes = 600 →
  candidate1_percent + candidate2_percent + candidate3_percent + candidate4_percent = 1 →
  ∃ (total_votes : ℕ), total_votes = 2400 ∧ 
    (candidate2_percent : ℝ) * total_votes = candidate2_votes := by
  sorry

end local_election_vote_count_l1967_196746


namespace complex_fraction_evaluation_l1967_196762

/-- Given a = 15, b = 19, c = 25, and S = a + b + c = 59, prove that the expression
    (a² * (1/b - 1/c) + b² * (1/c - 1/a) + c² * (1/a - 1/b) + 37) /
    (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b) + 2)
    equals 77.5 -/
theorem complex_fraction_evaluation (a b c S : ℚ) 
    (ha : a = 15) (hb : b = 19) (hc : c = 25) (hS : S = a + b + c) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b) + 37) /
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b) + 2) = 77.5 := by
  sorry

end complex_fraction_evaluation_l1967_196762


namespace chess_tournament_results_l1967_196745

/-- Represents a chess tournament with given conditions -/
structure ChessTournament where
  n : ℕ  -- number of players
  total_score : ℕ
  one_player_score : ℕ
  h1 : total_score = 210
  h2 : one_player_score = 12
  h3 : n * (n - 1) = total_score

/-- Theorem stating the main results of the tournament analysis -/
theorem chess_tournament_results (t : ChessTournament) :
  (t.n = 15) ∧ 
  (∃ (max_squares : ℕ), max_squares = 33 ∧ 
    ∀ (squares : ℕ), (squares = number_of_squares_knight_can_reach_in_two_moves) → 
      squares ≤ max_squares) ∧
  (∃ (winner_score : ℕ), winner_score > t.one_player_score) :=
sorry

/-- Helper function to calculate the number of squares a knight can reach in two moves -/
def number_of_squares_knight_can_reach_in_two_moves : ℕ :=
sorry

end chess_tournament_results_l1967_196745


namespace orthocenter_locus_l1967_196798

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

-- Define the secant line
def SecantLine (K : ℝ × ℝ) (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * (p.1 - K.1)}

-- Define the orthocenter of a triangle
def Orthocenter (A P Q : ℝ × ℝ) : ℝ × ℝ :=
  sorry

-- State the theorem
theorem orthocenter_locus
  (O : ℝ × ℝ)
  (r k : ℝ)
  (h_k : k > r) :
  ∀ (A : ℝ × ℝ) (m : ℝ),
  A ∈ Circle O r →
  ∃ (P Q : ℝ × ℝ),
  P ∈ Circle O r ∩ SecantLine (k, 0) m ∧
  Q ∈ Circle O r ∩ SecantLine (k, 0) m ∧
  P ≠ Q ∧
  let H := Orthocenter A P Q
  (H.1 - 2*k*m^2/(m^2 + 1))^2 + (H.2 + 2*k*m/(m^2 + 1))^2 = r^2 :=
sorry

end orthocenter_locus_l1967_196798


namespace curve_equation_min_distance_l1967_196726

-- Define the points A and B
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (3, 0)

-- Define the curve C
def C : Set (ℝ × ℝ) := {P | (P.1 - 5)^2 + P.2^2 = 16}

-- Define the line l1
def l1 : Set (ℝ × ℝ) := {Q | Q.1 + Q.2 + 3 = 0}

-- Theorem for the equation of curve C
theorem curve_equation :
  C = {P : ℝ × ℝ | Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) = 2 * Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)} :=
sorry

-- Theorem for the minimum distance
theorem min_distance :
  ∀ Q ∈ l1, ∃ M ∈ C, ∀ M' ∈ C,
    (∃ t : ℝ, M' = (1 - t) • Q + t • M) →
    Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) ≤ Real.sqrt ((M'.1 - Q.1)^2 + (M'.2 - Q.2)^2) ∧
    Real.sqrt ((M.1 - Q.1)^2 + (M.2 - Q.2)^2) = 4 :=
sorry

end curve_equation_min_distance_l1967_196726


namespace max_quarters_sasha_l1967_196727

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 25 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The total value Sasha has in dollars -/
def total_value : ℚ := 210 / 100

theorem max_quarters_sasha : 
  ∃ (q : ℕ), q * quarter_value + q * dime_value = total_value ∧ 
  ∀ (n : ℕ), n * quarter_value + n * dime_value = total_value → n ≤ q :=
by sorry

end max_quarters_sasha_l1967_196727
