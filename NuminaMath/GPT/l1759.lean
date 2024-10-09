import Mathlib

namespace red_car_initial_distance_ahead_l1759_175978

theorem red_car_initial_distance_ahead 
    (Speed_red Speed_black : ℕ) (Time : ℝ)
    (H1 : Speed_red = 10)
    (H2 : Speed_black = 50)
    (H3 : Time = 0.5) :
    let Distance_black := Speed_black * Time
    let Distance_red := Speed_red * Time
    Distance_black - Distance_red = 20 := 
by
  let Distance_black := Speed_black * Time
  let Distance_red := Speed_red * Time
  sorry

end red_car_initial_distance_ahead_l1759_175978


namespace rectangle_area_l1759_175919

theorem rectangle_area (x : ℝ) (w : ℝ) (h : w^2 + (2 * w)^2 = x^2) : 
  2 * (w^2) = (2 / 5) * x^2 :=
by
  sorry

end rectangle_area_l1759_175919


namespace sum_of_eight_digits_l1759_175912

open Nat

theorem sum_of_eight_digits {a b c d e f g h : ℕ} 
  (h_distinct : ∀ i j, i ∈ [a, b, c, d, e, f, g, h] → j ∈ [a, b, c, d, e, f, g, h] → i ≠ j → i ≠ j)
  (h_vertical_sum : a + b + c + d + e = 25)
  (h_horizontal_sum : f + g + h + b = 15) 
  (h_digits_set : ∀ x ∈ [a, b, c, d, e, f, g, h], x ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9} : Set ℕ)) : 
  a + b + c + d + e + f + g + h - b = 39 := 
sorry

end sum_of_eight_digits_l1759_175912


namespace total_fish_is_22_l1759_175931

def gold_fish : ℕ := 15
def blue_fish : ℕ := 7
def total_fish : ℕ := gold_fish + blue_fish

theorem total_fish_is_22 : total_fish = 22 :=
by
  -- the proof should be written here
  sorry

end total_fish_is_22_l1759_175931


namespace giant_kite_area_72_l1759_175973

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

end giant_kite_area_72_l1759_175973


namespace running_speed_l1759_175969

variables (w t_w t_r : ℝ)

-- Given conditions
def walking_speed : w = 8 := sorry
def walking_time_hours : t_w = 4.75 := sorry
def running_time_hours : t_r = 2 := sorry

-- Prove the man's running speed
theorem running_speed (w t_w t_r : ℝ) 
  (H1 : w = 8) 
  (H2 : t_w = 4.75) 
  (H3 : t_r = 2) : 
  (w * t_w) / t_r = 19 := 
sorry

end running_speed_l1759_175969


namespace simplify_expr_l1759_175972

theorem simplify_expr : 
  (3^2015 + 3^2013) / (3^2015 - 3^2013) = (5 : ℚ) / 4 := 
by
  sorry

end simplify_expr_l1759_175972


namespace possible_values_of_ABCD_l1759_175939

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

end possible_values_of_ABCD_l1759_175939


namespace find_a_l1759_175971

theorem find_a (f : ℝ → ℝ) (h1 : ∀ x, f (2^x) = x + 3) (h2 : f a = 5) : a = 4 := 
by
  sorry

end find_a_l1759_175971


namespace plane_equivalent_l1759_175988

def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (3 + 2*s - 3*t, 1 + s, 4 - 3*s + t)

def plane_equation (x y z : ℝ) : Prop :=
  x - 7*y + 3*z - 8 = 0

theorem plane_equivalent :
  ∃ (s t : ℝ), parametric_plane s t = (x, y, z) ↔ plane_equation x y z :=
by
  sorry

end plane_equivalent_l1759_175988


namespace probability_no_adjacent_same_color_l1759_175999

-- Define the problem space
def total_beads : ℕ := 9
def red_beads : ℕ := 4
def white_beads : ℕ := 3
def blue_beads : ℕ := 2

-- Define the total number of arrangements
def total_arrangements := Nat.factorial total_beads / (Nat.factorial red_beads * Nat.factorial white_beads * Nat.factorial blue_beads)

-- State the probability computation theorem
theorem probability_no_adjacent_same_color :
  (∃ valid_arrangements : ℕ,
     valid_arrangements / total_arrangements = 1 / 63) := sorry

end probability_no_adjacent_same_color_l1759_175999


namespace hannah_probability_12_flips_l1759_175943

/-!
We need to prove that the probability of getting fewer than 4 heads when flipping 12 coins is 299/4096.
-/

def probability_fewer_than_4_heads (flips : ℕ) : ℚ :=
  let total_outcomes := 2^flips
  let favorable_outcomes := (Nat.choose flips 0) + (Nat.choose flips 1) + (Nat.choose flips 2) + (Nat.choose flips 3)
  favorable_outcomes / total_outcomes

theorem hannah_probability_12_flips : probability_fewer_than_4_heads 12 = 299 / 4096 := by
  sorry

end hannah_probability_12_flips_l1759_175943


namespace cyclic_quadrilateral_JMIT_l1759_175976

theorem cyclic_quadrilateral_JMIT
  (a b c : ℂ)
  (I J M N T : ℂ)
  (hI : I = -(a*b + b*c + c*a))
  (hJ : J = a*b - b*c + c*a)
  (hM : M = (b^2 + c^2) / 2)
  (hN : N = b*c)
  (hT : T = 2*a^2 - b*c) :
  ∃ (k : ℝ), k = ((M - I) * (T - J)) / ((J - I) * (T - M)) :=
by
  sorry

end cyclic_quadrilateral_JMIT_l1759_175976


namespace solution_set_eq_l1759_175933

noncomputable def f (x : ℝ) : ℝ := x^6 + x^2
noncomputable def g (x : ℝ) : ℝ := (2*x + 3)^3 + 2*x + 3

theorem solution_set_eq : {x : ℝ | f x = g x} = {-1, 3} :=
by
  sorry

end solution_set_eq_l1759_175933


namespace april_total_earned_l1759_175951

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

end april_total_earned_l1759_175951


namespace cells_remain_illuminated_l1759_175904

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

end cells_remain_illuminated_l1759_175904


namespace factorize_expression_l1759_175924

variable (x y : ℝ)

theorem factorize_expression : 
  (y - 2 * x * y + x^2 * y) = y * (1 - x)^2 := 
by
  sorry

end factorize_expression_l1759_175924


namespace train_pass_man_time_l1759_175937

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

end train_pass_man_time_l1759_175937


namespace percent_of_games_lost_l1759_175930

theorem percent_of_games_lost (w l : ℕ) (h1 : w / l = 8 / 5) (h2 : w + l = 65) :
  (l * 100 / 65 : ℕ) = 38 :=
sorry

end percent_of_games_lost_l1759_175930


namespace determine_parallel_planes_l1759_175981

def Plane : Type := sorry
def Line : Type := sorry
def Parallel (x y : Line) : Prop := sorry
def Skew (x y : Line) : Prop := sorry
def PlaneParallel (α β : Plane) : Prop := sorry

variables (α β : Plane) (a b : Line)
variable (hSkew : Skew a b)
variable (hαa : Parallel a α) 
variable (hαb : Parallel b α)
variable (hβa : Parallel a β)
variable (hβb : Parallel b β)

theorem determine_parallel_planes : PlaneParallel α β := sorry

end determine_parallel_planes_l1759_175981


namespace find_unknown_value_l1759_175905

theorem find_unknown_value (x : ℝ) (h : (3 + 5 + 6 + 8 + x) / 5 = 7) : x = 13 :=
by
  sorry

end find_unknown_value_l1759_175905


namespace points_on_same_side_after_25_seconds_l1759_175957

def movement_time (side_length : ℕ) (perimeter : ℕ)
  (speed_A speed_B : ℕ) (start_mid_B : ℕ) : ℕ :=
  25

theorem points_on_same_side_after_25_seconds (side_length : ℕ) (perimeter : ℕ)
  (speed_A speed_B : ℕ) (start_mid_B : ℕ) :
  side_length = 100 ∧ perimeter = 400 ∧ speed_A = 5 ∧ speed_B = 10 ∧ start_mid_B = 50 →
  movement_time side_length perimeter speed_A speed_B start_mid_B = 25 :=
by
  intros h
  sorry

end points_on_same_side_after_25_seconds_l1759_175957


namespace min_balls_to_ensure_20_l1759_175996

theorem min_balls_to_ensure_20 (red green yellow blue purple white black : ℕ) (Hred : red = 30) (Hgreen : green = 25) (Hyellow : yellow = 18) (Hblue : blue = 15) (Hpurple : purple = 12) (Hwhite : white = 10) (Hblack : black = 7) :
  ∀ n, n ≥ 101 → (∃ r g y b p w bl, r + g + y + b + p + w + bl = n ∧ (r ≥ 20 ∨ g ≥ 20 ∨ y ≥ 20 ∨ b ≥ 20 ∨ p ≥ 20 ∨ w ≥ 20 ∨ bl ≥ 20)) :=
by
  intro n hn
  sorry

end min_balls_to_ensure_20_l1759_175996


namespace max_area_rectangle_shorter_side_l1759_175906

theorem max_area_rectangle_shorter_side (side_length : ℕ) (n : ℕ)
  (hsq : side_length = 40) (hn : n = 5) :
  ∃ (shorter_side : ℕ), shorter_side = 8 := by
  sorry

end max_area_rectangle_shorter_side_l1759_175906


namespace sticker_count_l1759_175964

def stickers_per_page : ℕ := 25
def num_pages : ℕ := 35
def total_stickers : ℕ := 875

theorem sticker_count : num_pages * stickers_per_page = total_stickers :=
by {
  sorry
}

end sticker_count_l1759_175964


namespace max_product_of_two_integers_sum_2000_l1759_175907

theorem max_product_of_two_integers_sum_2000 : 
  ∃ x y : ℤ, x + y = 2000 ∧ x * y = 1000000 := by
  sorry

end max_product_of_two_integers_sum_2000_l1759_175907


namespace length_HD_is_3_l1759_175927

noncomputable def square_side : ℝ := 8

noncomputable def midpoint_AD : ℝ := square_side / 2

noncomputable def length_FD : ℝ := midpoint_AD

theorem length_HD_is_3 :
  ∃ (x : ℝ), 0 < x ∧ x < square_side ∧ (8 - x) ^ 2 = x ^ 2 + length_FD ^ 2 ∧ x = 3 :=
by
  sorry

end length_HD_is_3_l1759_175927


namespace inequality_proof_l1759_175917

theorem inequality_proof (x : ℝ) (hx : x ≥ 1) : x^5 - 1 / x^4 ≥ 9 * (x - 1) := 
by sorry

end inequality_proof_l1759_175917


namespace time_to_finish_typing_l1759_175941

-- Definitions
def words_per_minute : ℕ := 38
def total_words : ℕ := 4560

-- Theorem to prove
theorem time_to_finish_typing : (total_words / words_per_minute) / 60 = 2 := by
  sorry

end time_to_finish_typing_l1759_175941


namespace greatest_integer_x_l1759_175967

theorem greatest_integer_x (x : ℤ) (h : 7 - 3 * x + 2 > 23) : x ≤ -5 :=
by {
  sorry
}

end greatest_integer_x_l1759_175967


namespace range_of_a_l1759_175995

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, 
    1 ≤ x ∧ x ≤ 2 ∧ 
    2 ≤ y ∧ y ≤ 3 → 
    x * y ≤ a * x^2 + 2 * y^2) ↔ 
  a ≥ -1 :=
by
  sorry

end range_of_a_l1759_175995


namespace mn_sum_eq_neg_one_l1759_175902

theorem mn_sum_eq_neg_one (m n : ℤ) (h : (∀ x : ℤ, (x + 2) * (x - 1) = x^2 + m * x + n)) :
  m + n = -1 :=
sorry

end mn_sum_eq_neg_one_l1759_175902


namespace coeff_sum_zero_l1759_175991

theorem coeff_sum_zero (a₀ a₁ a₂ a₃ a₄ : ℝ) (h : ∀ x : ℝ, (2*x + 1)^4 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4) :
  a₁ + a₂ + a₃ + a₄ = 0 :=
by
  sorry

end coeff_sum_zero_l1759_175991


namespace people_in_room_l1759_175922

theorem people_in_room (P C : ℚ) (H1 : (3 / 5) * P = (2 / 3) * C) (H2 : C / 3 = 5) : 
  P = 50 / 3 :=
by
  -- The proof would go here
  sorry

end people_in_room_l1759_175922


namespace cos_neg_pi_over_3_l1759_175998

theorem cos_neg_pi_over_3 : Real.cos (-π / 3) = 1 / 2 :=
by
  sorry

end cos_neg_pi_over_3_l1759_175998


namespace problem_l1759_175960

theorem problem (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y + x * y = 3) :
  (0 < x * y ∧ x * y ≤ 1) ∧ (∀ z : ℝ, z = x + 2 * y → z = 4 * Real.sqrt 2 - 3) :=
by
  sorry

end problem_l1759_175960


namespace higher_concentration_acid_solution_l1759_175953

theorem higher_concentration_acid_solution (x : ℝ) (h1 : 2 * (8 / 100 : ℝ) = 1.2 * (x / 100) + 0.8 * (5 / 100)) : x = 10 :=
sorry

end higher_concentration_acid_solution_l1759_175953


namespace correct_operation_l1759_175914

theorem correct_operation : (3 * a^2 * b^3 - 2 * a^2 * b^3 = a^2 * b^3) ∧ 
                            ¬(a^2 * a^3 = a^6) ∧ 
                            ¬(a^6 / a^2 = a^3) ∧ 
                            ¬((a^2)^3 = a^5) :=
by
  sorry

end correct_operation_l1759_175914


namespace dubblefud_red_balls_l1759_175985

theorem dubblefud_red_balls (R B G : ℕ) 
  (h1 : 3^R * 7^B * 11^G = 5764801)
  (h2 : B = G) :
  R = 7 :=
by
  sorry

end dubblefud_red_balls_l1759_175985


namespace samuel_teacups_left_l1759_175921

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

end samuel_teacups_left_l1759_175921


namespace solution_set_inequality_l1759_175901

theorem solution_set_inequality (x : ℝ) : (0 < x ∧ x < 1) ↔ (1 / (x - 1) < -1) :=
by
  sorry

end solution_set_inequality_l1759_175901


namespace annes_initial_bottle_caps_l1759_175961

-- Define the conditions
def albert_bottle_caps : ℕ := 9
def annes_added_bottle_caps : ℕ := 5
def annes_total_bottle_caps : ℕ := 15

-- Question (to prove)
theorem annes_initial_bottle_caps :
  annes_total_bottle_caps - annes_added_bottle_caps = 10 :=
by sorry

end annes_initial_bottle_caps_l1759_175961


namespace Nadine_pebbles_l1759_175938

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

end Nadine_pebbles_l1759_175938


namespace solve_for_x_l1759_175900

theorem solve_for_x (x : ℝ) (h : 3 - (1 / (2 - x)) = (1 / (2 - x))) : x = 4 / 3 := 
by {
  sorry
}

end solve_for_x_l1759_175900


namespace solve_for_b_l1759_175923

theorem solve_for_b (b : ℚ) (h : b + b / 4 = 5 / 2) : b = 2 := 
sorry

end solve_for_b_l1759_175923


namespace man_l1759_175983

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

end man_l1759_175983


namespace sum_abs_arithmetic_sequence_l1759_175992

variable (n : ℕ)

def S_n (n : ℕ) : ℚ :=
  - ((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 + ((205 : ℕ) / (2 : ℕ) : ℚ) * n

def T_n (n : ℕ) : ℚ :=
  if n ≤ 34 then
    -((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 + ((205 : ℕ) / (2 : ℕ) : ℚ) * n
  else
    ((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 - ((205 : ℕ) / (2 : ℕ) : ℚ) * n + 3502

theorem sum_abs_arithmetic_sequence :
  T_n n = (if n ≤ 34 then -((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 + ((205 : ℕ) / (2 : ℕ) : ℚ) * n
           else ((3 : ℕ) / (2 : ℕ) : ℚ) * n^2 - ((205 : ℕ) / (2 : ℕ) : ℚ) * n + 3502) :=
by sorry

end sum_abs_arithmetic_sequence_l1759_175992


namespace find_a_l1759_175987

variable {a : ℝ}

def p (a : ℝ) := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > -1 ∧ x₂ > -1 ∧ x₁ * x₁ + 2 * a * x₁ + 1 = 0 ∧ x₂ * x₂ + 2 * a * x₂ + 1 = 0

def q (a : ℝ) := ∀ x : ℝ, a * x * x - a * x + 1 > 0 

theorem find_a (a : ℝ) : (p a ∨ q a) ∧ ¬ q a → a ≤ -1 :=
sorry

end find_a_l1759_175987


namespace Joan_pays_139_20_l1759_175993

noncomputable def JKL : Type := ℝ × ℝ × ℝ

def conditions (J K L : ℝ) : Prop :=
  J + K + L = 600 ∧
  2 * J = K + 74 ∧
  L = K + 52

theorem Joan_pays_139_20 (J K L : ℝ) (h : conditions J K L) : J = 139.20 :=
by
  sorry

end Joan_pays_139_20_l1759_175993


namespace farmer_field_l1759_175974

theorem farmer_field (m : ℤ) : 
  (3 * m + 8) * (m - 3) = 85 → m = 6 :=
by
  sorry

end farmer_field_l1759_175974


namespace compare_abc_l1759_175986

/-- Define the constants a, b, and c as given in the problem -/
noncomputable def a : ℝ := -5 / 4 * Real.log (4 / 5)
noncomputable def b : ℝ := Real.exp (1 / 4) / 4
noncomputable def c : ℝ := 1 / 3

/-- The theorem to be proved: a < b < c -/
theorem compare_abc : a < b ∧ b < c :=
by
  sorry

end compare_abc_l1759_175986


namespace sin_value_of_arithmetic_sequence_l1759_175908

open Real

def arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d, ∀ n, a (n + 1) = a n + d

theorem sin_value_of_arithmetic_sequence (a : ℕ → ℝ) 
  (h_arith_seq : arithmetic_sequence a) 
  (h_cond : a 1 + a 5 + a 9 = 5 * π) : 
  sin (a 2 + a 8) = - (sqrt 3 / 2) :=
by
  sorry

end sin_value_of_arithmetic_sequence_l1759_175908


namespace find_a_100_l1759_175980

noncomputable def a : Nat → Nat
| 0 => 0
| 1 => 2
| (n+1) => a n + 2 * n

theorem find_a_100 : a 100 = 9902 := 
  sorry

end find_a_100_l1759_175980


namespace quadratic_transformation_l1759_175903

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

end quadratic_transformation_l1759_175903


namespace tom_total_seashells_l1759_175975

-- Define the number of seashells Tom gave to Jessica.
def seashells_given_to_jessica : ℕ := 2

-- Define the number of seashells Tom still has.
def seashells_tom_has_now : ℕ := 3

-- Theorem stating that the total number of seashells Tom found is the sum of seashells_given_to_jessica and seashells_tom_has_now.
theorem tom_total_seashells : seashells_given_to_jessica + seashells_tom_has_now = 5 := 
by
  sorry

end tom_total_seashells_l1759_175975


namespace find_real_a_l1759_175928

open Complex

noncomputable def pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

theorem find_real_a (a : ℝ) (i : ℂ) (h_i : i = Complex.I) :
  pure_imaginary ((2 + i) * (a - (2 * i))) ↔ a = -1 :=
by
  sorry

end find_real_a_l1759_175928


namespace fraction_equality_l1759_175994

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := 2 * x - 3

theorem fraction_equality :
  (f (g (f 3))) / (g (f (g 3))) = 59 / 35 := 
by
  sorry

end fraction_equality_l1759_175994


namespace part1_part2_l1759_175942

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

end part1_part2_l1759_175942


namespace green_ball_probability_l1759_175935

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

end green_ball_probability_l1759_175935


namespace discount_amount_correct_l1759_175989

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

end discount_amount_correct_l1759_175989


namespace girls_with_brown_eyes_and_light_brown_skin_l1759_175963

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

end girls_with_brown_eyes_and_light_brown_skin_l1759_175963


namespace ten_pow_n_plus_eight_div_nine_is_integer_l1759_175977

theorem ten_pow_n_plus_eight_div_nine_is_integer (n : ℕ) : ∃ k : ℤ, 10^n + 8 = 9 * k := 
sorry

end ten_pow_n_plus_eight_div_nine_is_integer_l1759_175977


namespace halfway_fraction_l1759_175909

-- Assume a definition for the two fractions
def fracA : ℚ := 1 / 4
def fracB : ℚ := 1 / 7

-- Define the target property we want to prove
theorem halfway_fraction : (fracA + fracB) / 2 = 11 / 56 := 
by 
  -- Proof will happen here, adding sorry to indicate it's skipped for now
  sorry

end halfway_fraction_l1759_175909


namespace proof_b_lt_a_lt_c_l1759_175948

noncomputable def a : ℝ := 2^(4/5)
noncomputable def b : ℝ := 4^(2/7)
noncomputable def c : ℝ := 25^(1/5)

theorem proof_b_lt_a_lt_c : b < a ∧ a < c := by
  sorry

end proof_b_lt_a_lt_c_l1759_175948


namespace no_psafe_numbers_l1759_175982

def is_psafe (n p : ℕ) : Prop := 
  ¬ (n % p = 0 ∨ n % p = 1 ∨ n % p = 2 ∨ n % p = 3 ∨ n % p = p - 3 ∨ n % p = p - 2 ∨ n % p = p - 1)

theorem no_psafe_numbers (N : ℕ) (hN : N = 10000) :
  ∀ n, (n ≤ N ∧ is_psafe n 5 ∧ is_psafe n 7 ∧ is_psafe n 11) → false :=
by
  sorry

end no_psafe_numbers_l1759_175982


namespace problem_solution_l1759_175952

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

end problem_solution_l1759_175952


namespace factorization_of_x12_minus_4096_l1759_175920

variable (x : ℝ)

theorem factorization_of_x12_minus_4096 : 
  x^12 - 4096 = (x^6 + 64) * (x + 2) * (x^2 - 2*x + 4) * (x - 2) * (x^2 + 2*x + 4) := by
  sorry

end factorization_of_x12_minus_4096_l1759_175920


namespace expected_value_correct_l1759_175984

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

end expected_value_correct_l1759_175984


namespace find_k_l1759_175925

noncomputable def arithmetic_sequence_sum (a₁ d : ℕ) (n : ℕ) : ℕ :=
  n * a₁ + (n * (n-1)) / 2 * d

theorem find_k (a₁ d : ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h₁ : a₁ = 1) (h₂ : d = 2) (h₃ : ∀ n, S (n+2) = 28 + S n) :
  k = 6 := by
  sorry

end find_k_l1759_175925


namespace probability_of_c_between_l1759_175913

noncomputable def probability_c_between (a b : ℝ) (hab : 0 < a ∧ a ≤ 1 ∧ 0 < b ∧ b ≤ 1) : ℝ :=
  let c := a / (a + b)
  if (1 / 4 : ℝ) ≤ c ∧ c ≤ (3 / 4 : ℝ) then sorry else sorry
  
theorem probability_of_c_between (a b : ℝ) (hab : 0 < a ∧ a ≤ 1 ∧ 0 < b ∧ b ≤ 1) : 
  probability_c_between a b hab = (2 / 3 : ℝ) :=
sorry

end probability_of_c_between_l1759_175913


namespace sodium_acetate_formed_is_3_l1759_175945

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

end sodium_acetate_formed_is_3_l1759_175945


namespace sum_a1_a5_l1759_175990

theorem sum_a1_a5 (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = n^2 + 1)
  (ha : ∀ n, a (n + 1) = S (n + 1) - S n) :
  a 1 + a 5 = 11 :=
sorry

end sum_a1_a5_l1759_175990


namespace sean_whistles_l1759_175944

def charles_whistles : ℕ := 128
def sean_more_whistles : ℕ := 95

theorem sean_whistles : charles_whistles + sean_more_whistles = 223 :=
by {
  sorry
}

end sean_whistles_l1759_175944


namespace work_completion_days_l1759_175955

theorem work_completion_days (Dx : ℕ) (Dy : ℕ) (days_y_worked : ℕ) (days_x_finished_remaining : ℕ)
  (work_rate_y : ℝ) (work_rate_x : ℝ) 
  (h1 : Dy = 24)
  (h2 : days_y_worked = 12)
  (h3 : days_x_finished_remaining = 18)
  (h4 : work_rate_y = 1 / Dy)
  (h5 : 12 * work_rate_y = 1 / 2)
  (h6 : work_rate_x = 1 / (2 * days_x_finished_remaining))
  (h7 : Dx * work_rate_x = 1) : Dx = 36 := sorry

end work_completion_days_l1759_175955


namespace mail_sorting_time_l1759_175918

theorem mail_sorting_time :
  (1 / (1 / 3 + 1 / 6) = 2) :=
by
  sorry

end mail_sorting_time_l1759_175918


namespace solve_inequality_system_l1759_175926

theorem solve_inequality_system
  (x : ℝ)
  (h1 : 3 * (x - 1) < 5 * x + 11)
  (h2 : 2 * x > (9 - x) / 4) :
  x > 1 :=
sorry

end solve_inequality_system_l1759_175926


namespace conjugate_axis_length_l1759_175970

variable (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
variable (e : ℝ) (h3 : e = Real.sqrt 7 / 2)
variable (c : ℝ) (h4 : c = a * e)
variable (P : ℝ × ℝ) (h5 : P = (c, b^2 / a))
variable (F1 F2 : ℝ × ℝ) (h6 : F1 = (-c, 0)) (h7 : F2 = (c, 0))
variable (h8 : dist P F2 = 9 / 2)
variable (h9 : P.1 = c) (h10 : P.2 = b^2 / a)
variable (h11 : PF_2 ⊥ F_1F_2)

theorem conjugate_axis_length : 2 * b = 6 * Real.sqrt 3 := by
  sorry

end conjugate_axis_length_l1759_175970


namespace geometric_sequence_at_t_l1759_175916

theorem geometric_sequence_at_t (a : ℕ → ℕ) (S : ℕ → ℕ) (t : ℕ) :
  (∀ n, a n = a 1 * (3 ^ (n - 1))) →
  a 1 = 1 →
  S t = (a 1 * (1 - 3 ^ t)) / (1 - 3) →
  S t = 364 →
  a t = 243 :=
by {
  sorry
}

end geometric_sequence_at_t_l1759_175916


namespace cost_of_soccer_basketball_balls_max_basketballs_l1759_175958

def cost_of_balls (x y : ℕ) : Prop :=
  (7 * x = 5 * y) ∧ (40 * x + 20 * y = 3400)

def cost_constraint (x y m : ℕ) : Prop :=
  (x = 50) ∧ (y = 70) ∧ (70 * m + 50 * (100 - m) ≤ 6300)

theorem cost_of_soccer_basketball_balls (x y : ℕ) (h : cost_of_balls x y) : x = 50 ∧ y = 70 :=
  by sorry

theorem max_basketballs (x y m : ℕ) (h : cost_constraint x y m) : m ≤ 65 :=
  by sorry

end cost_of_soccer_basketball_balls_max_basketballs_l1759_175958


namespace positive_real_number_solution_l1759_175936

theorem positive_real_number_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 11) (h3 : (x - 6) / 11 = 6 / (x - 11)) : x = 17 :=
sorry

end positive_real_number_solution_l1759_175936


namespace expected_value_of_third_flip_l1759_175910

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

end expected_value_of_third_flip_l1759_175910


namespace john_made_money_l1759_175965

theorem john_made_money 
  (repair_cost : ℕ := 20000) 
  (discount_percentage : ℕ := 20) 
  (prize_money : ℕ := 70000) 
  (keep_percentage : ℕ := 90) : 
  (prize_money * keep_percentage / 100) - (repair_cost - (repair_cost * discount_percentage / 100)) = 47000 := 
by 
  sorry

end john_made_money_l1759_175965


namespace singleBase12Digit_l1759_175940

theorem singleBase12Digit (n : ℕ) : 
  (7 ^ 6 ^ 5 ^ 3 ^ 2 ^ 1) % 11 = 4 :=
sorry

end singleBase12Digit_l1759_175940


namespace jack_cleaning_time_is_one_hour_l1759_175968

def jackGrove : ℕ × ℕ := (4, 5)
def timeToCleanEachTree : ℕ := 6
def timeReductionFactor : ℕ := 2
def totalCleaningTimeWithHelpMin : ℕ :=
  (jackGrove.fst * jackGrove.snd) * (timeToCleanEachTree / timeReductionFactor)
def totalCleaningTimeWithHelpHours : ℕ :=
  totalCleaningTimeWithHelpMin / 60

theorem jack_cleaning_time_is_one_hour :
  totalCleaningTimeWithHelpHours = 1 := by
  sorry

end jack_cleaning_time_is_one_hour_l1759_175968


namespace solution_set_of_abs_inequality_l1759_175950

theorem solution_set_of_abs_inequality :
  {x : ℝ // |2 * x - 1| < 3} = {x : ℝ // -1 < x ∧ x < 2} :=
by sorry

end solution_set_of_abs_inequality_l1759_175950


namespace geometric_sequence_solution_l1759_175934

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) :=
  ∀ n, a n = a1 * q ^ (n - 1)

theorem geometric_sequence_solution {a : ℕ → ℝ} {q a1 : ℝ}
  (h1 : geometric_sequence a q a1)
  (h2 : a 3 + a 5 = 20)
  (h3 : a 4 = 8) :
  a 2 + a 6 = 34 := by
  sorry

end geometric_sequence_solution_l1759_175934


namespace find_a_plus_b_l1759_175956

theorem find_a_plus_b (x a b : ℝ) (ha : x = a + Real.sqrt b)
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : x^2 + 5 * x + 4/x + 1/(x^2) = 34) : a + b = 5 :=
sorry

end find_a_plus_b_l1759_175956


namespace part1_part2_l1759_175962

open Set

-- Definitions from conditions in a)
def R : Set ℝ := univ
def A : Set ℝ := {x | (x + 2) * (x - 3) < 0}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Question part (1)
theorem part1 (a : ℝ) (h : a = 1) :
  (compl A) ∪ B a = {x | x ≤ -2 ∨ x > 1} :=
by 
  simp [h]
  sorry

-- Question part (2)
theorem part2 (a : ℝ) :
  A ⊆ B a → a ≤ -2 :=
by 
  sorry

end part1_part2_l1759_175962


namespace three_pow_zero_l1759_175997

theorem three_pow_zero : 3^0 = 1 :=
by sorry

end three_pow_zero_l1759_175997


namespace smallest_number_with_sum_32_and_distinct_digits_l1759_175966

def is_distinct (l : List Nat) : Prop := l.Nodup

def sum_of_digits (n : Nat) : Nat :=
  (Nat.digits 10 n).sum

theorem smallest_number_with_sum_32_and_distinct_digits :
  ∀ n : Nat, is_distinct (Nat.digits 10 n) ∧ sum_of_digits n = 32 → 26789 ≤ n :=
by
  intro n
  intro h
  sorry

end smallest_number_with_sum_32_and_distinct_digits_l1759_175966


namespace cost_of_pencils_and_notebooks_l1759_175915

variable (P N : ℝ)

theorem cost_of_pencils_and_notebooks
  (h1 : 4 * P + 3 * N = 9600)
  (h2 : 2 * P + 2 * N = 5400) :
  8 * P + 7 * N = 20400 := by
  sorry

end cost_of_pencils_and_notebooks_l1759_175915


namespace value_of_a_l1759_175979

noncomputable def f (x a : ℝ) : ℝ := 2 * x^2 - 3 * x - Real.log x + Real.exp (x - a) + 4 * Real.exp (a - x)

theorem value_of_a (a x0 : ℝ) (h : f x0 a = 3) : a = 1 - Real.log 2 :=
by
  sorry

end value_of_a_l1759_175979


namespace weight_of_B_l1759_175932

theorem weight_of_B (A B C : ℝ) (h1 : (A + B + C) / 3 = 45) (h2 : (A + B) / 2 = 40) (h3 : (B + C) / 2 = 46) : B = 37 :=
by
  sorry

end weight_of_B_l1759_175932


namespace exists_unique_inverse_l1759_175911

theorem exists_unique_inverse (p : ℕ) (a : ℕ) (hp : Nat.Prime p) (h_gcd : Nat.gcd p a = 1) : 
  ∃! (b : ℕ), b ∈ Finset.range p ∧ (a * b) % p = 1 := 
sorry

end exists_unique_inverse_l1759_175911


namespace problem_statement_l1759_175949

def reading_method (n : ℕ) : String := sorry
-- Assume reading_method correctly implements the reading method for integers

def is_read_with_only_one_zero (n : ℕ) : Prop :=
  (reading_method n).count '0' = 1

theorem problem_statement : is_read_with_only_one_zero 83721000 = false := sorry

end problem_statement_l1759_175949


namespace john_subtraction_number_l1759_175946

theorem john_subtraction_number (a b : ℕ) (h1 : a = 40) (h2 : b = 1) :
  40^2 - ((2 * 40 * 1) - 1^2) = 39^2 :=
by
  -- sorry indicates the proof is skipped
  sorry

end john_subtraction_number_l1759_175946


namespace combine_octahedrons_tetrahedrons_to_larger_octahedron_l1759_175959

theorem combine_octahedrons_tetrahedrons_to_larger_octahedron (edge : ℝ) :
  ∃ (octahedrons : ℕ) (tetrahedrons : ℕ),
    octahedrons = 6 ∧ tetrahedrons = 8 ∧
    (∃ (new_octahedron_edge : ℝ), new_octahedron_edge = 2 * edge) :=
by {
  -- The proof will construct the larger octahedron
  sorry
}

end combine_octahedrons_tetrahedrons_to_larger_octahedron_l1759_175959


namespace remainder_div_polynomial_l1759_175947

theorem remainder_div_polynomial :
  ∀ (x : ℝ), 
  ∃ (Q : ℝ → ℝ) (R : ℝ → ℝ), 
    R x = (3^101 - 2^101) * x + (2^101 - 2 * 3^101) ∧
    x^101 = (x^2 - 5 * x + 6) * Q x + R x :=
by
  sorry

end remainder_div_polynomial_l1759_175947


namespace moving_circle_passes_through_fixed_point_l1759_175954
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

end moving_circle_passes_through_fixed_point_l1759_175954


namespace distance_on_third_day_is_36_difference_between_longest_and_shortest_is_57_average_daily_distance_is_50_l1759_175929

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

end distance_on_third_day_is_36_difference_between_longest_and_shortest_is_57_average_daily_distance_is_50_l1759_175929
