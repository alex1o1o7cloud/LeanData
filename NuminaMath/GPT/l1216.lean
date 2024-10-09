import Mathlib

namespace union_of_sets_eq_A_l1216_121668

noncomputable def A : Set ℝ := {x | x / ((x + 1) * (x - 4)) < 0}
noncomputable def B : Set ℝ := {x | Real.log x < 1}

theorem union_of_sets_eq_A: A ∪ B = A := by
  sorry

end union_of_sets_eq_A_l1216_121668


namespace interest_credited_cents_l1216_121661

theorem interest_credited_cents (P : ℝ) (rt : ℝ) (A : ℝ) (interest : ℝ) :
  A = 255.31 →
  rt = 1 + 0.05 * (1/6) →
  P = A / rt →
  interest = A - P →
  (interest * 100) % 100 = 10 :=
by
  intro hA
  intro hrt
  intro hP
  intro hint
  sorry

end interest_credited_cents_l1216_121661


namespace calculate_perimeter_l1216_121628

def four_squares_area : ℝ := 144 -- total area of the figure in cm²
noncomputable def area_of_one_square : ℝ := four_squares_area / 4 -- area of one square in cm²
noncomputable def side_length_of_square : ℝ := Real.sqrt area_of_one_square -- side length of one square in cm

def number_of_vertical_segments : ℕ := 4 -- based on the arrangement
def number_of_horizontal_segments : ℕ := 6 -- based on the arrangement

noncomputable def total_perimeter : ℝ := (number_of_vertical_segments + number_of_horizontal_segments) * side_length_of_square

theorem calculate_perimeter : total_perimeter = 60 := by
  sorry

end calculate_perimeter_l1216_121628


namespace find_a_find_b_l1216_121642

section Problem1

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^4 - 4 * x^3 + a * x^2 - 1

-- Condition 1: f is monotonically increasing on [0, 1]
def f_increasing_on_interval_01 (a : ℝ) : Prop :=
  ∀ x y, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ x ≤ y → f x a ≤ f y a

-- Condition 2: f is monotonically decreasing on [1, 2]
def f_decreasing_on_interval_12 (a : ℝ) : Prop :=
  ∀ x y, 1 ≤ x ∧ x ≤ 2 ∧ 1 ≤ y ∧ y ≤ 2 ∧ x ≤ y → f y a ≤ f x a

-- Proof of a part
theorem find_a : ∃ a, f_increasing_on_interval_01 a ∧ f_decreasing_on_interval_12 a ∧ a = 4 :=
  sorry

end Problem1

section Problem2

noncomputable def f_fixed (x : ℝ) : ℝ := x^4 - 4 * x^3 + 4 * x^2 - 1
noncomputable def g (x : ℝ) (b : ℝ) : ℝ := b * x^2 - 1

-- Condition for intersections
def intersect_at_two_points (b : ℝ) : Prop :=
  ∃ x1 x2, x1 ≠ x2 ∧ f_fixed x1 = g x1 b ∧ f_fixed x2 = g x2 b

-- Proof of b part
theorem find_b : ∃ b, intersect_at_two_points b ∧ (b = 0 ∨ b = 4) :=
  sorry

end Problem2

end find_a_find_b_l1216_121642


namespace expand_expression_l1216_121687

theorem expand_expression (x : ℝ) :
  (2 * x + 3) * (4 * x - 5) = 8 * x^2 + 2 * x - 15 :=
by
  sorry

end expand_expression_l1216_121687


namespace remaining_volume_of_cube_with_hole_l1216_121604

theorem remaining_volume_of_cube_with_hole : 
  let side_length_cube := 8 
  let side_length_hole := 4 
  let volume_cube := side_length_cube ^ 3 
  let cross_section_hole := side_length_hole ^ 2
  let volume_hole := cross_section_hole * side_length_cube
  let remaining_volume := volume_cube - volume_hole
  remaining_volume = 384 := by {
    sorry
  }

end remaining_volume_of_cube_with_hole_l1216_121604


namespace largest_of_three_l1216_121633

theorem largest_of_three (a b c : ℕ) (h1 : a = 5) (h2 : b = 8) (h3 : c = 4) : max a (max b c) = 8 := 
sorry

end largest_of_three_l1216_121633


namespace cost_of_magazine_l1216_121606

theorem cost_of_magazine (B M : ℝ) 
  (h1 : 2 * B + 2 * M = 26) 
  (h2 : B + 3 * M = 27) : 
  M = 7 := 
by 
  sorry

end cost_of_magazine_l1216_121606


namespace inequality_proof_l1216_121657

theorem inequality_proof (x y z : ℝ) 
  (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z)
  (hx1 : x ≤ 1) (hy1 : y ≤ 1) (hz1 : z ≤ 1) :
  2 * (x^3 + y^3 + z^3) - (x^2 * y + y^2 * z + z^2 * x) ≤ 3 :=
by
  sorry

end inequality_proof_l1216_121657


namespace find_A_l1216_121665

theorem find_A (A : ℕ) (h1 : A < 5) (h2 : (9 * 100 + A * 10 + 7) / 10 * 10 = 930) : A = 3 :=
sorry

end find_A_l1216_121665


namespace number_of_pumpkin_pies_l1216_121658

-- Definitions for the conditions
def apple_pies : ℕ := 2
def pecan_pies : ℕ := 4
def total_pies : ℕ := 13

-- The proof statement
theorem number_of_pumpkin_pies
  (h_apple : apple_pies = 2)
  (h_pecan : pecan_pies = 4)
  (h_total : total_pies = 13) : 
  total_pies - (apple_pies + pecan_pies) = 7 :=
by 
  sorry

end number_of_pumpkin_pies_l1216_121658


namespace smallest_integer_k_l1216_121699

theorem smallest_integer_k :
  ∃ k : ℕ, 
    k > 1 ∧ 
    k % 19 = 1 ∧ 
    k % 14 = 1 ∧ 
    k % 9 = 1 ∧ 
    k = 2395 :=
by {
  sorry
}

end smallest_integer_k_l1216_121699


namespace find_triangle_lengths_l1216_121614

-- Conditions:
-- 1. Two right-angled triangles are similar.
-- 2. Bigger triangle sides: x + 1 and y + 5, Area larger by 8 cm^2

def triangle_lengths (x y : ℝ) : Prop := 
  (y = 5 * x ∧ 
  (5 / 2) * (x + 1) ^ 2 - (5 / 2) * x ^ 2 = 8)

theorem find_triangle_lengths (x y : ℝ) : triangle_lengths x y ↔ (x = 1.1 ∧ y = 5.5) :=
sorry

end find_triangle_lengths_l1216_121614


namespace dogs_in_school_l1216_121647

theorem dogs_in_school
  (sit: ℕ) (sit_and_stay: ℕ) (stay: ℕ) (stay_and_roll_over: ℕ)
  (roll_over: ℕ) (sit_and_roll_over: ℕ) (all_three: ℕ) (none: ℕ)
  (h1: sit = 50) (h2: sit_and_stay = 17) (h3: stay = 29)
  (h4: stay_and_roll_over = 12) (h5: roll_over = 34)
  (h6: sit_and_roll_over = 18) (h7: all_three = 9) (h8: none = 9) :
  sit + stay + roll_over + sit_and_stay + stay_and_roll_over + sit_and_roll_over - 2 * all_three + none = 84 :=
by sorry

end dogs_in_school_l1216_121647


namespace squared_greater_abs_greater_l1216_121635

theorem squared_greater_abs_greater {a b : ℝ} : a^2 > b^2 ↔ |a| > |b| :=
by sorry

end squared_greater_abs_greater_l1216_121635


namespace sum_of_coordinates_of_D_l1216_121652

def Point := (ℝ × ℝ)

def isMidpoint (M C D : Point) : Prop :=
  M = ((C.1 + D.1) / 2, (C.2 + D.2) / 2)

theorem sum_of_coordinates_of_D (M C : Point) (D : Point) (hM : isMidpoint M C D) (hC : C = (2, 10)) :
  D.1 + D.2 = 12 :=
sorry

end sum_of_coordinates_of_D_l1216_121652


namespace min_area_ABCD_l1216_121679

section Quadrilateral

variables {S1 S2 S3 S4 : ℝ}

-- Define the areas of the triangles
def area_APB := S1
def area_BPC := S2
def area_CPD := S3
def area_DPA := S4

-- Condition: Product of the areas of ΔAPB and ΔCPD is 36
axiom prod_APB_CPD : S1 * S3 = 36

-- We need to prove that the minimum area of the quadrilateral ABCD is 24
theorem min_area_ABCD : S1 + S2 + S3 + S4 ≥ 24 :=
by
  sorry

end Quadrilateral

end min_area_ABCD_l1216_121679


namespace frustum_volume_fraction_l1216_121697

theorem frustum_volume_fraction {V_original V_frustum : ℚ} 
(base_edge : ℚ) (height : ℚ) 
(h1 : base_edge = 24) (h2 : height = 18) 
(h3 : V_original = (1 / 3) * (base_edge ^ 2) * height)
(smaller_base_edge : ℚ) (smaller_height : ℚ) 
(h4 : smaller_height = (1 / 3) * height) (h5 : smaller_base_edge = base_edge / 3) 
(V_smaller : ℚ) (h6 : V_smaller = (1 / 3) * (smaller_base_edge ^ 2) * smaller_height)
(h7 : V_frustum = V_original - V_smaller) :
V_frustum / V_original = 13 / 27 :=
sorry

end frustum_volume_fraction_l1216_121697


namespace smaller_angle_36_degrees_l1216_121660

noncomputable def smaller_angle_measure (larger smaller : ℝ) : Prop :=
(larger + smaller = 180) ∧ (larger = 4 * smaller)

theorem smaller_angle_36_degrees : ∃ (smaller : ℝ), smaller_angle_measure (4 * smaller) smaller ∧ smaller = 36 :=
by
  sorry

end smaller_angle_36_degrees_l1216_121660


namespace total_jumps_l1216_121678

-- Definitions based on given conditions
def Ronald_jumps : ℕ := 157
def Rupert_jumps : ℕ := Ronald_jumps + 86

-- The theorem we want to prove
theorem total_jumps : Ronald_jumps + Rupert_jumps = 400 :=
by
  sorry

end total_jumps_l1216_121678


namespace cubic_polynomials_integer_roots_l1216_121654

theorem cubic_polynomials_integer_roots (a b : ℤ) :
  (∀ α1 α2 α3 : ℤ, α1 + α2 + α3 = 0 ∧ α1 * α2 + α2 * α3 + α3 * α1 = a ∧ α1 * α2 * α3 = -b) →
  (∀ β1 β2 β3 : ℤ, β1 + β2 + β3 = 0 ∧ β1 * β2 + β2 * β3 + β3 * β1 = b ∧ β1 * β2 * β3 = -a) →
  a = 0 ∧ b = 0 :=
by
  sorry

end cubic_polynomials_integer_roots_l1216_121654


namespace find_sum_u_v_l1216_121603

theorem find_sum_u_v : ∃ (u v : ℚ), 5 * u - 6 * v = 35 ∧ 3 * u + 5 * v = -10 ∧ u + v = -40 / 43 :=
by
  sorry

end find_sum_u_v_l1216_121603


namespace min_expression_l1216_121626

theorem min_expression : ∀ x y : ℝ, ∃ x, 4 * x^2 + 4 * x * (Real.sin y) - (Real.cos y)^2 = -1 := by
  sorry

end min_expression_l1216_121626


namespace quadrilateral_side_length_l1216_121664

theorem quadrilateral_side_length (r a b c x : ℝ) (h_radius : r = 100 * Real.sqrt 6) 
    (h_a : a = 100) (h_b : b = 200) (h_c : c = 200) :
    x = 100 * Real.sqrt 2 := 
sorry

end quadrilateral_side_length_l1216_121664


namespace largest_angle_of_convex_pentagon_l1216_121666

theorem largest_angle_of_convex_pentagon :
  ∀ (x : ℝ), (x + 2) + (2 * x + 3) + (3 * x + 4) + (4 * x + 5) + (5 * x + 6) = 540 →
  5 * (104 / 3 : ℝ) + 6 = 538 / 3 := 
by
  intro x
  intro h
  sorry

end largest_angle_of_convex_pentagon_l1216_121666


namespace Cole_drive_time_to_work_l1216_121601

theorem Cole_drive_time_to_work :
  ∀ (D T_work T_home : ℝ),
    (T_work = D / 80) →
    (T_home = D / 120) →
    (T_work + T_home = 3) →
    (T_work * 60 = 108) :=
by
  intros D T_work T_home h1 h2 h3
  sorry

end Cole_drive_time_to_work_l1216_121601


namespace total_chickens_l1216_121616

open Nat

theorem total_chickens 
  (Q S C : ℕ) 
  (h1 : Q = 2 * S + 25) 
  (h2 : S = 3 * C - 4) 
  (h3 : C = 37) : 
  Q + S + C = 383 := by
  sorry

end total_chickens_l1216_121616


namespace last_two_digits_of_100_factorial_l1216_121634

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_nonzero_digits (n : ℕ) : ℕ := sorry

theorem last_two_digits_of_100_factorial :
  last_two_nonzero_digits (factorial 100) = 24 :=
sorry

end last_two_digits_of_100_factorial_l1216_121634


namespace labels_closer_than_distance_l1216_121693

noncomputable def exists_points_with_labels_closer_than_distance (f : ℝ × ℝ → ℝ) : Prop :=
  ∃ (P Q : ℝ × ℝ), P ≠ Q ∧ |f P - f Q| < dist P Q

-- Statement of the problem
theorem labels_closer_than_distance :
  ∀ (f : ℝ × ℝ → ℝ), exists_points_with_labels_closer_than_distance f :=
sorry

end labels_closer_than_distance_l1216_121693


namespace alice_always_wins_l1216_121656

theorem alice_always_wins (n : ℕ) (initial_coins : ℕ) (alice_first_move : ℕ) (total_coins : ℕ) :
  initial_coins = 1331 → alice_first_move = 1 → total_coins = 1331 →
  (∀ (k : ℕ), 
    let alice_total := (k * (k + 1)) / 2;
    let basilio_min_total := (k * (k - 1)) / 2;
    let basilio_max_total := (k * (k + 1)) / 2 - 1;
    k * k ≤ total_coins ∧ total_coins ≤ k * (k + 1) - 1 →
    ¬ (total_coins = k * k + k - 1 ∨ total_coins = k * (k + 1) - 1)) →
  alice_first_move = 1 ∧ initial_coins = 1331 ∧ total_coins = 1331 → alice_wins :=
sorry

end alice_always_wins_l1216_121656


namespace inequality_proof_l1216_121683

variable (a b c d : ℝ)
variable (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ)

-- Define conditions
def positive (x : ℝ) := x > 0
def unit_circle (x y : ℝ) := x^2 + y^2 = 1

-- Define the main theorem
theorem inequality_proof
  (ha : positive a)
  (hb : positive b)
  (hc : positive c)
  (hd : positive d)
  (habcd : a * b + c * d = 1)
  (hP1 : unit_circle x1 y1)
  (hP2 : unit_circle x2 y2)
  (hP3 : unit_circle x3 y3)
  (hP4 : unit_circle x4 y4)
  : 
  (a * y1 + b * y2 + c * y3 + d * y4)^2 + (a * x4 + b * x3 + c * x2 + d * x1)^2
  ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := sorry

end inequality_proof_l1216_121683


namespace eggs_in_each_basket_l1216_121674

theorem eggs_in_each_basket (n : ℕ) (h₁ : 5 ≤ n) (h₂ : n ∣ 30) (h₃ : n ∣ 42) : n = 6 :=
sorry

end eggs_in_each_basket_l1216_121674


namespace circle_radius_l1216_121610

theorem circle_radius (C : ℝ) (r : ℝ) (h1 : C = 72 * Real.pi) (h2 : C = 2 * Real.pi * r) : r = 36 :=
by
  sorry

end circle_radius_l1216_121610


namespace total_profit_is_correct_l1216_121650

-- Definitions for the investments and profit shares
def x_investment : ℕ := 5000
def y_investment : ℕ := 15000
def x_share_of_profit : ℕ := 400

-- The theorem states that the total profit is Rs. 1600 given the conditions
theorem total_profit_is_correct (h1 : x_share_of_profit = 400) (h2 : x_investment = 5000) (h3 : y_investment = 15000) : 
  let y_share_of_profit := 3 * x_share_of_profit
  let total_profit := x_share_of_profit + y_share_of_profit
  total_profit = 1600 :=
by
  sorry

end total_profit_is_correct_l1216_121650


namespace no_ghost_not_multiple_of_p_l1216_121644

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sequence_S (p : ℕ) (S : ℕ → ℕ) : Prop :=
  (is_prime p ∧ p % 2 = 1) ∧
  (∀ i, 1 ≤ i ∧ i < p → S i = i) ∧
  (∀ n, n ≥ p → (S n > S (n-1) ∧ 
    ∀ (a b c : ℕ), (a < b ∧ b < c ∧ c < n ∧ S a < S b ∧ S b < S c ∧
    S b - S a = S c - S b → false)))

def is_ghost (p : ℕ) (S : ℕ → ℕ) (g : ℕ) : Prop :=
  ∀ n : ℕ, S n ≠ g

theorem no_ghost_not_multiple_of_p (p : ℕ) (S : ℕ → ℕ) :
  (is_prime p ∧ p % 2 = 1) ∧ sequence_S p S → 
  ∀ g : ℕ, is_ghost p S g → p ∣ g :=
by 
  sorry

end no_ghost_not_multiple_of_p_l1216_121644


namespace train_speed_l1216_121688

noncomputable def train_length : ℝ := 150
noncomputable def bridge_length : ℝ := 250
noncomputable def crossing_time : ℝ := 28.79769618430526

noncomputable def speed_m_per_s : ℝ := (train_length + bridge_length) / crossing_time
noncomputable def speed_kmph : ℝ := speed_m_per_s * 3.6

theorem train_speed : speed_kmph = 50 := by
  sorry

end train_speed_l1216_121688


namespace length_of_shorter_side_l1216_121677

/-- 
A rectangular plot measuring L meters by 50 meters is to be enclosed by wire fencing. 
If the poles of the fence are kept 5 meters apart, 26 poles will be needed.
What is the length of the shorter side of the rectangular plot?
-/
theorem length_of_shorter_side
(L: ℝ) 
(h1: ∃ L: ℝ, L > 0) -- There's some positive length for the side L
(h2: ∀ distance: ℝ, distance = 5) -- Poles are kept 5 meters apart
(h3: ∀ poles: ℝ, poles = 26) -- 26 poles will be needed
(h4: 125 = 2 * (L + 50)) -- Use the perimeter calculated
: L = 12.5
:= sorry

end length_of_shorter_side_l1216_121677


namespace initial_distance_between_jack_and_christina_l1216_121622

theorem initial_distance_between_jack_and_christina
  (jack_speed : ℝ)
  (christina_speed : ℝ)
  (lindy_speed : ℝ)
  (lindy_total_distance : ℝ)
  (meeting_time : ℝ)
  (combined_speed : ℝ) :
  jack_speed = 5 ∧
  christina_speed = 3 ∧
  lindy_speed = 9 ∧
  lindy_total_distance = 270 ∧
  meeting_time = lindy_total_distance / lindy_speed ∧
  combined_speed = jack_speed + christina_speed →
  meeting_time = 30 ∧
  combined_speed = 8 →
  (combined_speed * meeting_time) = 240 :=
by
  sorry

end initial_distance_between_jack_and_christina_l1216_121622


namespace program_selection_count_l1216_121648

theorem program_selection_count :
  let courses := ["English", "Algebra", "Geometry", "History", "Science", "Art", "Latin"]
  let english := 1
  let math_courses := ["Algebra", "Geometry"]
  let science_courses := ["Science"]
  ∃ (programs : Finset (Finset String)) (count : ℕ),
    (count = 9) ∧
    (programs.card = count) ∧
    ∀ p ∈ programs,
      "English" ∈ p ∧
      (∃ m ∈ p, m ∈ math_courses) ∧
      (∃ s ∈ p, s ∈ science_courses) ∧
      p.card = 5 :=
sorry

end program_selection_count_l1216_121648


namespace cow_difference_l1216_121636

variables (A M R : Nat)

def Aaron_has_four_times_as_many_cows_as_Matthews : Prop := A = 4 * M
def Matthews_has_cows : Prop := M = 60
def Total_cows_for_three := A + M + R = 570

theorem cow_difference (h1 : Aaron_has_four_times_as_many_cows_as_Matthews A M) 
                       (h2 : Matthews_has_cows M)
                       (h3 : Total_cows_for_three A M R) :
  (A + M) - R = 30 :=
by
  sorry

end cow_difference_l1216_121636


namespace slip_4_goes_in_B_l1216_121698

-- Definitions for the slips, cups, and conditions
def slips : List ℝ := [1, 1.5, 2, 2, 2.5, 2.5, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5]
def cupSum (c : Char) : ℝ := 
  match c with
  | 'A' => 6
  | 'B' => 7
  | 'C' => 8
  | 'D' => 9
  | 'E' => 10
  | 'F' => 11
  | _   => 0

def cupAssignments : Char → List ℝ
  | 'F' => [2]
  | 'B' => [3]
  | _   => []

theorem slip_4_goes_in_B :
  (∃ cupA cupB cupC cupD cupE cupF : List ℝ, 
    cupA.sum = cupSum 'A' ∧
    cupB.sum = cupSum 'B' ∧
    cupC.sum = cupSum 'C' ∧
    cupD.sum = cupSum 'D' ∧
    cupE.sum = cupSum 'E' ∧
    cupF.sum = cupSum 'F' ∧
    slips = cupA ++ cupB ++ cupC ++ cupD ++ cupE ++ cupF ∧
    cupF.contains 2 ∧
    cupB.contains 3 ∧
    cupB.contains 4) :=
sorry

end slip_4_goes_in_B_l1216_121698


namespace book_selection_l1216_121645

theorem book_selection :
  let tier1 := 3
  let tier2 := 5
  let tier3 := 8
  tier1 + tier2 + tier3 = 16 :=
by
  let tier1 := 3
  let tier2 := 5
  let tier3 := 8
  sorry

end book_selection_l1216_121645


namespace janna_wrote_more_words_than_yvonne_l1216_121643

theorem janna_wrote_more_words_than_yvonne :
  ∃ (janna_words_written yvonne_words_written : ℕ), 
    yvonne_words_written = 400 ∧
    janna_words_written > yvonne_words_written ∧
    ∃ (removed_words added_words : ℕ),
      removed_words = 20 ∧
      added_words = 2 * removed_words ∧
      (janna_words_written + yvonne_words_written - removed_words + added_words + 30 = 1000) ∧
      (janna_words_written - yvonne_words_written = 130) :=
by
  sorry

end janna_wrote_more_words_than_yvonne_l1216_121643


namespace relationship_among_sets_l1216_121615

-- Definitions based on the conditions
def RegularQuadrilateralPrism (x : Type) : Prop := -- prisms with a square base and perpendicular lateral edges
  sorry

def RectangularPrism (x : Type) : Prop := -- prisms with a rectangular base and perpendicular lateral edges
  sorry

def RightQuadrilateralPrism (x : Type) : Prop := -- prisms whose lateral edges are perpendicular to the base, and the base can be any quadrilateral
  sorry

def RightParallelepiped (x : Type) : Prop := -- prisms with lateral edges perpendicular to the base
  sorry

-- Sets
def M : Set Type := { x | RegularQuadrilateralPrism x }
def P : Set Type := { x | RectangularPrism x }
def N : Set Type := { x | RightQuadrilateralPrism x }
def Q : Set Type := { x | RightParallelepiped x }

-- Proof problem statement
theorem relationship_among_sets : M ⊂ P ∧ P ⊂ Q ∧ Q ⊂ N := 
  by
    sorry

end relationship_among_sets_l1216_121615


namespace inversions_range_l1216_121605

/-- Given any permutation of 10 elements, 
    the number of inversions (or disorders) in the permutation 
    can take any value from 0 to 45.
-/
theorem inversions_range (perm : List ℕ) (h_length : perm.length = 10):
  ∃ S, 0 ≤ S ∧ S ≤ 45 :=
sorry

end inversions_range_l1216_121605


namespace find_borrowed_amount_l1216_121641

noncomputable def borrowed_amount (P : ℝ) : Prop :=
  let interest_paid := P * (4 / 100) * 2
  let interest_earned := P * (6 / 100) * 2
  let total_gain := 120 * 2
  interest_earned - interest_paid = total_gain

theorem find_borrowed_amount : ∃ P : ℝ, borrowed_amount P ∧ P = 3000 :=
by
  use 3000
  unfold borrowed_amount
  simp
  sorry

end find_borrowed_amount_l1216_121641


namespace evaluate_at_neg_one_l1216_121620

def f (x : ℝ) : ℝ := -2 * x ^ 2 + 1

theorem evaluate_at_neg_one : f (-1) = -1 := 
by
  -- Proof goes here
  sorry

end evaluate_at_neg_one_l1216_121620


namespace domain_tan_3x_sub_pi_over_4_l1216_121619

noncomputable def domain_of_f : Set ℝ :=
  {x : ℝ | ∀ k : ℤ, x ≠ (k * Real.pi) / 3 + Real.pi / 4}

theorem domain_tan_3x_sub_pi_over_4 :
  ∀ x : ℝ, x ∈ domain_of_f ↔ ∀ k : ℤ, x ≠ (k * Real.pi) / 3 + Real.pi / 4 :=
by
  intro x
  sorry

end domain_tan_3x_sub_pi_over_4_l1216_121619


namespace unique_set_of_consecutive_integers_l1216_121663

theorem unique_set_of_consecutive_integers (a b c : ℕ) : 
  (a + b + c = 36) ∧ (b = a + 1) ∧ (c = a + 2) → 
  ∃! a : ℕ, (a = 11 ∧ b = 12 ∧ c = 13) := 
sorry

end unique_set_of_consecutive_integers_l1216_121663


namespace taller_tree_height_l1216_121609

-- Given conditions
variables (h : ℕ) (ratio_cond : (h - 20) * 7 = h * 5)

-- Proof goal
theorem taller_tree_height : h = 70 :=
sorry

end taller_tree_height_l1216_121609


namespace mean_of_five_numbers_l1216_121696

theorem mean_of_five_numbers (sum_of_numbers : ℚ) (number_of_elements : ℕ)
  (h_sum : sum_of_numbers = 3 / 4) (h_elements : number_of_elements = 5) :
  (sum_of_numbers / number_of_elements : ℚ) = 3 / 20 :=
by
  sorry

end mean_of_five_numbers_l1216_121696


namespace ball_color_problem_l1216_121695

theorem ball_color_problem
  (n : ℕ)
  (h₀ : ∀ i : ℕ, i ≤ 49 → ∃ r : ℕ, r = 49 ∧ i = 50) 
  (h₁ : ∀ i : ℕ, i > 49 → ∃ r : ℕ, r = 49 + 7 * (i - 50) / 8 ∧ i = n)
  (h₂ : 90 ≤ (49 + (7 * (n - 50) / 8)) * 10 / n) :
  n ≤ 210 := 
sorry

end ball_color_problem_l1216_121695


namespace persimmons_count_l1216_121640

theorem persimmons_count (x : ℕ) (h : x - 5 = 12) : x = 17 :=
by
  sorry

end persimmons_count_l1216_121640


namespace weight_of_B_l1216_121681

theorem weight_of_B (A B C : ℝ)
  (h1 : (A + B + C) / 3 = 45)
  (h2 : (A + B) / 2 = 40)
  (h3 : (B + C) / 2 = 43) :
  B = 31 :=
by sorry

end weight_of_B_l1216_121681


namespace probability_same_flips_l1216_121639

-- Define the probability of getting the first head on the nth flip
def prob_first_head_on_nth_flip (n : ℕ) : ℚ :=
  (1 / 2) ^ n

-- Define the probability that all three get the first head on the nth flip
def prob_all_three_first_head_on_nth_flip (n : ℕ) : ℚ :=
  (prob_first_head_on_nth_flip n) ^ 3

-- Define the total probability considering all n
noncomputable def total_prob_all_three_same_flips : ℚ :=
  ∑' n, prob_all_three_first_head_on_nth_flip (n + 1)

-- The statement to prove
theorem probability_same_flips : total_prob_all_three_same_flips = 1 / 7 :=
by sorry

end probability_same_flips_l1216_121639


namespace find_x_minus_y_l1216_121638

open Real

theorem find_x_minus_y (x y : ℝ) (h : (sin x ^ 2 - cos x ^ 2 + cos x ^ 2 * cos y ^ 2 - sin x ^ 2 * sin y ^ 2) / sin (x + y) = 1) :
  ∃ k : ℤ, x - y = π / 2 + 2 * k * π :=
by
  sorry

end find_x_minus_y_l1216_121638


namespace total_courses_l1216_121617

-- Define the conditions as variables
def max_courses : Nat := 40
def sid_courses : Nat := 4 * max_courses

-- State the theorem we want to prove
theorem total_courses : max_courses + sid_courses = 200 := 
  by
    -- This is where the actual proof would go
    sorry

end total_courses_l1216_121617


namespace infinite_primes_of_form_l1216_121607

theorem infinite_primes_of_form (p : ℕ) (hp : Nat.Prime p) (hpodd : p % 2 = 1) :
  ∃ᶠ n in at_top, Nat.Prime (2 * p * n + 1) :=
sorry

end infinite_primes_of_form_l1216_121607


namespace hyperbola_eccentricity_l1216_121651

theorem hyperbola_eccentricity (a : ℝ) (h : a > 0) (h_asymptote : Real.tan (Real.pi / 6) = 1 / a) :
  let c := Real.sqrt (a^2 + 1)
  let e := c / a
  e = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end hyperbola_eccentricity_l1216_121651


namespace find_range_a_l1216_121655

-- Define the parabola equation y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line equation y = (√3/3) * (x - a)
def line (x y a : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - a)

-- Define the focus of the parabola
def focus (x y : ℝ) : Prop := x = 1 ∧ y = 0

-- Define the condition that F is outside the circle with diameter CD
def F_outside_circle_CD (x1 y1 x2 y2 a : ℝ) : Prop :=
  (x1 - 1) * (x2 - 1) + y1 * y2 > 0

-- Define the parabola-line intersection points and the related Vieta's formulas
def intersection_points (a : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 + x2 = 2 * a + 12 ∧ x1 * x2 = a^2

-- Define the final condition for a
def range_a (a : ℝ) : Prop :=
  -3 < a ∧ a < -2 * Real.sqrt 5 + 3

-- Main theorem statement
theorem find_range_a (a : ℝ) (hneg : a < 0)
  (x1 x2 y1 y2 : ℝ)
  (hparabola1 : parabola x1 y1)
  (hparabola2 : parabola x2 y2)
  (hline1 : line x1 y1 a)
  (hline2 : line x2 y2 a)
  (hfocus : focus 1 0)
  (hF_out : F_outside_circle_CD x1 y1 x2 y2 a)
  (hintersect : intersection_points a x1 x2) :
  range_a a := 
sorry

end find_range_a_l1216_121655


namespace smallest_integer_ending_in_9_and_divisible_by_11_l1216_121623

theorem smallest_integer_ending_in_9_and_divisible_by_11 : ∃ n : ℕ, n > 0 ∧ n % 10 = 9 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 9 → m % 11 = 0 → m ≥ n :=
  sorry

end smallest_integer_ending_in_9_and_divisible_by_11_l1216_121623


namespace difference_in_money_in_nickels_l1216_121692

-- Define the given conditions
def alice_quarters (p : ℕ) : ℕ := 3 * p + 2
def bob_quarters (p : ℕ) : ℕ := 2 * p + 8

-- Define the difference in their money in nickels
def difference_in_nickels (p : ℕ) : ℕ := 5 * (p - 6)

-- The proof problem statement
theorem difference_in_money_in_nickels (p : ℕ) : 
  (5 * (alice_quarters p - bob_quarters p)) = difference_in_nickels p :=
by 
  sorry

end difference_in_money_in_nickels_l1216_121692


namespace rangeOfA_l1216_121649

theorem rangeOfA (a : ℝ) : 
  (∃ x : ℝ, 9^x + a * 3^x + 4 = 0) → a ≤ -4 :=
by
  sorry

end rangeOfA_l1216_121649


namespace matrix_identity_l1216_121602

noncomputable def N : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 4], ![-2, 1]]
noncomputable def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem matrix_identity :
  N * N = 4 • N + -11 • I :=
by
  sorry

end matrix_identity_l1216_121602


namespace find_a61_l1216_121646

def seq (a : ℕ → ℕ) : Prop :=
  (∀ n, a (2 * n + 1) = a n + a (n + 1)) ∧
  (∀ n, a (2 * n) = a n) ∧
  a 1 = 1

theorem find_a61 (a : ℕ → ℕ) (h : seq a) : a 61 = 9 :=
by
  sorry

end find_a61_l1216_121646


namespace percentage_increase_l1216_121694

theorem percentage_increase (M N : ℝ) (h : M ≠ N) : 
  (200 * (M - N) / (M + N) = ((200 : ℝ) * (M - N) / (M + N))) :=
by
  -- Translate the problem conditions into Lean definitions
  let average := (M + N) / 2
  let increase := (M - N)
  let fraction_of_increase_over_average := (increase / average) * 100

  -- Additional annotations and calculations to construct the proof would go here
  sorry

end percentage_increase_l1216_121694


namespace unique_zero_property_l1216_121624

theorem unique_zero_property (x : ℝ) (h1 : ∀ a : ℝ, x * a = x) (h2 : ∀ (a : ℝ), a ≠ 0 → x / a = x) :
  x = 0 :=
sorry

end unique_zero_property_l1216_121624


namespace clock_palindromes_l1216_121667

theorem clock_palindromes : 
  let valid_hours := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 22]
  let valid_minutes := [0, 1, 2, 3, 4, 5]
  let two_digit_palindromes := 9 * 6
  let four_digit_palindromes := 6
  (two_digit_palindromes + four_digit_palindromes) = 60 := 
by
  sorry

end clock_palindromes_l1216_121667


namespace same_color_probability_is_correct_l1216_121625

-- Define the variables and conditions
def total_sides : ℕ := 12
def pink_sides : ℕ := 3
def green_sides : ℕ := 4
def blue_sides : ℕ := 5

-- Calculate individual probabilities
def pink_probability : ℚ := (pink_sides : ℚ) / total_sides
def green_probability : ℚ := (green_sides : ℚ) / total_sides
def blue_probability : ℚ := (blue_sides : ℚ) / total_sides

-- Calculate the probabilities that both dice show the same color
def both_pink_probability : ℚ := pink_probability ^ 2
def both_green_probability : ℚ := green_probability ^ 2
def both_blue_probability : ℚ := blue_probability ^ 2

-- The final probability that both dice come up the same color
def same_color_probability : ℚ := both_pink_probability + both_green_probability + both_blue_probability

theorem same_color_probability_is_correct : same_color_probability = 25 / 72 := by
  sorry

end same_color_probability_is_correct_l1216_121625


namespace cistern_wet_surface_area_l1216_121686

theorem cistern_wet_surface_area
  (length : ℝ) (width : ℝ) (breadth : ℝ)
  (h_length : length = 9)
  (h_width : width = 6)
  (h_breadth : breadth = 2.25) :
  (length * width + 2 * (length * breadth) + 2 * (width * breadth)) = 121.5 :=
by
  -- Proof goes here
  sorry

end cistern_wet_surface_area_l1216_121686


namespace path_counts_l1216_121670

    noncomputable def x : ℝ := 2 + Real.sqrt 2
    noncomputable def y : ℝ := 2 - Real.sqrt 2

    theorem path_counts (n : ℕ) :
      ∃ α : ℕ → ℕ, (α (2 * n - 1) = 0) ∧ (α (2 * n) = (1 / Real.sqrt 2) * ((x ^ (n - 1)) - (y ^ (n - 1)))) :=
    by
      sorry
    
end path_counts_l1216_121670


namespace correct_ratio_l1216_121630

theorem correct_ratio (a b : ℝ) (h : 4 * a = 5 * b) : a / b = 5 / 4 :=
by
  sorry

end correct_ratio_l1216_121630


namespace calories_per_pound_of_body_fat_l1216_121621

theorem calories_per_pound_of_body_fat (gained_weight : ℕ) (calories_burned_per_day : ℕ) 
  (days_to_lose_weight : ℕ) (calories_consumed_per_day : ℕ) : 
  gained_weight = 5 → 
  calories_burned_per_day = 2500 → 
  days_to_lose_weight = 35 → 
  calories_consumed_per_day = 2000 → 
  (calories_burned_per_day * days_to_lose_weight - calories_consumed_per_day * days_to_lose_weight) / gained_weight = 3500 :=
by 
  intros h1 h2 h3 h4
  sorry

end calories_per_pound_of_body_fat_l1216_121621


namespace original_price_l1216_121629

theorem original_price (P : ℝ) (h : P * 0.80 = 960) : P = 1200 :=
sorry

end original_price_l1216_121629


namespace gcf_50_75_l1216_121600

theorem gcf_50_75 : Nat.gcd 50 75 = 25 := by
  sorry

end gcf_50_75_l1216_121600


namespace perfect_squares_with_property_l1216_121684

open Nat

def is_prime_power (n : ℕ) : Prop :=
  ∃ p k : ℕ, p.Prime ∧ k > 0 ∧ n = p^k

def satisfies_property (n : ℕ) : Prop :=
  ∀ a : ℕ, a ∣ n → a ≥ 15 → is_prime_power (a + 15)

theorem perfect_squares_with_property :
  {n | satisfies_property n ∧ ∃ k : ℕ, n = k^2} = {1, 4, 9, 16, 49, 64, 196} :=
by
  sorry

end perfect_squares_with_property_l1216_121684


namespace simplify_expression_l1216_121611

theorem simplify_expression (x : ℝ) : 3 * x + 4 * (2 - x) - 2 * (3 - 2 * x) + 5 * (2 + 3 * x) = 18 * x + 12 :=
by
  sorry

end simplify_expression_l1216_121611


namespace repair_cost_l1216_121637

variable (R : ℝ)

theorem repair_cost (purchase_price transportation_charges profit_rate selling_price : ℝ) (h1 : purchase_price = 12000) (h2 : transportation_charges = 1000) (h3 : profit_rate = 0.5) (h4 : selling_price = 27000) :
  R = 5000 :=
by
  have total_cost := purchase_price + R + transportation_charges
  have selling_price_eq := 1.5 * total_cost
  have sp_eq_27000 := selling_price = 27000
  sorry

end repair_cost_l1216_121637


namespace proposition_D_l1216_121675

/-- Lean statement for proving the correct proposition D -/
theorem proposition_D {a b : ℝ} (h : |a| < b) : a^2 < b^2 :=
sorry

end proposition_D_l1216_121675


namespace mn_value_l1216_121659

-- Definitions
def exponent_m := 2
def exponent_n := 2

-- Theorem statement
theorem mn_value : exponent_m * exponent_n = 4 :=
by
  sorry

end mn_value_l1216_121659


namespace carly_lollipops_total_l1216_121689

theorem carly_lollipops_total (C : ℕ) (h1 : C / 2 = cherry_lollipops)
  (h2 : C / 2 = 3 * 7) : C = 42 :=
by
  sorry

end carly_lollipops_total_l1216_121689


namespace sqrt_57_in_range_l1216_121653

theorem sqrt_57_in_range (h1 : 49 < 57) (h2 : 57 < 64) (h3 : 7^2 = 49) (h4 : 8^2 = 64) : 7 < Real.sqrt 57 ∧ Real.sqrt 57 < 8 := by
  sorry

end sqrt_57_in_range_l1216_121653


namespace bus_people_final_count_l1216_121676

theorem bus_people_final_count (initial_people : ℕ) (people_on : ℤ) (people_off : ℤ) :
  initial_people = 22 → people_on = 4 → people_off = -8 → initial_people + people_on + people_off = 18 :=
by
  intro h_initial h_on h_off
  rw [h_initial, h_on, h_off]
  norm_num

end bus_people_final_count_l1216_121676


namespace AB_length_l1216_121672

noncomputable def length_of_AB (x y : ℝ) (P_ratio Q_ratio : ℝ × ℝ) (PQ_distance : ℝ) : ℝ :=
    x + y

theorem AB_length (x y : ℝ) (P_ratio : ℝ × ℝ := (3, 5)) (Q_ratio : ℝ × ℝ := (4, 5)) (PQ_distance : ℝ := 3) 
    (h1 : 5 * x = 3 * y) -- P divides AB in the ratio 3:5
    (h2 : 5 * (x + 3) = 4 * (y - 3)) -- Q divides AB in the ratio 4:5 and PQ = 3 units
    : length_of_AB x y P_ratio Q_ratio PQ_distance = 43.2 := 
by sorry

end AB_length_l1216_121672


namespace initial_num_nuts_l1216_121632

theorem initial_num_nuts (total_nuts : ℕ) (h1 : 1/6 * total_nuts = 5) : total_nuts = 30 := 
sorry

end initial_num_nuts_l1216_121632


namespace rhombus_diagonal_l1216_121613

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) 
  (h_d1 : d1 = 70) 
  (h_area : area = 5600): 
  (area = (d1 * d2) / 2) → d2 = 160 :=
by
  sorry

end rhombus_diagonal_l1216_121613


namespace mark_sprinted_distance_l1216_121669

def speed := 6 -- miles per hour
def time := 4 -- hours

/-- Mark sprinted exactly 24 miles. -/
theorem mark_sprinted_distance : speed * time = 24 := by
  sorry

end mark_sprinted_distance_l1216_121669


namespace intervals_of_increase_decrease_a_neg1_max_min_values_a_neg2_no_a_for_monotonic_function_l1216_121680

noncomputable def quadratic_function (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * a * x + 3

theorem intervals_of_increase_decrease_a_neg1 : 
  ∀ x : ℝ, quadratic_function (-1) x = x^2 - 2 * x + 3 → 
  (∀ x ≥ 1, quadratic_function (-1) x ≥ quadratic_function (-1) 1) ∧ 
  (∀ x ≤ 1, quadratic_function (-1) x ≤ quadratic_function (-1) 1) :=
  sorry

theorem max_min_values_a_neg2 :
  ∃ min : ℝ, min = -1 ∧ (∀ x : ℝ, quadratic_function (-2) x ≥ min) ∧ 
  (∀ x : ℝ, ∃ y : ℝ, y > x → quadratic_function (-2) y > quadratic_function (-2) x) :=
  sorry

theorem no_a_for_monotonic_function : 
  ∀ a : ℝ, ¬ (∀ x y : ℝ, x ≤ y → quadratic_function a x ≤ quadratic_function a y) ∧ ¬ (∀ x y : ℝ, x ≤ y → quadratic_function a x ≥ quadratic_function a y) :=
  sorry

end intervals_of_increase_decrease_a_neg1_max_min_values_a_neg2_no_a_for_monotonic_function_l1216_121680


namespace geometric_sequence_a5_value_l1216_121673

-- Definition of geometric sequence and the specific condition a_3 * a_7 = 8
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_a5_value
  (a : ℕ → ℝ)
  (geom_seq : is_geometric_sequence a)
  (cond : a 3 * a 7 = 8) :
  a 5 = 2 * Real.sqrt 2 ∨ a 5 = -2 * Real.sqrt 2 :=
sorry

end geometric_sequence_a5_value_l1216_121673


namespace product_of_points_is_correct_l1216_121682

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 4
  else 0

def totalPoints (rolls : List ℕ) : ℕ :=
  rolls.map f |> List.sum

def AlexRolls := [6, 4, 3, 2, 1]
def BobRolls := [5, 6, 2, 3, 3]

def AlexPoints := totalPoints AlexRolls
def BobPoints := totalPoints BobRolls

theorem product_of_points_is_correct : AlexPoints * BobPoints = 672 := by
  sorry

end product_of_points_is_correct_l1216_121682


namespace wax_total_is_correct_l1216_121618

-- Define the given conditions
def current_wax : ℕ := 20
def additional_wax : ℕ := 146

-- The total amount of wax required is the sum of current_wax and additional_wax
def total_wax := current_wax + additional_wax

-- The proof goal is to show that the total_wax equals 166 grams
theorem wax_total_is_correct : total_wax = 166 := by
  sorry

end wax_total_is_correct_l1216_121618


namespace problem_statement_l1216_121631

theorem problem_statement (n : ℤ) (h_odd: Odd n) (h_pos: n > 0) (h_not_divisible_by_3: ¬(3 ∣ n)) : 24 ∣ (n^2 - 1) :=
sorry

end problem_statement_l1216_121631


namespace coloring_count_is_2_l1216_121691

noncomputable def count_colorings (initial_color : String) : Nat := 
  if initial_color = "R" then 2 else 0 -- Assumes only the case of initial red color is valid for simplicity

theorem coloring_count_is_2 (h1 : True) (h2 : True) (h3 : True) (h4 : True):
  count_colorings "R" = 2 := by
  sorry

end coloring_count_is_2_l1216_121691


namespace solution_set_of_inequality_l1216_121612

theorem solution_set_of_inequality (x : ℝ) : 3 * x - 7 ≤ 2 → x ≤ 3 :=
by
  intro h
  sorry

end solution_set_of_inequality_l1216_121612


namespace total_population_l1216_121685

def grown_ups : ℕ := 5256
def children : ℕ := 2987

theorem total_population : grown_ups + children = 8243 :=
by
  sorry

end total_population_l1216_121685


namespace cone_lateral_surface_area_l1216_121662

theorem cone_lateral_surface_area (r V : ℝ) (h l S : ℝ) 
  (radius_condition : r = 6)
  (volume_condition : V = 30 * Real.pi)
  (volume_formula : V = (1 / 3) * Real.pi * r^2 * h)
  (slant_height_formula : l = Real.sqrt (r^2 + h^2))
  (lateral_surface_area_formula : S = Real.pi * r * l) :
  S = 39 * Real.pi := 
sorry

end cone_lateral_surface_area_l1216_121662


namespace total_trees_in_gray_areas_l1216_121608

theorem total_trees_in_gray_areas (white_region_first : ℕ) (white_region_second : ℕ)
    (total_first : ℕ) (total_second : ℕ)
    (h1 : white_region_first = 82) (h2 : white_region_second = 82)
    (h3 : total_first = 100) (h4 : total_second = 90) :
  (total_first - white_region_first) + (total_second - white_region_second) = 26 := by
  sorry

end total_trees_in_gray_areas_l1216_121608


namespace skips_per_meter_l1216_121690

variable (a b c d e f g h : ℕ)

theorem skips_per_meter 
  (hops_skips : a * skips = b * hops)
  (jumps_hops : c * jumps = d * hops)
  (leaps_jumps : e * leaps = f * jumps)
  (leaps_meters : g * leaps = h * meters) :
  1 * skips = (g * b * f * d) / (a * e * h * c) * skips := 
sorry

end skips_per_meter_l1216_121690


namespace find_lunch_days_l1216_121627

variable (x y : ℕ) -- School days for School A and School B
def P_A := x / 2 -- Aliyah packs lunch half the time
def P_B := y / 4 -- Becky packs lunch a quarter of the time
def P_C := y / 2 -- Charlie packs lunch half the time

theorem find_lunch_days (x y : ℕ) :
  P_A x = x / 2 ∧
  P_B y = y / 4 ∧
  P_C y = y / 2 :=
by
  sorry

end find_lunch_days_l1216_121627


namespace last_non_zero_digit_of_40_l1216_121671

def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def last_non_zero_digit (n : ℕ) : ℕ :=
  let p := factorial n
  let digits : List ℕ := List.filter (λ d => d ≠ 0) (p.digits 10)
  digits.headD 0

theorem last_non_zero_digit_of_40 : last_non_zero_digit 40 = 6 := by
  sorry

end last_non_zero_digit_of_40_l1216_121671
