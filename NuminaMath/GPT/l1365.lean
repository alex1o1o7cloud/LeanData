import Mathlib

namespace probability_two_red_two_blue_l1365_136529

theorem probability_two_red_two_blue :
  let total_marbles := 20
  let red_marbles := 12
  let blue_marbles := 8
  let total_ways_to_choose_4 := Nat.choose total_marbles 4
  let ways_to_choose_2_red := Nat.choose red_marbles 2
  let ways_to_choose_2_blue := Nat.choose blue_marbles 2
  (ways_to_choose_2_red * ways_to_choose_2_blue : ℚ) / total_ways_to_choose_4 = 56 / 147 := 
by {
  sorry
}

end probability_two_red_two_blue_l1365_136529


namespace brianna_books_l1365_136502

theorem brianna_books :
  ∀ (books_per_month : ℕ) (given_books : ℕ) (bought_books : ℕ) (borrowed_books : ℕ) (total_books_needed : ℕ),
    (books_per_month = 2) →
    (given_books = 6) →
    (bought_books = 8) →
    (borrowed_books = bought_books - 2) →
    (total_books_needed = 12 * books_per_month) →
    (total_books_needed - (given_books + bought_books + borrowed_books)) = 4 :=
by
  intros
  sorry

end brianna_books_l1365_136502


namespace double_point_quadratic_l1365_136556

theorem double_point_quadratic (m x1 x2 : ℝ) 
  (H1 : x1 < 1) (H2 : 1 < x2)
  (H3 : ∃ (y1 y2 : ℝ), y1 = 2 * x1 ∧ y2 = 2 * x2 ∧ y1 = x1^2 + 2 * m * x1 - m ∧ y2 = x2^2 + 2 * m * x2 - m)
  : m < 1 :=
sorry

end double_point_quadratic_l1365_136556


namespace proportion_of_face_cards_l1365_136571

theorem proportion_of_face_cards (p : ℝ) (h : 1 - (1 - p)^3 = 19 / 27) : p = 1 / 3 :=
sorry

end proportion_of_face_cards_l1365_136571


namespace rope_length_eqn_l1365_136519

theorem rope_length_eqn (x : ℝ) : 8^2 + (x - 3)^2 = x^2 := 
by 
  sorry

end rope_length_eqn_l1365_136519


namespace prove_a1_geq_2k_l1365_136511

variable (n k : ℕ) (a : ℕ → ℕ)
variable (h1: ∀ i, 1 ≤ i → i ≤ n → 1 < a i)
variable (h2: ∀ i j, 1 ≤ i → i < j → j ≤ n → ¬ (a i ∣ a j))
variable (h3: 3^k < 2*n ∧ 2*n < 3^(k + 1))

theorem prove_a1_geq_2k : a 1 ≥ 2^k :=
by
  sorry

end prove_a1_geq_2k_l1365_136511


namespace solve_equation_l1365_136516

theorem solve_equation :
  ∀ x y : ℝ, (x + y)^2 = (x + 1) * (y - 1) → x = -1 ∧ y = 1 :=
by
  intro x y
  sorry

end solve_equation_l1365_136516


namespace find_x_l1365_136546

theorem find_x (x : ℝ) (h : 5.76 = 0.12 * 0.40 * x) : x = 120 := 
sorry

end find_x_l1365_136546


namespace total_juice_drank_l1365_136560

open BigOperators

theorem total_juice_drank (joe_juice sam_fraction alex_fraction : ℚ) :
  joe_juice = 3 / 4 ∧ sam_fraction = 1 / 2 ∧ alex_fraction = 1 / 4 → 
  sam_fraction * joe_juice + alex_fraction * joe_juice = 9 / 16 :=
by
  sorry

end total_juice_drank_l1365_136560


namespace circle_intersection_unique_point_l1365_136518

open Complex

def distance (a b : ℝ × ℝ) : ℝ :=
  (a.1 - b.1)^2 + (a.2 - b.2)^2

theorem circle_intersection_unique_point :
  ∃ k : ℝ, (distance (0, 0) (-5 / 2, 0) - 3 / 2 = k ∨ distance (0, 0) (-5 / 2, 0) + 3 / 2 = k)
  ↔ (k = 2 ∨ k = 5) := sorry

end circle_intersection_unique_point_l1365_136518


namespace sum_of_a_b_l1365_136591

theorem sum_of_a_b (a b : ℝ) (h1 : |a| = 6) (h2 : |b| = 4) (h3 : a * b < 0) :
    a + b = 2 ∨ a + b = -2 :=
sorry

end sum_of_a_b_l1365_136591


namespace determine_m_for_unique_solution_l1365_136582

-- Define the quadratic equation and the condition for a unique solution
def quadratic_eq_has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c = 0

-- Define the specific quadratic equation and its discriminant
def specific_quadratic_eq (m : ℝ) : Prop :=
  quadratic_eq_has_one_solution 3 (-7) m

-- State the main theorem to prove the value of m
theorem determine_m_for_unique_solution :
  specific_quadratic_eq (49 / 12) :=
by
  unfold specific_quadratic_eq quadratic_eq_has_one_solution
  sorry

end determine_m_for_unique_solution_l1365_136582


namespace factorization_correct_l1365_136599

theorem factorization_correct (a : ℝ) : a^2 - 2 * a - 15 = (a + 3) * (a - 5) := 
by
  sorry

end factorization_correct_l1365_136599


namespace total_commencement_addresses_l1365_136561

-- Define the given conditions
def sandoval_addresses := 12
def sandoval_rainy_addresses := 5
def sandoval_public_holidays := 2
def sandoval_non_rainy_addresses := sandoval_addresses - sandoval_rainy_addresses

def hawkins_addresses := sandoval_addresses / 2
def sloan_addresses := sandoval_addresses + 10
def sloan_non_rainy_addresses := sloan_addresses -- assuming no rainy day details are provided

def davenport_addresses := (sandoval_non_rainy_addresses + sloan_non_rainy_addresses) / 2 - 3
def davenport_addresses_rounded := 11 -- rounding down to nearest integer as per given solution

def adkins_addresses := hawkins_addresses + davenport_addresses_rounded + 2

-- Calculate the total number of addresses
def total_addresses := sandoval_addresses + hawkins_addresses + sloan_addresses + davenport_addresses_rounded + adkins_addresses

-- The proof goal statement
theorem total_commencement_addresses : total_addresses = 70 := by
  -- Proof to be provided here
  sorry

end total_commencement_addresses_l1365_136561


namespace container_unoccupied_volume_is_628_l1365_136567

def rectangular_prism_volume (length width height : ℕ) : ℕ :=
  length * width * height

def water_volume (total_volume : ℕ) : ℕ :=
  total_volume / 3

def ice_cubes_volume (number_of_cubes volume_per_cube : ℕ) : ℕ :=
  number_of_cubes * volume_per_cube

def unoccupied_volume (total_volume occupied_volume : ℕ) : ℕ :=
  total_volume - occupied_volume

theorem container_unoccupied_volume_is_628 :
  let length := 12
  let width := 10
  let height := 8
  let number_of_ice_cubes := 12
  let volume_per_ice_cube := 1
  let V := rectangular_prism_volume length width height
  let V_water := water_volume V
  let V_ice := ice_cubes_volume number_of_ice_cubes volume_per_ice_cube
  let V_occupied := V_water + V_ice
  unoccupied_volume V V_occupied = 628 :=
by
  sorry

end container_unoccupied_volume_is_628_l1365_136567


namespace no_x2_term_imp_a_eq_half_l1365_136527

theorem no_x2_term_imp_a_eq_half (a : ℝ) :
  (∀ x : ℝ, (x + 1) * (x^2 - 2 * a * x + a^2) = x^3 + (1 - 2 * a) * x^2 + ((a^2 - 2 * a) * x + a^2)) →
  (∀ c : ℝ, (1 - 2 * a) = 0) →
  a = 1 / 2 :=
by
  intros h_prod h_eq
  have h_eq' : 1 - 2 * a = 0 := h_eq 0
  linarith

end no_x2_term_imp_a_eq_half_l1365_136527


namespace unique_solution_l1365_136506

-- Define the condition that p, q, r are prime numbers
variables {p q r : ℕ}

-- Assume that p, q, and r are primes
axiom p_prime : Prime p
axiom q_prime : Prime q
axiom r_prime : Prime r

-- Define the main theorem to state that the only solution is (7, 3, 2)
theorem unique_solution : p + q^2 = r^4 → (p, q, r) = (7, 3, 2) :=
by
  sorry

end unique_solution_l1365_136506


namespace sequence_a4_eq_15_l1365_136515

theorem sequence_a4_eq_15 (a : ℕ → ℕ) :
  a 1 = 1 ∧ (∀ n, a (n + 1) = 2 * a n + 1) → a 4 = 15 :=
by
  sorry

end sequence_a4_eq_15_l1365_136515


namespace selling_price_to_equal_percentage_profit_and_loss_l1365_136545

-- Definition of the variables and conditions
def cost_price : ℝ := 1500
def sp_profit_25 : ℝ := 1875
def sp_loss : ℝ := 1280

theorem selling_price_to_equal_percentage_profit_and_loss :
  ∃ SP : ℝ, SP = 1720.05 ∧
  (sp_profit_25 = cost_price * 1.25) ∧
  (sp_loss < cost_price) ∧
  (14.67 = ((SP - cost_price) / cost_price) * 100) ∧
  (14.67 = ((cost_price - sp_loss) / cost_price) * 100) :=
by
  sorry

end selling_price_to_equal_percentage_profit_and_loss_l1365_136545


namespace calculate_g_g_2_l1365_136536

def g (x : ℤ) : ℤ := 2 * x^2 + 2 * x - 1

theorem calculate_g_g_2 : g (g 2) = 263 :=
by
  sorry

end calculate_g_g_2_l1365_136536


namespace problem1_problem2_l1365_136526

section ProofProblems

variables {a b : ℝ}

-- Given that a and b are distinct positive numbers
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom a_neq_b : a ≠ b

-- Problem (i): Prove that a^4 + b^4 > a^3 * b + a * b^3
theorem problem1 : a^4 + b^4 > a^3 * b + a * b^3 :=
by {
  sorry
}

-- Problem (ii): Prove that a^5 + b^5 > a^3 * b^2 + a^2 * b^3
theorem problem2 : a^5 + b^5 > a^3 * b^2 + a^2 * b^3 :=
by {
  sorry
}

end ProofProblems

end problem1_problem2_l1365_136526


namespace problem_solution_l1365_136524

theorem problem_solution :
  (-2: ℤ)^2004 + 3 * (-2: ℤ)^2003 = -2^2003 := 
by
  sorry

end problem_solution_l1365_136524


namespace two_rides_combinations_l1365_136535

-- Define the number of friends
def num_friends : ℕ := 7

-- Define the size of the group for one ride
def ride_group_size : ℕ := 4

-- Define the number of combinations of choosing 'ride_group_size' out of 'num_friends'
def combinations_first_ride : ℕ := Nat.choose num_friends ride_group_size

-- Define the number of friends left for the second ride
def remaining_friends : ℕ := num_friends - ride_group_size

-- Define the number of combinations of choosing 'ride_group_size' out of 'remaining_friends' friends
def combinations_second_ride : ℕ := Nat.choose remaining_friends ride_group_size

-- Define the total number of possible combinations for two rides
def total_combinations : ℕ := combinations_first_ride * combinations_second_ride

-- The final theorem stating the total number of combinations is equal to 525
theorem two_rides_combinations : total_combinations = 525 := by
  -- Placeholder for proof
  sorry

end two_rides_combinations_l1365_136535


namespace length_of_midsegment_l1365_136597

/-- Given a quadrilateral ABCD where sides AB and CD are parallel with lengths 7 and 3 
    respectively, and the other sides BC and DA are of lengths 5 and 4 respectively, 
    prove that the length of the segment joining the midpoints of sides BC and DA is 5. -/
theorem length_of_midsegment (A B C D : ℝ × ℝ)
  (HAB : A.1 = 0 ∧ A.2 = 0 ∧ B.1 = 7 ∧ B.2 = 0)
  (HBC : dist B C = 5)
  (HCD : dist C D = 3)
  (HDA : dist D A = 4)
  (Hparallel : B.2 = 0 ∧ D.2 ≠ 0 → C.2 = D.2) :
  dist ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ((A.1 + D.1) / 2, (A.2 + D.2) / 2) = 5 :=
sorry

end length_of_midsegment_l1365_136597


namespace product_positive_l1365_136563

variables {x y : ℝ}

noncomputable def non_zero (z : ℝ) := z ≠ 0

theorem product_positive (hx : non_zero x) (hy : non_zero y) 
(h1 : x^2 - x > y^2) (h2 : y^2 - y > x^2) : x * y > 0 :=
by
  sorry

end product_positive_l1365_136563


namespace opposite_of_3_is_neg3_l1365_136550

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l1365_136550


namespace incorrect_expression_l1365_136531

variable (D : ℚ) (P Q : ℕ) (r s : ℕ)

-- D represents a repeating decimal.
-- P denotes the r figures of D which do not repeat themselves.
-- Q denotes the s figures of D which repeat themselves.

theorem incorrect_expression :
  10^r * (10^s - 1) * D ≠ Q * (P - 1) :=
sorry

end incorrect_expression_l1365_136531


namespace regular_polygon_sides_l1365_136573

theorem regular_polygon_sides (h : ∀ n : ℕ, n > 2 → 160 * n = 180 * (n - 2) → n = 18) : 
∀ n : ℕ, n > 2 → 160 * n = 180 * (n - 2) → n = 18 :=
by
  exact h

end regular_polygon_sides_l1365_136573


namespace ram_task_completion_days_l1365_136549

theorem ram_task_completion_days (R : ℕ) (h1 : ∀ k : ℕ, k = R / 2) (h2 : 1 / R + 2 / R = 1 / 12) : R = 36 :=
sorry

end ram_task_completion_days_l1365_136549


namespace max_playground_area_l1365_136569

theorem max_playground_area
  (l w : ℝ)
  (h_fence : 2 * l + 2 * w = 400)
  (h_l_min : l ≥ 100)
  (h_w_min : w ≥ 50) :
  l * w ≤ 10000 :=
by
  sorry

end max_playground_area_l1365_136569


namespace kelcie_books_multiple_l1365_136575

theorem kelcie_books_multiple (x : ℕ) :
  let megan_books := 32
  let kelcie_books := megan_books / 4
  let greg_books := x * kelcie_books + 9
  let total_books := megan_books + kelcie_books + greg_books
  total_books = 65 → x = 2 :=
by
  intros megan_books kelcie_books greg_books total_books h
  sorry

end kelcie_books_multiple_l1365_136575


namespace train_length_correct_l1365_136578

noncomputable def length_of_first_train (speed1 speed2 : ℝ) (time : ℝ) (length2 : ℝ) : ℝ :=
  let speed1_ms := speed1 * 1000 / 3600
  let speed2_ms := speed2 * 1000 / 3600
  let relative_speed := speed1_ms - speed2_ms
  let total_distance := relative_speed * time
  total_distance - length2

theorem train_length_correct :
  length_of_first_train 72 36 69.99440044796417 300 = 399.9440044796417 :=
by
  sorry

end train_length_correct_l1365_136578


namespace smallest_largest_sum_l1365_136507

theorem smallest_largest_sum (a b c : ℝ) (m M : ℝ) 
  (h1 : a + b + c = 3)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : m = (1/3))
  (h4 : M = 1) :
  (m + M) = 4 / 3 := by
sorry

end smallest_largest_sum_l1365_136507


namespace interchanged_digit_multiple_of_sum_l1365_136548

theorem interchanged_digit_multiple_of_sum (n a b : ℕ) 
  (h1 : n = 10 * a + b) 
  (h2 : n = 3 * (a + b)) 
  (h3 : 1 ≤ a) (h4 : a ≤ 9) 
  (h5 : 0 ≤ b) (h6 : b ≤ 9) : 
  10 * b + a = 8 * (a + b) := 
by 
  sorry

end interchanged_digit_multiple_of_sum_l1365_136548


namespace max_value_relationship_l1365_136566

theorem max_value_relationship (x y : ℝ) :
  (2005 - (x + y)^2 = 2005) → (x = -y) :=
by
  intro h
  sorry

end max_value_relationship_l1365_136566


namespace five_card_draw_probability_l1365_136568

noncomputable def probability_at_least_one_card_from_each_suit : ℚ := 3 / 32

theorem five_card_draw_probability :
  let deck_size := 52
  let suits := 4
  let cards_drawn := 5
  (1 : ℚ) * (3 / 4) * (1 / 2) * (1 / 4) = probability_at_least_one_card_from_each_suit := by
  sorry

end five_card_draw_probability_l1365_136568


namespace joyce_apples_l1365_136510

/-- Joyce starts with some apples. She gives 52 apples to Larry and ends up with 23 apples. 
    Prove that Joyce initially had 75 apples. -/
theorem joyce_apples (initial_apples given_apples final_apples : ℕ) 
  (h1 : given_apples = 52) 
  (h2 : final_apples = 23) 
  (h3 : initial_apples = given_apples + final_apples) : 
  initial_apples = 75 := 
by 
  sorry

end joyce_apples_l1365_136510


namespace lap_distance_l1365_136552

theorem lap_distance (boys_laps : ℕ) (girls_extra_laps : ℕ) (total_girls_miles : ℚ) : 
  boys_laps = 27 → girls_extra_laps = 9 → total_girls_miles = 27 →
  (total_girls_miles / (boys_laps + girls_extra_laps) = 3 / 4) :=
by
  intros hb hg hm
  sorry

end lap_distance_l1365_136552


namespace bella_steps_l1365_136543

/-- Bella begins to walk from her house toward her friend Ella's house. At the same time, Ella starts to skate toward Bella's house. They each maintain a constant speed, and Ella skates three times as fast as Bella walks. The distance between their houses is 10560 feet, and Bella covers 3 feet with each step. Prove that Bella will take 880 steps by the time she meets Ella. -/
theorem bella_steps 
  (d : ℝ)    -- distance between their houses in feet
  (s_bella : ℝ)    -- speed of Bella in feet per minute
  (s_ella : ℝ)    -- speed of Ella in feet per minute
  (steps_per_ft : ℝ)    -- feet per step of Bella
  (h1 : d = 10560)    -- distance between their houses is 10560 feet
  (h2 : s_ella = 3 * s_bella)    -- Ella skates three times as fast as Bella
  (h3 : steps_per_ft = 3)    -- Bella covers 3 feet with each step
  : (10560 / (4 * s_bella)) * s_bella / 3 = 880 :=
by
  -- proof here 
  sorry

end bella_steps_l1365_136543


namespace ratio_of_overtime_to_regular_rate_l1365_136554

def regular_rate : ℝ := 3
def regular_hours : ℕ := 40
def total_pay : ℝ := 186
def overtime_hours : ℕ := 11

theorem ratio_of_overtime_to_regular_rate 
  (r : ℝ) (h : ℕ) (T : ℝ) (h_ot : ℕ) 
  (h_r : r = regular_rate) 
  (h_h : h = regular_hours) 
  (h_T : T = total_pay)
  (h_hot : h_ot = overtime_hours) :
  (T - (h * r)) / h_ot / r = 2 := 
by {
  sorry 
}

end ratio_of_overtime_to_regular_rate_l1365_136554


namespace angle_SQR_measure_l1365_136503

theorem angle_SQR_measure
    (angle_PQR : ℝ)
    (angle_PQS : ℝ)
    (h1 : angle_PQR = 40)
    (h2 : angle_PQS = 15) : 
    angle_PQR - angle_PQS = 25 := 
by
    sorry

end angle_SQR_measure_l1365_136503


namespace parking_garage_floors_l1365_136532

theorem parking_garage_floors 
  (total_time : ℕ)
  (time_per_floor : ℕ)
  (gate_time : ℕ)
  (every_n_floors : ℕ) 
  (F : ℕ) 
  (h1 : total_time = 1440)
  (h2 : time_per_floor = 80)
  (h3 : gate_time = 120)
  (h4 : every_n_floors = 3)
  :
  F = 13 :=
by
  have total_id_time : ℕ := gate_time * ((F - 1) / every_n_floors)
  have total_drive_time : ℕ := time_per_floor * (F - 1)
  have total_time_calc : ℕ := total_drive_time + total_id_time
  have h5 := total_time_calc = total_time
  -- Now we simplify the algebraic equation given the problem conditions
  sorry

end parking_garage_floors_l1365_136532


namespace vasya_read_entire_book_l1365_136541

theorem vasya_read_entire_book :
  let day1 := 1 / 2
  let day2 := 1 / 3 * (1 - day1)
  let days12 := day1 + day2
  let day3 := 1 / 2 * days12
  (days12 + day3) = 1 :=
by
  sorry

end vasya_read_entire_book_l1365_136541


namespace radius_of_circle_is_zero_l1365_136500

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop := 2 * x^2 - 8 * x + 2 * y^2 - 4 * y + 10 = 0

-- Define the goal: To prove that given this equation, the radius of the circle is 0
theorem radius_of_circle_is_zero :
  ∀ x y : ℝ, circle_eq x y → (x - 2)^2 + (y - 1)^2 = 0 :=
sorry

end radius_of_circle_is_zero_l1365_136500


namespace eunji_received_900_won_l1365_136558

-- Define the conditions
def eunji_pocket_money (X : ℝ) : Prop :=
  (X / 2 + 550 = 1000)

-- Define the theorem to prove the question equals the correct answer
theorem eunji_received_900_won {X : ℝ} (h : eunji_pocket_money X) : X = 900 :=
  by
    sorry

end eunji_received_900_won_l1365_136558


namespace distribute_cousins_l1365_136537

-- Define the variables and the conditions
noncomputable def ways_to_distribute_cousins (cousins : ℕ) (rooms : ℕ) : ℕ :=
  if cousins = 5 ∧ rooms = 3 then 66 else sorry

-- State the problem
theorem distribute_cousins: ways_to_distribute_cousins 5 3 = 66 :=
by
  sorry

end distribute_cousins_l1365_136537


namespace mean_of_xyz_l1365_136505

theorem mean_of_xyz (x y z : ℝ) (seven_mean : ℝ)
  (h1 : seven_mean = 45)
  (h2 : (7 * seven_mean + x + y + z) / 10 = 58) :
  (x + y + z) / 3 = 265 / 3 :=
by
  sorry

end mean_of_xyz_l1365_136505


namespace least_four_digit_divisible_1_2_4_8_l1365_136512

theorem least_four_digit_divisible_1_2_4_8 : ∃ n : ℕ, ∀ d1 d2 d3 d4 : ℕ, 
  n = d1*1000 + d2*100 + d3*10 + d4 ∧
  1000 ≤ n ∧ n < 10000 ∧
  d1 ≠ d2 ∧ d2 ≠ d3 ∧ d3 ≠ d4 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d2 ≠ d4 ∧
  n % 1 = 0 ∧
  n % 2 = 0 ∧
  n % 4 = 0 ∧
  n % 8 = 0 ∧
  n = 1248 :=
by
  sorry

end least_four_digit_divisible_1_2_4_8_l1365_136512


namespace paint_time_for_two_people_l1365_136528

/-- 
Proof Problem Statement: Prove that it would take 12 hours for two people to paint the house
given that six people can paint it in 4 hours, assuming everyone works at the same rate.
--/
theorem paint_time_for_two_people 
  (h1 : 6 * 4 = 24) 
  (h2 : ∀ (n : ℕ) (t : ℕ), n * t = 24 → t = 24 / n) : 
  2 * 12 = 24 :=
sorry

end paint_time_for_two_people_l1365_136528


namespace robin_total_cost_l1365_136555

def num_letters_in_name (name : String) : Nat := name.length

def calculate_total_cost (names : List String) (cost_per_bracelet : Nat) : Nat :=
  let total_bracelets := names.foldl (fun acc name => acc + num_letters_in_name name) 0
  total_bracelets * cost_per_bracelet

theorem robin_total_cost : 
  calculate_total_cost ["Jessica", "Tori", "Lily", "Patrice"] 2 = 44 :=
by
  sorry

end robin_total_cost_l1365_136555


namespace decreasing_function_range_l1365_136540

noncomputable def f (a x : ℝ) := a * (x^3) - x + 1

theorem decreasing_function_range (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≤ 0) → a ≤ 0 := by
  sorry

end decreasing_function_range_l1365_136540


namespace initial_strawberry_plants_l1365_136586

theorem initial_strawberry_plants (P : ℕ) (h1 : 24 * P - 4 = 500) : P = 21 := 
by
  sorry

end initial_strawberry_plants_l1365_136586


namespace equation_solution_l1365_136509

theorem equation_solution (x y z : ℕ) :
  x^2 + y^2 = 2^z ↔ ∃ n : ℕ, x = 2^n ∧ y = 2^n ∧ z = 2 * n + 1 := 
sorry

end equation_solution_l1365_136509


namespace min_solution_l1365_136588

theorem min_solution :
  ∀ (x : ℝ), (min (1 / (1 - x)) (2 / (1 - x)) = 2 / (x - 1) - 3) → x = 7 / 3 := 
by
  sorry

end min_solution_l1365_136588


namespace three_digit_numbers_l1365_136539

theorem three_digit_numbers (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 999) (h2 : n^2 % 1000 = n % 1000) : 
  n = 376 ∨ n = 625 :=
by
  sorry

end three_digit_numbers_l1365_136539


namespace paths_A_to_D_l1365_136581

noncomputable def num_paths_from_A_to_D : ℕ := 
  2 * 2 * 2 + 1

theorem paths_A_to_D : num_paths_from_A_to_D = 9 := 
by
  sorry

end paths_A_to_D_l1365_136581


namespace last_two_digits_l1365_136590

theorem last_two_digits (a b : ℕ) (n : ℕ) (h : b ≡ 25 [MOD 100]) (h_pow : (25 : ℕ) ^ n ≡ 25 [MOD 100]) :
  (33 * b ^ n) % 100 = 25 :=
by
  sorry

end last_two_digits_l1365_136590


namespace cone_altitude_to_radius_ratio_l1365_136525

theorem cone_altitude_to_radius_ratio (r h : ℝ) (V_cone V_sphere : ℝ)
  (h1 : V_sphere = (4 / 3) * Real.pi * r^3)
  (h2 : V_cone = (1 / 3) * Real.pi * r^2 * h)
  (h3 : V_cone = (1 / 3) * V_sphere) :
  h / r = 4 / 3 :=
by
  sorry

end cone_altitude_to_radius_ratio_l1365_136525


namespace molecular_weight_of_3_moles_l1365_136501

def molecular_weight_one_mole : ℝ := 176.14
def number_of_moles : ℝ := 3
def total_weight := number_of_moles * molecular_weight_one_mole

theorem molecular_weight_of_3_moles :
  total_weight = 528.42 := sorry

end molecular_weight_of_3_moles_l1365_136501


namespace percentage_increase_ticket_price_l1365_136576

-- Definitions for the conditions
def last_year_income := 100.0
def clubs_share_last_year := 0.10 * last_year_income
def rental_cost := 0.90 * last_year_income
def new_clubs_share := 0.20
def new_income := rental_cost / (1 - new_clubs_share)

-- Lean 4 theorem statement
theorem percentage_increase_ticket_price : 
  new_income = 112.5 → ((new_income - last_year_income) / last_year_income * 100) = 12.5 := 
by
  sorry

end percentage_increase_ticket_price_l1365_136576


namespace abc_value_l1365_136594

theorem abc_value (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c = 30) 
  (h5 : (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c + 504 / (a * b * c) = 1) :
  a * b * c = 1176 := 
sorry

end abc_value_l1365_136594


namespace volume_of_rotated_solid_l1365_136521

theorem volume_of_rotated_solid (unit_cylinder_r1 h1 r2 h2 : ℝ) :
  unit_cylinder_r1 = 6 → h1 = 1 → r2 = 3 → h2 = 4 → 
  (π * unit_cylinder_r1^2 * h1 + π * r2^2 * h2) = 72 * π :=
by 
-- We place the arguments and sorry for skipping the proof
  sorry

end volume_of_rotated_solid_l1365_136521


namespace width_of_room_l1365_136522

theorem width_of_room
  (carpet_has : ℕ)
  (room_length : ℕ)
  (carpet_needs : ℕ)
  (h1 : carpet_has = 18)
  (h2 : room_length = 4)
  (h3 : carpet_needs = 62) :
  (carpet_has + carpet_needs) = room_length * 20 :=
by
  sorry

end width_of_room_l1365_136522


namespace circle_tangent_to_y_axis_l1365_136583

theorem circle_tangent_to_y_axis (m : ℝ) :
  (0 < m) → (∀ p : ℝ × ℝ, (p.1 - m)^2 + p.2^2 = 4 ↔ p.1 ^ 2 = p.2^2) → (m = 2 ∨ m = -2) :=
by
  sorry

end circle_tangent_to_y_axis_l1365_136583


namespace females_in_band_not_orchestra_l1365_136577

/-- The band at Pythagoras High School has 120 female members. -/
def females_in_band : ℕ := 120

/-- The orchestra at Pythagoras High School has 70 female members. -/
def females_in_orchestra : ℕ := 70

/-- There are 45 females who are members of both the band and the orchestra. -/
def females_in_both : ℕ := 45

/-- The combined total number of students involved in either the band or orchestra or both is 250. -/
def total_students : ℕ := 250

/-- The number of females in the band who are NOT in the orchestra. -/
def females_in_band_only : ℕ := females_in_band - females_in_both

theorem females_in_band_not_orchestra : females_in_band_only = 75 := by
  sorry

end females_in_band_not_orchestra_l1365_136577


namespace add_fractions_l1365_136580

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l1365_136580


namespace unique_solutions_l1365_136534

noncomputable def func_solution (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, (x - 2) * f y + f (y + 2 * f x) = f (x + y * f x)

theorem unique_solutions (f : ℝ → ℝ) :
  func_solution f → (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x - 1) :=
by
  intro h
  sorry

end unique_solutions_l1365_136534


namespace sum_of_four_consecutive_integers_with_product_5040_eq_34_l1365_136574

theorem sum_of_four_consecutive_integers_with_product_5040_eq_34 :
  ∃ a b c d : ℕ, a * b * c * d = 5040 ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ (a + b + c + d) = 34 :=
sorry

end sum_of_four_consecutive_integers_with_product_5040_eq_34_l1365_136574


namespace find_star_value_l1365_136544

theorem find_star_value (x : ℤ) :
  45 - (28 - (37 - (15 - x))) = 58 ↔ x = 19 :=
  by
    sorry

end find_star_value_l1365_136544


namespace find_a_l1365_136517

theorem find_a (a : ℂ) (h : a / (1 - I) = (1 + I) / I) : a = -2 * I := 
by
  sorry

end find_a_l1365_136517


namespace currency_exchange_rate_l1365_136596

theorem currency_exchange_rate (b g x : ℕ) (h1 : 1 * b * g = b * g) (h2 : 1 = 1) :
  (b + g) ^ 2 + 1 = b * g * x → x = 5 :=
sorry

end currency_exchange_rate_l1365_136596


namespace tickets_used_to_buy_toys_l1365_136570

-- Definitions for the conditions
def initial_tickets : ℕ := 13
def leftover_tickets : ℕ := 7

-- The theorem we want to prove
theorem tickets_used_to_buy_toys : initial_tickets - leftover_tickets = 6 :=
by
  sorry

end tickets_used_to_buy_toys_l1365_136570


namespace range_of_a_l1365_136523

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) + 3 * x

def has_pos_extremum (a : ℝ) : Prop :=
  ∃ x : ℝ, (3 + a * Real.exp (a * x) = 0) ∧ (x > 0)

theorem range_of_a (a : ℝ) : has_pos_extremum a → a < -3 := by
  sorry

end range_of_a_l1365_136523


namespace platform_length_correct_l1365_136513

noncomputable def train_speed_kmph : ℝ := 72
noncomputable def train_speed_mps : ℝ := train_speed_kmph * (1000 / 3600)
noncomputable def time_cross_platform : ℝ := 30
noncomputable def time_cross_man : ℝ := 19
noncomputable def length_train : ℝ := train_speed_mps * time_cross_man
noncomputable def total_distance_cross_platform : ℝ := train_speed_mps * time_cross_platform
noncomputable def length_platform : ℝ := total_distance_cross_platform - length_train

theorem platform_length_correct : length_platform = 220 := by
  sorry

end platform_length_correct_l1365_136513


namespace kira_travel_time_l1365_136504

def total_travel_time (hours_between_stations : ℕ) (break_minutes : ℕ) : ℕ :=
  let travel_time_hours := 2 * hours_between_stations
  let travel_time_minutes := travel_time_hours * 60
  travel_time_minutes + break_minutes

theorem kira_travel_time : total_travel_time 2 30 = 270 :=
  by sorry

end kira_travel_time_l1365_136504


namespace saturday_earnings_l1365_136593

variable (S : ℝ)
variable (totalEarnings : ℝ := 5182.50)
variable (difference : ℝ := 142.50)

theorem saturday_earnings : 
  S + (S - difference) = totalEarnings → S = 2662.50 := 
by 
  intro h 
  sorry

end saturday_earnings_l1365_136593


namespace roots_eq_s_l1365_136551

theorem roots_eq_s (n c d : ℝ) (h₁ : c * d = 6) (h₂ : c + d = n)
  (h₃ : c^2 + 1 / d = c^2 + d^2 + 1 / c): 
  (n + 217 / 6) = d^2 + 1/ c * (n + c + d)
  :=
by
  -- The proof will go here
  sorry

end roots_eq_s_l1365_136551


namespace problem1_l1365_136598

theorem problem1
  (a b c : ℝ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 0) :
  a + b + c ≥ Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) :=
sorry

end problem1_l1365_136598


namespace price_each_puppy_l1365_136565

def puppies_initial : ℕ := 8
def puppies_given_away : ℕ := puppies_initial / 2
def puppies_remaining_after_giveaway : ℕ := puppies_initial - puppies_given_away
def puppies_kept : ℕ := 1
def puppies_to_sell : ℕ := puppies_remaining_after_giveaway - puppies_kept
def stud_fee : ℕ := 300
def profit : ℕ := 1500
def total_amount_made : ℕ := profit + stud_fee
def price_per_puppy : ℕ := total_amount_made / puppies_to_sell

theorem price_each_puppy :
  price_per_puppy = 600 :=
sorry

end price_each_puppy_l1365_136565


namespace solve_equation1_solve_equation2_solve_equation3_solve_equation4_l1365_136572

theorem solve_equation1 (x : ℝ) : (x - 1) ^ 2 = 4 ↔ x = 3 ∨ x = -1 :=
by sorry

theorem solve_equation2 (x : ℝ) : x ^ 2 + 3 * x - 4 = 0 ↔ x = 1 ∨ x = -4 :=
by sorry

theorem solve_equation3 (x : ℝ) : 4 * x * (2 * x + 1) = 3 * (2 * x + 1) ↔ x = -1 / 2 ∨ x = 3 / 4 :=
by sorry

theorem solve_equation4 (x : ℝ) : 2 * x ^ 2 + 5 * x - 3 = 0 ↔ x = 1 / 2 ∨ x = -3 :=
by sorry

end solve_equation1_solve_equation2_solve_equation3_solve_equation4_l1365_136572


namespace arithmetic_sequence_s9_l1365_136542

noncomputable def arithmetic_sum (a1 d n : ℝ) : ℝ :=
  n * (2*a1 + (n - 1)*d) / 2

noncomputable def general_term (a1 d n : ℝ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_s9 (a1 d : ℝ)
  (h1 : general_term a1 d 3 + general_term a1 d 4 + general_term a1 d 8 = 25) :
  arithmetic_sum a1 d 9 = 75 :=
by sorry

end arithmetic_sequence_s9_l1365_136542


namespace fish_initial_numbers_l1365_136530

theorem fish_initial_numbers (x y : ℕ) (h1 : x + y = 100) (h2 : x - 30 = y - 40) : x = 45 ∧ y = 55 :=
by
  sorry

end fish_initial_numbers_l1365_136530


namespace probability_same_number_l1365_136585

def is_multiple (n factor : ℕ) : Prop :=
  ∃ k : ℕ, n = k * factor

def multiples_below (factor upper_limit : ℕ) : ℕ :=
  (upper_limit - 1) / factor

theorem probability_same_number :
  let upper_limit := 250
  let billy_factor := 20
  let bobbi_factor := 30
  let common_factor := 60
  let billy_multiples := multiples_below billy_factor upper_limit
  let bobbi_multiples := multiples_below bobbi_factor upper_limit
  let common_multiples := multiples_below common_factor upper_limit
  (common_multiples : ℚ) / (billy_multiples * bobbi_multiples) = 1 / 24 :=
by
  sorry

end probability_same_number_l1365_136585


namespace find_y_find_x_l1365_136587

section
variables (a b : ℝ × ℝ) (x y : ℝ)

-- Definition of vectors a and b
def vec_a : ℝ × ℝ := (3, -2)
def vec_b (y : ℝ) : ℝ × ℝ := (-1, y)

-- Definition of perpendicular condition
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0
-- Proof that y = -3/2 if a is perpendicular to b
theorem find_y (h : perpendicular vec_a (vec_b y)) : y = -3 / 2 :=
sorry

-- Definition of vectors a and c
def vec_c (x : ℝ) : ℝ × ℝ := (x, 5)

-- Definition of parallel condition
def parallel (u v : ℝ × ℝ) : Prop := u.1 / v.1 = u.2 / v.2
-- Proof that x = -15/2 if a is parallel to c
theorem find_x (h : parallel vec_a (vec_c x)) : x = -15 / 2 :=
sorry
end

end find_y_find_x_l1365_136587


namespace number_of_friends_l1365_136514

theorem number_of_friends (total_bill : ℝ) (discount_rate : ℝ) (paid_amount : ℝ) (n : ℝ) 
  (h_total_bill : total_bill = 400) 
  (h_discount_rate : discount_rate = 0.05)
  (h_paid_amount : paid_amount = 63.59) 
  (h_total_paid : n * paid_amount = total_bill * (1 - discount_rate)) : n = 6 := 
by
  -- proof goes here
  sorry

end number_of_friends_l1365_136514


namespace exists_nat_with_digit_sum_1000_and_square_sum_1000000_l1365_136520

-- Define a function to calculate the sum of digits in base-10
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem
theorem exists_nat_with_digit_sum_1000_and_square_sum_1000000 :
  ∃ n : ℕ, sum_of_digits n = 1000 ∧ sum_of_digits (n^2) = 1000000 :=
by
  sorry

end exists_nat_with_digit_sum_1000_and_square_sum_1000000_l1365_136520


namespace inequality_solution_set_l1365_136559

theorem inequality_solution_set (x : ℝ) : 9 > -3 * x → x > -3 :=
by
  intro h
  sorry

end inequality_solution_set_l1365_136559


namespace solve_students_and_apples_l1365_136557

noncomputable def students_and_apples : Prop :=
  ∃ (x y : ℕ), y = 4 * x + 3 ∧ 6 * (x - 1) ≤ y ∧ y ≤ 6 * (x - 1) + 2 ∧ x = 4 ∧ y = 19

theorem solve_students_and_apples : students_and_apples :=
  sorry

end solve_students_and_apples_l1365_136557


namespace gold_initial_amount_l1365_136553

theorem gold_initial_amount :
  ∃ x : ℝ, x - (x / 2 * (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6)) = 1 ∧ x = 1.2 :=
by
  existsi 1.2
  sorry

end gold_initial_amount_l1365_136553


namespace total_games_l1365_136564

-- The conditions
def working_games : ℕ := 6
def bad_games : ℕ := 5

-- The theorem to prove
theorem total_games : working_games + bad_games = 11 :=
by
  sorry

end total_games_l1365_136564


namespace arrangement_of_numbers_l1365_136533

theorem arrangement_of_numbers (numbers : Finset ℕ) 
  (h1 : numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) 
  (h_sum : ∀ a b c d e f, a + b + c + d + e + f = 33)
  (h_group_sum : ∀ k1 k2 k3 k4, k1 + k2 + k3 + k4 = 26)
  : ∃ (n : ℕ), n = 2304 := by
  sorry

end arrangement_of_numbers_l1365_136533


namespace circle_area_difference_l1365_136595

theorem circle_area_difference (r1 r2 : ℝ) (π : ℝ) (A1 A2 diff : ℝ) 
  (hr1 : r1 = 30)
  (hd2 : 2 * r2 = 30)
  (hA1 : A1 = π * r1^2)
  (hA2 : A2 = π * r2^2)
  (hdiff : diff = A1 - A2) :
  diff = 675 * π :=
by 
  sorry

end circle_area_difference_l1365_136595


namespace samia_walking_distance_l1365_136508

theorem samia_walking_distance
  (speed_bike : ℝ)
  (speed_walk : ℝ)
  (total_time : ℝ) 
  (fraction_bike : ℝ) 
  (d : ℝ)
  (walking_distance : ℝ) :
  speed_bike = 15 ∧ 
  speed_walk = 4 ∧ 
  total_time = 1 ∧ 
  fraction_bike = 2/3 ∧ 
  walking_distance = (1/3) * d ∧ 
  (53 * d / 180 = total_time) → 
  walking_distance = 1.1 := 
by 
  sorry

end samia_walking_distance_l1365_136508


namespace infinite_68_in_cells_no_repeats_in_cells_l1365_136579

-- Define the spiral placement function
def spiral (n : ℕ) : ℕ := sorry  -- This function should describe the placement of numbers in the spiral

-- Define a function to get the sum of the numbers in the nodes of a cell.
def cell_sum (cell : ℕ) : ℕ := sorry  -- This function should calculate the sum based on the spiral placement.

-- Proving that numbers divisible by 68 appear infinitely many times in cell centers
theorem infinite_68_in_cells : ∀ N : ℕ, ∃ n > N, 68 ∣ cell_sum n :=
by sorry

-- Proving that numbers in cell centers do not repeat
theorem no_repeats_in_cells : ∀ m n : ℕ, m ≠ n → cell_sum m ≠ cell_sum n :=
by sorry

end infinite_68_in_cells_no_repeats_in_cells_l1365_136579


namespace composite_fraction_l1365_136584

theorem composite_fraction (x : ℤ) (hx : x = 5^25) : 
  ∃ a b : ℤ, a > 1 ∧ b > 1 ∧ a * b = x^4 + x^3 + x^2 + x + 1 :=
by sorry

end composite_fraction_l1365_136584


namespace find_coordinates_of_P0_find_equation_of_l_l1365_136538

noncomputable def curve (x : ℝ) : ℝ := x^3 + x - 2

def tangent_slope (x : ℝ) : ℝ := 3 * x^2 + 1

def is_in_third_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

def line_eq (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

/-- Problem statement 1: Find the coordinates of P₀ --/
theorem find_coordinates_of_P0 (p0 : ℝ × ℝ)
    (h_tangent_parallel : tangent_slope p0.1 = 4)
    (h_third_quadrant : is_in_third_quadrant p0) :
    p0 = (-1, -4) :=
sorry

/-- Problem statement 2: Find the equation of line l --/
theorem find_equation_of_l (P0 : ℝ × ℝ)
    (h_P0_coordinates: P0 = (-1, -4))
    (h_perpendicular : ∀ (l1_slope : ℝ), l1_slope = 4 → ∃ l_slope : ℝ, l_slope = (-1) / 4)
    (x y : ℝ) : 
    line_eq 1 4 17 x y :=
sorry

end find_coordinates_of_P0_find_equation_of_l_l1365_136538


namespace candy_bar_cost_l1365_136547

def num_quarters := 4
def num_dimes := 3
def num_nickel := 1
def change_received := 4

def value_quarter := 25
def value_dime := 10
def value_nickel := 5

def total_paid := (num_quarters * value_quarter) + (num_dimes * value_dime) + (num_nickel * value_nickel)
def cost_candy_bar := total_paid - change_received

theorem candy_bar_cost : cost_candy_bar = 131 := by
  sorry

end candy_bar_cost_l1365_136547


namespace sum_of_ratios_l1365_136562

theorem sum_of_ratios (a b c : ℤ) (h : (a * a : ℚ) / (b * b) = 32 / 63) : a + b + c = 39 :=
sorry

end sum_of_ratios_l1365_136562


namespace y_intercept_of_line_l1365_136589

theorem y_intercept_of_line (m : ℝ) (x1 y1 : ℝ) (h_slope : m = -3) (h_x_intercept : x1 = 7) (h_y_intercept : y1 = 0) : 
  ∃ y_intercept : ℝ, y_intercept = 21 ∧ y_intercept = -m * 0 + 21 :=
by
  sorry

end y_intercept_of_line_l1365_136589


namespace candle_height_relation_l1365_136592

variables (t : ℝ)

def height_candle_A (t : ℝ) := 12 - 2 * t
def height_candle_B (t : ℝ) := 9 - 2 * t

theorem candle_height_relation : 
  12 - 2 * (15 / 4) = 3 * (9 - 2 * (15 / 4)) :=
by
  sorry

end candle_height_relation_l1365_136592
