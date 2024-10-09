import Mathlib

namespace ratio_of_surface_areas_l1307_130796

theorem ratio_of_surface_areas {r R : ℝ} 
  (h : (4/3) * Real.pi * r^3 / ((4/3) * Real.pi * R^3) = 1 / 8) :
  (4 * Real.pi * r^2) / (4 * Real.pi * R^2) = 1 / 4 := 
sorry

end ratio_of_surface_areas_l1307_130796


namespace total_visitors_count_l1307_130784

def initial_morning_visitors : ℕ := 500
def noon_departures : ℕ := 119
def additional_afternoon_arrivals : ℕ := 138

def afternoon_arrivals : ℕ := noon_departures + additional_afternoon_arrivals
def total_visitors : ℕ := initial_morning_visitors + afternoon_arrivals

theorem total_visitors_count : total_visitors = 757 := 
by sorry

end total_visitors_count_l1307_130784


namespace bob_plate_price_correct_l1307_130748

-- Assuming units and specific values for the problem
def anne_plate_area : ℕ := 20 -- in square units
def bob_clay_usage : ℕ := 600 -- total clay used by Bob in square units
def bob_number_of_plates : ℕ := 15
def anne_plate_price : ℕ := 50 -- in cents
def anne_number_of_plates : ℕ := 30
def total_anne_earnings : ℕ := anne_number_of_plates * anne_plate_price

-- Condition
def bob_plate_area : ℕ := bob_clay_usage / bob_number_of_plates

-- Prove the price of one of Bob's plates
theorem bob_plate_price_correct : bob_number_of_plates * bob_plate_area = bob_clay_usage →
                                  bob_number_of_plates * 100 = total_anne_earnings :=
by
  intros 
  sorry

end bob_plate_price_correct_l1307_130748


namespace value_of_a_squared_plus_b_squared_l1307_130711

variable (a b : ℝ)

theorem value_of_a_squared_plus_b_squared (h1 : a - b = 10) (h2 : a * b = 55) : a^2 + b^2 = 210 := 
by 
sorry

end value_of_a_squared_plus_b_squared_l1307_130711


namespace age_relationships_l1307_130741

variables (a b c d : ℕ)

theorem age_relationships (h1 : a + b = b + c + d + 18) (h2 : 2 * a = 3 * c) :
  c = 2 * a / 3 ∧ d = a / 3 - 18 :=
by
  sorry

end age_relationships_l1307_130741


namespace minimum_value_fraction_l1307_130738

theorem minimum_value_fraction (m n : ℝ) (h1 : m + 4 * n = 1) (h2 : m > 0) (h3 : n > 0): 
  (1 / m + 4 / n) ≥ 25 :=
sorry

end minimum_value_fraction_l1307_130738


namespace smallest_number_jungkook_l1307_130701

theorem smallest_number_jungkook (jungkook yoongi yuna : ℕ) 
  (hj : jungkook = 6 - 3) (hy : yoongi = 4) (hu : yuna = 5) : 
  jungkook < yoongi ∧ jungkook < yuna :=
by
  sorry

end smallest_number_jungkook_l1307_130701


namespace friends_meeting_games_only_l1307_130726

theorem friends_meeting_games_only 
  (M P G MP MG PG MPG : ℕ) 
  (h1 : M + MP + MG + MPG = 10) 
  (h2 : P + MP + PG + MPG = 20) 
  (h3 : MP = 4) 
  (h4 : MG = 2) 
  (h5 : PG = 0) 
  (h6 : MPG = 2) 
  (h7 : M + P + G + MP + MG + PG + MPG = 31) : 
  G = 1 := 
by
  sorry

end friends_meeting_games_only_l1307_130726


namespace minimum_value_of_function_l1307_130727

theorem minimum_value_of_function (x : ℝ) (hx : x > 1) : (x + 4 / (x - 1)) ≥ 5 := by
  sorry

end minimum_value_of_function_l1307_130727


namespace intersection_complement_range_m_l1307_130735

open Set

variable (A : Set ℝ) (B : ℝ → Set ℝ) (m : ℝ)

def setA : Set ℝ := Icc (-1 : ℝ) (3 : ℝ)
def setB (m : ℝ) : Set ℝ := Icc m (m + 6)

theorem intersection_complement (m : ℝ) (h : m = 2) : 
  (setA ∩ (setB 2)ᶜ) = Ico (-1 : ℝ) (2 : ℝ) :=
by
  sorry

theorem range_m (m : ℝ) : 
  A ∪ B m = B m ↔ -3 ≤ m ∧ m ≤ -1 :=
by
  sorry

end intersection_complement_range_m_l1307_130735


namespace column_of_2023_l1307_130705

theorem column_of_2023 : 
  let columns := ["G", "H", "I", "J", "K", "L", "M"]
  let pattern := ["H", "I", "J", "K", "L", "M", "L", "K", "J", "I", "H", "G"]
  let n := 2023
  (pattern.get! ((n - 2) % 12)) = "I" :=
by
  -- Sorry is a placeholder for the proof
  sorry

end column_of_2023_l1307_130705


namespace rectangular_solid_surface_area_l1307_130702

theorem rectangular_solid_surface_area (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) (h_volume : a * b * c = 1001) :
  2 * (a * b + b * c + c * a) = 622 :=
by
  sorry

end rectangular_solid_surface_area_l1307_130702


namespace line_circle_separation_l1307_130771

theorem line_circle_separation (a b : ℝ) (h : a^2 + b^2 < 1) :
    let d := 1 / (Real.sqrt (a^2 + b^2))
    d > 1 := by
    sorry

end line_circle_separation_l1307_130771


namespace percentage_decrease_stock_l1307_130732

theorem percentage_decrease_stock (F J M : ℝ)
  (h1 : J = F - 0.10 * F)
  (h2 : M = J - 0.20 * J) :
  (F - M) / F * 100 = 28 := by
sorry

end percentage_decrease_stock_l1307_130732


namespace jason_picked_7_pears_l1307_130706

def pears_picked_by_jason (total_pears mike_pears : ℕ) : ℕ :=
  total_pears - mike_pears

theorem jason_picked_7_pears :
  pears_picked_by_jason 15 8 = 7 :=
by
  -- Proof is required but we can insert sorry here to skip it for now
  sorry

end jason_picked_7_pears_l1307_130706


namespace pow_addition_l1307_130785

theorem pow_addition : (-2)^2 + 2^2 = 8 :=
by
  sorry

end pow_addition_l1307_130785


namespace parallel_lines_l1307_130728

noncomputable def line1 (x y : ℝ) : Prop := x - y + 1 = 0
noncomputable def line2 (a x y : ℝ) : Prop := x + a * y + 3 = 0

theorem parallel_lines (a x y : ℝ) : (∀ (x y : ℝ), line1 x y → line2 a x y → x = y ∨ (line1 x y ∧ x ≠ y)) → 
  (a = -1 ∧ ∃ d : ℝ, d = Real.sqrt 2) :=
sorry

end parallel_lines_l1307_130728


namespace area_of_rhombus_perimeter_of_rhombus_l1307_130789

-- Definitions and conditions for the area of the rhombus
def d1 : ℕ := 18
def d2 : ℕ := 16

-- Definition for the side length of the rhombus
def side_length : ℕ := 10

-- Statement for the area of the rhombus
theorem area_of_rhombus : (d1 * d2) / 2 = 144 := by
  sorry

-- Statement for the perimeter of the rhombus
theorem perimeter_of_rhombus : 4 * side_length = 40 := by
  sorry

end area_of_rhombus_perimeter_of_rhombus_l1307_130789


namespace find_f3_l1307_130719

theorem find_f3 (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 3 * f (1 - x) = 4 * x^3) : f 3 = -25.5 :=
sorry

end find_f3_l1307_130719


namespace little_john_friends_share_l1307_130715

-- Noncomputable definition for dealing with reals
noncomputable def amount_given_to_each_friend :=
  let total_initial := 7.10
  let total_left := 4.05
  let spent_on_sweets := 1.05
  let total_given_away := total_initial - total_left
  let total_given_to_friends := total_given_away - spent_on_sweets
  total_given_to_friends / 2

-- The theorem stating the result
theorem little_john_friends_share :
  amount_given_to_each_friend = 1.00 :=
by
  sorry

end little_john_friends_share_l1307_130715


namespace man_born_in_1936_l1307_130779

noncomputable def year_of_birth (x : ℕ) : ℕ :=
  x^2 - 2 * x

theorem man_born_in_1936 :
  ∃ x : ℕ, x < 50 ∧ year_of_birth x < 1950 ∧ year_of_birth x = 1892 :=
by
  sorry

end man_born_in_1936_l1307_130779


namespace max_quarters_l1307_130795

theorem max_quarters (a b c : ℕ) (h1 : a + b + c = 120) (h2 : 5 * a + 10 * b + 25 * c = 1000) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c) : c ≤ 19 :=
sorry

example : ∃ a b c : ℕ, a + b + c = 120 ∧ 5 * a + 10 * b + 25 * c = 1000 ∧ 0 < a ∧ 0 < b ∧ 0 < c ∧ c = 19 :=
sorry

end max_quarters_l1307_130795


namespace find_xyz_sum_l1307_130737

variables {x y z : ℝ}

def system_of_equations (x y z : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧
  (x^2 + x * y + y^2 = 12) ∧
  (y^2 + y * z + z^2 = 9) ∧
  (z^2 + z * x + x^2 = 21)

theorem find_xyz_sum (x y z : ℝ) (h : system_of_equations x y z) : 
  x * y + y * z + z * x = 12 :=
sorry

end find_xyz_sum_l1307_130737


namespace infinite_sum_converges_to_3_l1307_130797

theorem infinite_sum_converges_to_3 :
  (∑' k : ℕ, (7 ^ k) / ((4 ^ k - 3 ^ k) * (4 ^ (k + 1) - 3 ^ (k + 1)))) = 3 :=
by
  sorry

end infinite_sum_converges_to_3_l1307_130797


namespace initial_capacity_l1307_130790

theorem initial_capacity (x : ℝ) (h1 : 0.9 * x = 198) : x = 220 :=
by
  sorry

end initial_capacity_l1307_130790


namespace multiplication_modulo_l1307_130758

theorem multiplication_modulo :
  ∃ n : ℕ, (253 * 649 ≡ n [MOD 100]) ∧ (0 ≤ n) ∧ (n < 100) ∧ (n = 97) := 
by
  sorry

end multiplication_modulo_l1307_130758


namespace lana_extra_flowers_l1307_130773

theorem lana_extra_flowers :
  ∀ (tulips roses used total_extra : ℕ),
    tulips = 36 →
    roses = 37 →
    used = 70 →
    total_extra = (tulips + roses - used) →
    total_extra = 3 :=
by
  intros tulips roses used total_extra ht hr hu hte
  rw [ht, hr, hu] at hte
  sorry

end lana_extra_flowers_l1307_130773


namespace not_all_pieces_found_l1307_130722

theorem not_all_pieces_found (k p v : ℕ) (h1 : p + v > 0) (h2 : k % 2 = 1) : k + 4 * p + 8 * v ≠ 1988 :=
by
  sorry

end not_all_pieces_found_l1307_130722


namespace largest_integer_satisfying_conditions_l1307_130764

theorem largest_integer_satisfying_conditions (n : ℤ) (m : ℤ) :
  n^2 = (m + 1)^3 - m^3 ∧ ∃ k : ℤ, 2 * n + 103 = k^2 → n = 313 := 
by 
  sorry

end largest_integer_satisfying_conditions_l1307_130764


namespace Maaza_liters_l1307_130744

theorem Maaza_liters 
  (M L : ℕ)
  (Pepsi : ℕ := 144)
  (Sprite : ℕ := 368)
  (total_liters := M + Pepsi + Sprite)
  (cans_required : ℕ := 281)
  (H : total_liters = cans_required * L)
  : M = 50 :=
by
  sorry

end Maaza_liters_l1307_130744


namespace smallest_balanced_number_l1307_130761

theorem smallest_balanced_number :
  ∃ (a b c : ℕ), 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  100 * a + 10 * b + c = 
  (10 * a + b) + (10 * b + a) + (10 * a + c) + (10 * c + a) + (10 * b + c) + (10 * c + b) ∧ 
  100 * a + 10 * b + c = 132 :=
sorry

end smallest_balanced_number_l1307_130761


namespace q_is_false_of_pq_false_and_notp_false_l1307_130712

variables (p q : Prop)

theorem q_is_false_of_pq_false_and_notp_false (hpq_false : ¬(p ∧ q)) (hnotp_false : ¬(¬p)) : ¬q := 
by 
  sorry

end q_is_false_of_pq_false_and_notp_false_l1307_130712


namespace rachel_took_money_l1307_130713

theorem rachel_took_money (x y : ℕ) (h₁ : x = 5) (h₂ : y = 3) : x - y = 2 :=
by {
  sorry
}

end rachel_took_money_l1307_130713


namespace simplify_fraction_l1307_130746

theorem simplify_fraction : (3^9 / 9^3) = 27 :=
by
  sorry

end simplify_fraction_l1307_130746


namespace probability_of_selecting_quarter_l1307_130740

theorem probability_of_selecting_quarter 
  (value_quarters value_nickels value_pennies total_value : ℚ)
  (coin_value_quarter coin_value_nickel coin_value_penny : ℚ) 
  (h1 : value_quarters = 10)
  (h2 : value_nickels = 10)
  (h3 : value_pennies = 10)
  (h4 : coin_value_quarter = 0.25)
  (h5 : coin_value_nickel = 0.05)
  (h6 : coin_value_penny = 0.01)
  (total_coins : ℚ) 
  (h7 : total_coins = (value_quarters / coin_value_quarter) + (value_nickels / coin_value_nickel) + (value_pennies / coin_value_penny)) : 
  (value_quarters / coin_value_quarter) / total_coins = 1 / 31 :=
by
  sorry

end probability_of_selecting_quarter_l1307_130740


namespace initial_students_count_l1307_130754

theorem initial_students_count (N T : ℕ) (h1 : T = N * 90) (h2 : (T - 120) / (N - 3) = 95) : N = 33 :=
by
  sorry

end initial_students_count_l1307_130754


namespace problem1_problem2_problem3_general_conjecture_l1307_130707

noncomputable def f (x : ℝ) : ℝ := 1 / (2^x + Real.sqrt 2)

-- Prove f(0) + f(1) = sqrt(2) / 2
theorem problem1 : f 0 + f 1 = Real.sqrt 2 / 2 := by
  sorry

-- Prove f(-1) + f(2) = sqrt(2) / 2
theorem problem2 : f (-1) + f 2 = Real.sqrt 2 / 2 := by
  sorry

-- Prove f(-2) + f(3) = sqrt(2) / 2
theorem problem3 : f (-2) + f 3 = Real.sqrt 2 / 2 := by
  sorry

-- Prove ∀ x, f(-x) + f(x+1) = sqrt(2) / 2
theorem general_conjecture (x : ℝ) : f (-x) + f (x + 1) = Real.sqrt 2 / 2 := by
  sorry

end problem1_problem2_problem3_general_conjecture_l1307_130707


namespace width_of_room_l1307_130734

theorem width_of_room 
  (length : ℝ) 
  (cost : ℝ) 
  (rate : ℝ) 
  (h_length : length = 6.5) 
  (h_cost : cost = 10725) 
  (h_rate : rate = 600) 
  : (cost / rate) / length = 2.75 :=
by
  rw [h_length, h_cost, h_rate]
  norm_num

end width_of_room_l1307_130734


namespace sum_of_perimeters_correct_l1307_130798

noncomputable def sum_of_perimeters (s w : ℝ) : ℝ :=
  let l := 2 * w
  let square_area := s^2
  let rectangle_area := l * w
  let sq_perimeter := 4 * s
  let rect_perimeter := 2 * l + 2 * w
  sq_perimeter + rect_perimeter

theorem sum_of_perimeters_correct (s w : ℝ) (h1 : s^2 + 2 * w^2 = 130) (h2 : s^2 - 2 * w^2 = 50) :
  sum_of_perimeters s w = 12 * Real.sqrt 10 + 12 * Real.sqrt 5 :=
by sorry

end sum_of_perimeters_correct_l1307_130798


namespace small_bonsai_sold_eq_l1307_130757

-- Define the conditions
def small_bonsai_cost : ℕ := 30
def big_bonsai_cost : ℕ := 20
def big_bonsai_sold : ℕ := 5
def total_earnings : ℕ := 190

-- The proof problem: Prove that the number of small bonsai sold is 3
theorem small_bonsai_sold_eq : ∃ x : ℕ, 30 * x + 20 * 5 = 190 ∧ x = 3 :=
by
  sorry

end small_bonsai_sold_eq_l1307_130757


namespace range_of_m_l1307_130709

def is_ellipse (m : ℝ) : Prop :=
  ∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

theorem range_of_m (m : ℝ) (h : is_ellipse m) : m > 5 :=
sorry

end range_of_m_l1307_130709


namespace largest_possible_value_of_s_l1307_130783

theorem largest_possible_value_of_s (p q r s : ℝ)
  (h₁ : p + q + r + s = 12)
  (h₂ : pq + pr + ps + qr + qs + rs = 24) : 
  s ≤ 3 + 3 * Real.sqrt 5 :=
sorry

end largest_possible_value_of_s_l1307_130783


namespace problem_statement_l1307_130768

theorem problem_statement (x : ℝ) (h : (x / 5) / 3 = 5 / (x / 3)) : x = 15 ∨ x = -15 := 
by
  sorry

end problem_statement_l1307_130768


namespace ordered_triples_unique_solution_l1307_130723

theorem ordered_triples_unique_solution :
  ∃! (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ ab = c ∧ bc = a ∧ ca = b ∧ a + b + c = 2 :=
sorry

end ordered_triples_unique_solution_l1307_130723


namespace cyclic_sums_sine_cosine_l1307_130724

theorem cyclic_sums_sine_cosine (α β γ : ℝ) (h : α + β + γ = Real.pi) : 
  (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) = 
  2 * (Real.sin α + Real.sin β + Real.sin γ) * 
      (Real.cos α + Real.cos β + Real.cos γ) - 
  2 * (Real.sin α + Real.sin β + Real.sin γ) := 
  sorry

end cyclic_sums_sine_cosine_l1307_130724


namespace calculate_expression_l1307_130718

theorem calculate_expression : (35 / (5 * 2 + 5)) * 6 = 14 :=
by
  sorry

end calculate_expression_l1307_130718


namespace find_integers_divisible_by_18_in_range_l1307_130731

theorem find_integers_divisible_by_18_in_range :
  ∃ n : ℕ, (n % 18 = 0) ∧ (n ≥ 900) ∧ (n ≤ 930) ∧ (n = 900 ∨ n = 918) :=
sorry

end find_integers_divisible_by_18_in_range_l1307_130731


namespace jared_popcorn_l1307_130766

-- Define the given conditions
def pieces_per_serving := 30
def number_of_friends := 3
def pieces_per_friend := 60
def servings_ordered := 9

-- Define the total pieces of popcorn
def total_pieces := servings_ordered * pieces_per_serving

-- Define the total pieces of popcorn eaten by Jared's friends
def friends_total_pieces := number_of_friends * pieces_per_friend

-- State the theorem
theorem jared_popcorn : total_pieces - friends_total_pieces = 90 :=
by 
  -- The detailed proof would go here.
  sorry

end jared_popcorn_l1307_130766


namespace jims_investment_l1307_130760

theorem jims_investment (total_investment : ℝ) (john_ratio : ℝ) (james_ratio : ℝ) (jim_ratio : ℝ) 
                        (h_total_investment : total_investment = 80000)
                        (h_ratio_john : john_ratio = 4)
                        (h_ratio_james : james_ratio = 7)
                        (h_ratio_jim : jim_ratio = 9) : 
    jim_ratio * (total_investment / (john_ratio + james_ratio + jim_ratio)) = 36000 :=
by 
  sorry

end jims_investment_l1307_130760


namespace alice_wrong_questions_l1307_130743

theorem alice_wrong_questions :
  ∃ a b e : ℕ,
    (a + b = 6 + 8 + e) ∧
    (a + 8 = b + 6 + 3) ∧
    a = 9 :=
by {
  sorry
}

end alice_wrong_questions_l1307_130743


namespace solve_linear_system_l1307_130792

-- Given conditions
def matrix : Matrix (Fin 2) (Fin 3) ℚ :=
  ![![1, -1, 1], ![1, 1, 3]]

def system_of_equations (x y : ℚ) : Prop :=
  (x - y = 1) ∧ (x + y = 3)

-- Desired solution
def solution (x y : ℚ) : Prop :=
  x = 2 ∧ y = 1

-- Proof problem statement
theorem solve_linear_system : ∃ x y : ℚ, system_of_equations x y ∧ solution x y := by
  sorry

end solve_linear_system_l1307_130792


namespace min_attempts_to_pair_keys_suitcases_l1307_130782

theorem min_attempts_to_pair_keys_suitcases (n : ℕ) : ∃ p : ℕ, (∀ (keyOpen : Fin n → Fin n), ∃ f : (Fin n × Fin n) → Bool, ∀ (i j : Fin n), i ≠ j → (keyOpen i = j ↔ f (i, j) = tt)) ∧ p = Nat.choose n 2 := by
  sorry

end min_attempts_to_pair_keys_suitcases_l1307_130782


namespace min_radius_cylinder_proof_l1307_130710

-- Defining the radius of the hemisphere
def radius_hemisphere : ℝ := 10

-- Defining the angle alpha which is less than or equal to 30 degrees
def angle_alpha_leq_30 (α : ℝ) : Prop := α ≤ 30 * Real.pi / 180

-- Minimum radius of the cylinder given alpha <= 30 degrees
noncomputable def min_radius_cylinder : ℝ :=
  10 * (2 / Real.sqrt 3 - 1)

theorem min_radius_cylinder_proof (α : ℝ) (hα : angle_alpha_leq_30 α) :
  min_radius_cylinder = 10 * (2 / Real.sqrt 3 - 1) :=
by
  -- Here would go the detailed proof steps
  sorry

end min_radius_cylinder_proof_l1307_130710


namespace area_of_trapezium_l1307_130745

-- Definitions based on conditions
def length_parallel_side1 : ℝ := 20 -- length of the first parallel side
def length_parallel_side2 : ℝ := 18 -- length of the second parallel side
def distance_between_sides : ℝ := 5 -- distance between the parallel sides

-- Statement to prove
theorem area_of_trapezium (a b h : ℝ) :
  a = length_parallel_side1 → b = length_parallel_side2 → h = distance_between_sides →
  (a + b) * h / 2 = 95 :=
by
  intros ha hb hh
  rw [ha, hb, hh]
  sorry

end area_of_trapezium_l1307_130745


namespace row_seat_notation_l1307_130742

-- Define that the notation (4, 5) corresponds to "Row 4, Seat 5"
def notation_row_seat := (4, 5)

-- Prove that "Row 5, Seat 4" should be denoted as (5, 4)
theorem row_seat_notation : (5, 4) = (5, 4) :=
by sorry

end row_seat_notation_l1307_130742


namespace triangle_centroid_altitude_l1307_130794

/-- In triangle XYZ with side lengths XY = 7, XZ = 24, and YZ = 25, the length of GQ where Q 
    is the foot of the altitude from the centroid G to the side YZ is 56/25. -/
theorem triangle_centroid_altitude :
  let XY := 7
  let XZ := 24
  let YZ := 25
  let GQ := 56 / 25
  GQ = (56 : ℝ) / 25 :=
by
  -- proof goes here
  sorry

end triangle_centroid_altitude_l1307_130794


namespace find_k_l1307_130750

theorem find_k (k : ℤ) (h1 : ∃(a b c : ℤ), a = (36 + k) ∧ b = (300 + k) ∧ c = (596 + k) ∧ (∃ d, 
  (a = d^2) ∧ (b = (d + 1)^2) ∧ (c = (d + 2)^2)) ) : k = 925 := by
  sorry

end find_k_l1307_130750


namespace bacteria_growth_l1307_130762

-- Defining the function for bacteria growth
def bacteria_count (t : ℕ) (initial_count : ℕ) (division_time : ℕ) : ℕ :=
  initial_count * 2 ^ (t / division_time)

-- The initial conditions given in the problem
def initial_bacteria : ℕ := 1
def division_interval : ℕ := 10
def total_time : ℕ := 2 * 60

-- Stating the hypothesis and the goal
theorem bacteria_growth : bacteria_count total_time initial_bacteria division_interval = 2 ^ 12 :=
by
  -- Proof would go here
  sorry

end bacteria_growth_l1307_130762


namespace proof_problem_l1307_130791

theorem proof_problem (a b : ℝ) (n : ℕ) 
  (P1 P2 : ℝ × ℝ)
  (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_n_gt_1 : n > 1)
  (h_P1_on_curve : P1.1 ^ n = a * P1.2 ^ n + b)
  (h_P2_on_curve : P2.1 ^ n = a * P2.2 ^ n + b)
  (h_y1_lt_y2 : P1.2 < P2.2)
  (A : ℝ) (h_A : A = (1/2) * |P1.1 * P2.2 - P2.1 * P1.2|) :
  b * P2.2 > 2 * n * P1.2 ^ (n - 1) * a ^ (1 - (1 / n)) * A :=
sorry

end proof_problem_l1307_130791


namespace triangle_angle_ge_60_l1307_130721

theorem triangle_angle_ge_60 {A B C : ℝ} (h : A + B + C = 180) :
  A < 60 ∧ B < 60 ∧ C < 60 → false :=
by
  sorry

end triangle_angle_ge_60_l1307_130721


namespace minimum_value_of_a_l1307_130774

-- Define the given condition
axiom a_pos : ℝ → Prop
axiom positive : ∀ (x : ℝ), 0 < x

-- Definition of the equation
def equation (x y a : ℝ) : Prop :=
  (2 * x - y / Real.exp 1) * Real.log (y / x) = x / (a * Real.exp 1)

-- The mathematical statement we need to prove
theorem minimum_value_of_a (x y a : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x ≠ y) (h_eq : equation x y a) : 
  a ≥ 1 / Real.exp 1 :=
sorry

end minimum_value_of_a_l1307_130774


namespace degree_g_is_six_l1307_130747

theorem degree_g_is_six 
  (f g : Polynomial ℂ) 
  (h : Polynomial ℂ) 
  (h_def : h = f.comp g + Polynomial.X * g) 
  (deg_h : h.degree = 7) 
  (deg_f : f.degree = 3) 
  : g.degree = 6 := 
sorry

end degree_g_is_six_l1307_130747


namespace find_N_l1307_130777

theorem find_N (a b c N : ℝ) (h1 : a + b + c = 120) (h2 : a - 10 = N) 
               (h3 : b + 10 = N) (h4 : 7 * c = N): N = 56 :=
by
  sorry

end find_N_l1307_130777


namespace inequality_ac2_geq_bc2_l1307_130736

theorem inequality_ac2_geq_bc2 (a b c : ℝ) (h : a > b) : a * c^2 ≥ b * c^2 :=
sorry

end inequality_ac2_geq_bc2_l1307_130736


namespace red_card_events_l1307_130714

-- Definitions based on the conditions
inductive Person
| A | B | C | D

inductive Card
| Red | Black | Blue | White

-- Definition of the events
def event_A_receives_red (distribution : Person → Card) : Prop :=
  distribution Person.A = Card.Red

def event_B_receives_red (distribution : Person → Card) : Prop :=
  distribution Person.B = Card.Red

-- The relationship between the two events
def mutually_exclusive_but_not_opposite (distribution : Person → Card) : Prop :=
  (event_A_receives_red distribution → ¬ event_B_receives_red distribution) ∧
  (event_B_receives_red distribution → ¬ event_A_receives_red distribution)

-- The formal theorem statement
theorem red_card_events (distribution : Person → Card) :
  mutually_exclusive_but_not_opposite distribution :=
sorry

end red_card_events_l1307_130714


namespace commute_weeks_per_month_l1307_130765

variable (total_commute_one_way : ℕ)
variable (gas_cost_per_gallon : ℝ)
variable (car_mileage : ℝ)
variable (commute_days_per_week : ℕ)
variable (individual_monthly_payment : ℝ)
variable (number_of_people : ℕ)

theorem commute_weeks_per_month :
  total_commute_one_way = 21 →
  gas_cost_per_gallon = 2.5 →
  car_mileage = 30 →
  commute_days_per_week = 5 →
  individual_monthly_payment = 14 →
  number_of_people = 5 →
  (individual_monthly_payment * number_of_people) / 
  ((total_commute_one_way * 2 / car_mileage) * gas_cost_per_gallon * commute_days_per_week) = 4 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end commute_weeks_per_month_l1307_130765


namespace darkest_cell_product_l1307_130778

theorem darkest_cell_product (a b c d : ℕ)
  (h1 : a > 1) (h2 : b > 1) (h3 : c = a * b)
  (h4 : d = c * (9 * 5) * (9 * 11)) :
  d = 245025 :=
by
  sorry

end darkest_cell_product_l1307_130778


namespace simplify_and_evaluate_l1307_130703

theorem simplify_and_evaluate (x : ℝ) (h : x = 3) : ((x - 2) / (x - 1)) / ((x + 1) - (3 / (x - 1))) = 1 / 5 :=
by
  sorry

end simplify_and_evaluate_l1307_130703


namespace find_non_integer_solution_l1307_130781

noncomputable def q (x y : ℝ) (b : Fin 10 → ℝ) : ℝ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 +
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3

theorem find_non_integer_solution (b : Fin 10 → ℝ)
  (h0 : q 0 0 b = 0)
  (h1 : q 1 0 b = 0)
  (h2 : q (-1) 0 b = 0)
  (h3 : q 0 1 b = 0)
  (h4 : q 0 (-1) b = 0)
  (h5 : q 1 1 b = 0)
  (h6 : q 1 (-1) b = 0)
  (h7 : q (-1) 1 b = 0)
  (h8 : q (-1) (-1) b = 0) :
  ∃ r s : ℝ, q r s b = 0 ∧ ¬ (∃ n : ℤ, r = n) ∧ ¬ (∃ n : ℤ, s = n) :=
sorry

end find_non_integer_solution_l1307_130781


namespace part1_part2_l1307_130770

def f (x m : ℝ) : ℝ := |x - m| - |x + 3 * m|

theorem part1 (x : ℝ) : 
  (∀ (x : ℝ), f x 1 ≥ 1 → x ≤ -3 / 2) :=
sorry

theorem part2 (x t : ℝ) (h : ∀ (x t : ℝ), f x m < |2 + t| + |t - 1|) : 
  0 < m ∧ m < 3 / 4 :=
sorry

end part1_part2_l1307_130770


namespace lowest_possible_score_l1307_130716

def total_points_first_four_tests : ℕ := 82 + 90 + 78 + 85
def required_total_points_for_seven_tests : ℕ := 80 * 7
def points_needed_for_last_three_tests : ℕ :=
  required_total_points_for_seven_tests - total_points_first_four_tests

theorem lowest_possible_score 
  (max_points_per_test : ℕ)
  (points_first_four_tests : ℕ := total_points_first_four_tests)
  (required_points : ℕ := required_total_points_for_seven_tests)
  (total_points_needed_last_three : ℕ := points_needed_for_last_three_tests) :
  ∃ (lowest_score : ℕ), 
    max_points_per_test = 100 ∧
    points_first_four_tests = 335 ∧
    required_points = 560 ∧
    total_points_needed_last_three = 225 ∧
    lowest_score = 25 :=
by
  sorry

end lowest_possible_score_l1307_130716


namespace min_folds_to_exceed_thickness_l1307_130717

def initial_thickness : ℝ := 0.1
def desired_thickness : ℝ := 12

theorem min_folds_to_exceed_thickness : ∃ (n : ℕ), initial_thickness * 2^n > desired_thickness ∧ ∀ m < n, initial_thickness * 2^m ≤ desired_thickness := by
  sorry

end min_folds_to_exceed_thickness_l1307_130717


namespace math_problem_l1307_130753

variables (a b c d m : ℤ)

theorem math_problem (h1 : a = -b) (h2 : c * d = 1) (h3 : m = -1) : c * d - a - b + m^2022 = 2 :=
by
  sorry

end math_problem_l1307_130753


namespace transformed_line_l1307_130776

-- Define the original line equation
def original_line (x y : ℝ) : Prop := (x - 2 * y = 2)

-- Define the transformation
def transformation (x y x' y' : ℝ) : Prop :=
  (x' = x) ∧ (y' = 2 * y)

-- Prove that the transformed line equation holds
theorem transformed_line (x y x' y' : ℝ) (h₁ : original_line x y) (h₂ : transformation x y x' y') :
  x' - y' = 2 :=
sorry

end transformed_line_l1307_130776


namespace simplify_exp_l1307_130749

theorem simplify_exp : (10^8 / (10 * 10^5)) = 100 := 
by
  -- The proof is omitted; we are stating the problem.
  sorry

end simplify_exp_l1307_130749


namespace average_speed_increased_pace_l1307_130751

theorem average_speed_increased_pace 
  (speed_constant : ℝ) (time_constant : ℝ) (distance_increased : ℝ) (total_time : ℝ) 
  (h1 : speed_constant = 15) 
  (h2 : time_constant = 3) 
  (h3 : distance_increased = 190) 
  (h4 : total_time = 13) :
  (distance_increased / (total_time - time_constant)) = 19 :=
by
  sorry

end average_speed_increased_pace_l1307_130751


namespace total_trolls_l1307_130720

noncomputable def troll_count (forest bridge plains : ℕ) : ℕ := forest + bridge + plains

theorem total_trolls (forest_trolls bridge_trolls plains_trolls : ℕ)
  (h1 : forest_trolls = 6)
  (h2 : bridge_trolls = 4 * forest_trolls - 6)
  (h3 : plains_trolls = bridge_trolls / 2) :
  troll_count forest_trolls bridge_trolls plains_trolls = 33 :=
by
  -- Proof steps would be filled in here
  sorry

end total_trolls_l1307_130720


namespace solution_set_of_new_inequality_l1307_130730

-- Define the conditions
variable (a b c x : ℝ)

-- ax^2 + bx + c > 0 has solution set {-3 < x < 2}
def inequality_solution_set (a b c : ℝ) : Prop := ∀ x : ℝ, (-3 < x ∧ x < 2) → a * x^2 + b * x + c > 0

-- Prove that cx^2 + bx + a > 0 has solution set {x < -1/3 ∨ x > 1/2}
theorem solution_set_of_new_inequality
  (a b c : ℝ)
  (h : a < 0 ∧ inequality_solution_set a b c) :
  ∀ x : ℝ, (x < -1/3 ∨ x > 1/2) ↔ (c * x^2 + b * x + a > 0) := sorry

end solution_set_of_new_inequality_l1307_130730


namespace combined_probability_l1307_130769

-- Definitions:
def number_of_ways_to_get_3_heads_and_1_tail := Nat.choose 4 3
def probability_of_specific_sequence_of_3_heads_and_1_tail := (1/2) ^ 4
def probability_of_3_heads_and_1_tail := number_of_ways_to_get_3_heads_and_1_tail * probability_of_specific_sequence_of_3_heads_and_1_tail

def favorable_outcomes_die := 2
def total_outcomes_die := 6
def probability_of_number_greater_than_4 := favorable_outcomes_die / total_outcomes_die

-- Proof statement:
theorem combined_probability : probability_of_3_heads_and_1_tail * probability_of_number_greater_than_4 = 1/12 := by
  sorry

end combined_probability_l1307_130769


namespace largest_square_side_l1307_130733

variable (length width : ℕ)
variable (h_length : length = 54)
variable (h_width : width = 20)
variable (num_squares : ℕ)
variable (h_num_squares : num_squares = 3)

theorem largest_square_side : (length : ℝ) / num_squares = 18 := by
  sorry

end largest_square_side_l1307_130733


namespace pieces_left_l1307_130780

def pieces_initial : ℕ := 900
def pieces_used : ℕ := 156

theorem pieces_left : pieces_initial - pieces_used = 744 := by
  sorry

end pieces_left_l1307_130780


namespace tire_circumference_l1307_130756

/-- 
Given:
1. The tire rotates at 400 revolutions per minute.
2. The car is traveling at a speed of 168 km/h.

Prove that the circumference of the tire is 7 meters.
-/
theorem tire_circumference (rpm : ℕ) (speed_km_h : ℕ) (C : ℕ) 
  (h1 : rpm = 400) 
  (h2 : speed_km_h = 168)
  (h3 : C = 7) : 
  C = (speed_km_h * 1000 / 60) / rpm :=
by
  rw [h1, h2]
  exact h3

end tire_circumference_l1307_130756


namespace magnitude_difference_l1307_130793

open Complex

noncomputable def c1 : ℂ := 18 - 5 * I
noncomputable def c2 : ℂ := 14 + 6 * I
noncomputable def c3 : ℂ := 3 - 12 * I
noncomputable def c4 : ℂ := 4 + 9 * I

theorem magnitude_difference : 
  Complex.abs ((c1 * c2) - (c3 * c4)) = Real.sqrt 146365 :=
by
  sorry

end magnitude_difference_l1307_130793


namespace stagePlayRolesAssignment_correct_l1307_130755

noncomputable def stagePlayRolesAssignment : ℕ :=
  let male_roles : ℕ := 4 * 3 -- ways to assign male roles
  let female_roles : ℕ := 5 * 4 -- ways to assign female roles
  let either_gender_roles : ℕ := 5 * 4 * 3 -- ways to assign either-gender roles
  male_roles * female_roles * either_gender_roles -- total assignments

theorem stagePlayRolesAssignment_correct : stagePlayRolesAssignment = 14400 := by
  sorry

end stagePlayRolesAssignment_correct_l1307_130755


namespace candy_cost_l1307_130767

theorem candy_cost (C : ℝ) 
  (h1 : 20 * C + 80 * 5 = 100 * 6) : 
  C = 10 := 
by
  sorry

end candy_cost_l1307_130767


namespace sarah_min_width_l1307_130725

noncomputable def minWidth (S : Type) [LinearOrder S] (w : S) : Prop :=
  ∃ w, w ≥ 0 ∧ w * (w + 20) ≥ 150 ∧ ∀ w', (w' ≥ 0 ∧ w' * (w' + 20) ≥ 150) → w ≤ w'

theorem sarah_min_width : minWidth ℝ 10 :=
by {
  sorry -- proof goes here
}

end sarah_min_width_l1307_130725


namespace problem_conditions_l1307_130759

theorem problem_conditions (x y : ℝ) (h : x^2 + y^2 - x * y = 1) :
  ¬ (x + y ≤ 1) ∧ (x + y ≥ -2) ∧ (x^2 + y^2 ≤ 2) ∧ ¬ (x^2 + y^2 ≥ 1) :=
by
  sorry

end problem_conditions_l1307_130759


namespace total_amount_withdrawn_l1307_130729

def principal : ℤ := 20000
def interest_rate : ℚ := 3.33 / 100
def term : ℤ := 3

theorem total_amount_withdrawn :
  principal + (principal * interest_rate * term) = 21998 := by
  sorry

end total_amount_withdrawn_l1307_130729


namespace find_n_l1307_130752

open Nat

def is_solution_of_comb_perm (n : ℕ) : Prop :=
    3 * (factorial (n-1) / (factorial (n-5) * factorial 4)) = 5 * (n-2) * (n-3)

theorem find_n (n : ℕ) (h : is_solution_of_comb_perm n) (hn : n ≠ 0) : n = 9 :=
by
  -- will fill proof steps if required
  sorry

end find_n_l1307_130752


namespace number_of_n_such_that_n_div_25_minus_n_is_square_l1307_130700

theorem number_of_n_such_that_n_div_25_minus_n_is_square :
  ∃! n1 n2 : ℤ, ∀ n : ℤ, (n = n1 ∨ n = n2) ↔ ∃ k : ℤ, k^2 = n / (25 - n) :=
sorry

end number_of_n_such_that_n_div_25_minus_n_is_square_l1307_130700


namespace min_value_change_when_2x2_added_l1307_130739

variable (f : ℝ → ℝ)
variable (a b c : ℝ)

def quadratic (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem min_value_change_when_2x2_added
  (a b : ℝ)
  (h1 : ∀ x : ℝ, f x = a * x^2 + b * x + c)
  (h2 : ∀ x : ℝ, (a + 1) * x^2 + b * x + c > a * x^2 + b * x + c + 1)
  (h3 : ∀ x : ℝ, (a - 1) * x^2 + b * x + c < a * x^2 + b * x + c - 3) :
  ∀ x : ℝ, (a + 2) * x^2 + b * x + c = a * x^2 + b * x + (c + 1.5) :=
sorry

end min_value_change_when_2x2_added_l1307_130739


namespace winter_melon_ratio_l1307_130799

theorem winter_melon_ratio (T Ok_sales Choc_sales : ℕ) (hT : T = 50) 
  (hOk : Ok_sales = 3 * T / 10) (hChoc : Choc_sales = 15) :
  (T - (Ok_sales + Choc_sales)) / T = 2 / 5 :=
by
  sorry

end winter_melon_ratio_l1307_130799


namespace jim_catches_bob_in_20_minutes_l1307_130772

theorem jim_catches_bob_in_20_minutes
  (bob_speed : ℝ)
  (jim_speed : ℝ)
  (bob_head_start : ℝ)
  (bob_speed_mph : bob_speed = 6)
  (jim_speed_mph : jim_speed = 9)
  (bob_headstart_miles : bob_head_start = 1) :
  ∃ (m : ℝ), m = 20 := 
by
  sorry

end jim_catches_bob_in_20_minutes_l1307_130772


namespace seating_arrangement_l1307_130763

theorem seating_arrangement (students : ℕ) (desks : ℕ) (empty_desks : ℕ) 
  (h_students : students = 2) (h_desks : desks = 5) 
  (h_empty : empty_desks ≥ 1) :
  ∃ ways, ways = 12 := by
  sorry

end seating_arrangement_l1307_130763


namespace x_eq_3_minus_2t_and_y_eq_3t_plus_6_l1307_130788

theorem x_eq_3_minus_2t_and_y_eq_3t_plus_6 (t : ℝ) (x : ℝ) (y : ℝ) : x = 3 - 2 * t → y = 3 * t + 6 → x = 0 → y = 10.5 :=
by
  sorry

end x_eq_3_minus_2t_and_y_eq_3t_plus_6_l1307_130788


namespace product_xyz_is_minus_one_l1307_130775

-- Definitions of the variables and equations
variables (x y z : ℝ)

-- Assumptions based on the given conditions
def condition1 : Prop := x + (1 / y) = 2
def condition2 : Prop := y + (1 / z) = 2
def condition3 : Prop := z + (1 / x) = 2

-- The theorem stating the conclusion to be proved
theorem product_xyz_is_minus_one (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 z x) : x * y * z = -1 :=
by sorry

end product_xyz_is_minus_one_l1307_130775


namespace geometric_sum_l1307_130704

def S10 : ℕ := 36
def S20 : ℕ := 48

theorem geometric_sum (S30 : ℕ) (h1 : S10 = 36) (h2 : S20 = 48) : S30 = 52 :=
by
  have h3 : (S20 - S10) ^ 2 = S10 * (S30 - S20) :=
    sorry -- This is based on the properties of the geometric sequence
  sorry  -- Solve the equation to show S30 = 52

end geometric_sum_l1307_130704


namespace worker_late_by_10_minutes_l1307_130787

def usual_time : ℕ := 40
def speed_ratio : ℚ := 4 / 5
def time_new := (usual_time : ℚ) * (5 / 4) -- This is the equation derived from solving

theorem worker_late_by_10_minutes : 
  ((time_new : ℚ) - usual_time) = 10 :=
by
  sorry -- proof is skipped

end worker_late_by_10_minutes_l1307_130787


namespace correct_statements_l1307_130786

/-- The line (3+m)x+4y-3+3m=0 (m ∈ ℝ) always passes through the fixed point (-3, 3) -/
def statement1 (m : ℝ) : Prop :=
  ∀ x y : ℝ, (3 + m) * x + 4 * y - 3 + 3 * m = 0 → (x = -3 ∧ y = 3)

/-- For segment AB with endpoint B at (3,4) and A moving on the circle x²+y²=4,
    the trajectory equation of the midpoint M of segment AB is (x - 3/2)²+(y - 2)²=1 -/
def statement2 : Prop :=
  ∀ x y x1 y1 : ℝ, ((x1, y1) : ℝ × ℝ) ∈ {p | p.1^2 + p.2^2 = 4} → x = (x1 + 3) / 2 → y = (y1 + 4) / 2 → 
    (x - 3 / 2)^2 + (y - 2)^2 = 1

/-- Given M = {(x, y) | y = √(1 - x²)} and N = {(x, y) | y = x + b},
    if M ∩ N ≠ ∅, then b ∈ [-√2, √2] -/
def statement3 (b : ℝ) : Prop :=
  ∃ x y : ℝ, y = Real.sqrt (1 - x^2) ∧ y = x + b → b ∈ [-Real.sqrt 2, Real.sqrt 2]

/-- Given the circle C: (x - b)² + (y - c)² = a² (a > 0, b > 0, c > 0) intersects the x-axis and is
    separate from the y-axis, then the intersection point of the line ax + by + c = 0 and the line
    x + y + 1 = 0 is in the second quadrant -/
def statement4 (a b c : ℝ) : Prop :=
  a > 0 → b > 0 → c > 0 → b > a → a > c →
  ∃ x y : ℝ, (a * x + b * y + c = 0 ∧ x + y + 1 = 0) ∧ x < 0 ∧ y > 0

/-- Among the statements, the correct ones are 1, 2, and 4 -/
theorem correct_statements : 
  (∀ m : ℝ, statement1 m) ∧ statement2 ∧ (∀ b : ℝ, ¬ statement3 b) ∧ 
  (∀ a b c : ℝ, statement4 a b c) :=
by sorry

end correct_statements_l1307_130786


namespace proof_ac_plus_bd_l1307_130708

theorem proof_ac_plus_bd (a b c d : ℝ)
  (h1 : a + b + c = 10)
  (h2 : a + b + d = -6)
  (h3 : a + c + d = 0)
  (h4 : b + c + d = 15) :
  ac + bd = -130.111 := 
by
  sorry

end proof_ac_plus_bd_l1307_130708
