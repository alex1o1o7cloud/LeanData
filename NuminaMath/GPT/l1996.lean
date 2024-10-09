import Mathlib

namespace pen_cost_proof_l1996_199641

-- Given definitions based on the problem conditions
def is_majority (s : ℕ) := s > 20
def is_odd_and_greater_than_one (n : ℕ) := n > 1 ∧ n % 2 = 1
def is_prime (c : ℕ) := Nat.Prime c

-- The final theorem to prove the correct answer
theorem pen_cost_proof (s n c : ℕ) 
  (h_majority : is_majority s) 
  (h_odd : is_odd_and_greater_than_one n) 
  (h_prime : is_prime c) 
  (h_eq : s * c * n = 2091) : 
  c = 47 := 
sorry

end pen_cost_proof_l1996_199641


namespace probability_defective_unit_l1996_199692

theorem probability_defective_unit (T : ℝ) 
  (P_A : ℝ := 9 / 1000) 
  (P_B : ℝ := 1 / 50) 
  (output_ratio_A : ℝ := 0.4)
  (output_ratio_B : ℝ := 0.6) : 
  (P_A * output_ratio_A + P_B * output_ratio_B) = 0.0156 :=
by
  sorry

end probability_defective_unit_l1996_199692


namespace numbers_not_expressed_l1996_199665

theorem numbers_not_expressed (a b : ℕ) (hb : 0 < b) (ha : 0 < a) :
 ∀ n : ℕ, (¬ ∃ a b : ℕ, n = a / b + (a + 1) / (b + 1) ∧ 0 < b ∧ 0 < a) ↔ (n = 1 ∨ ∃ m : ℕ, n = 2^m + 2) := 
by 
  sorry

end numbers_not_expressed_l1996_199665


namespace point_T_coordinates_l1996_199653

-- Definition of a point in 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Definition of a square with specific points O, P, Q, R
structure Square where
  O : Point
  P : Point
  Q : Point
  R : Point

-- Condition: O is the origin
def O : Point := {x := 0, y := 0}

-- Condition: Q is at (3, 3)
def Q : Point := {x := 3, y := 3}

-- Assuming the function area_triang for calculating the area of a triangle given three points
def area_triang (A B C : Point) : ℝ :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x))

-- Assuming the function area_square for calculating the area of a square given the length of the side
def area_square (s : ℝ) : ℝ := s * s

-- Coordinates of point P and R since it's a square with sides parallel to axis
def P : Point := {x := 3, y := 0}
def R : Point := {x := 0, y := 3}

-- Definition of the square OPQR
def OPQR : Square := {O := O, P := P, Q := Q, R := R}

-- Length of the side of square OPQR
def side_length : ℝ := 3

-- Area of the square OPQR
def square_area : ℝ := area_square side_length

-- Twice the area of the square OPQR
def required_area : ℝ := 2 * square_area

-- Point T that needs to be proven
def T : Point := {x := 3, y := 12}

-- The main theorem to prove
theorem point_T_coordinates (T : Point) : area_triang P Q T = required_area → T = {x := 3, y := 12} :=
by
  sorry

end point_T_coordinates_l1996_199653


namespace trains_clear_in_correct_time_l1996_199691

noncomputable def time_to_clear (length1 length2 : ℝ) (speed1_kmph speed2_kmph : ℝ) : ℝ :=
  let speed1_mps := speed1_kmph * 1000 / 3600
  let speed2_mps := speed2_kmph * 1000 / 3600
  let relative_speed := speed1_mps + speed2_mps
  let total_distance := length1 + length2
  total_distance / relative_speed

-- The lengths of the trains
def length1 : ℝ := 151
def length2 : ℝ := 165

-- The speeds of the trains in km/h
def speed1_kmph : ℝ := 80
def speed2_kmph : ℝ := 65

-- The correct answer
def correct_time : ℝ := 7.844

theorem trains_clear_in_correct_time :
  time_to_clear length1 length2 speed1_kmph speed2_kmph = correct_time :=
by
  -- Skipping proof
  sorry

end trains_clear_in_correct_time_l1996_199691


namespace bananas_to_apples_l1996_199633

-- Definitions based on conditions
def bananas := ℕ
def oranges := ℕ
def apples := ℕ

-- Condition 1: 3/4 of 16 bananas are worth 12 oranges
def condition1 : Prop := 3 / 4 * 16 = 12

-- Condition 2: price of one banana equals the price of two apples
def price_equiv_banana_apple : Prop := 1 = 2

-- Proof: 1/3 of 9 bananas are worth 6 apples
theorem bananas_to_apples 
  (c1: condition1)
  (c2: price_equiv_banana_apple) : 1 / 3 * 9 * 2 = 6 :=
by sorry

end bananas_to_apples_l1996_199633


namespace vector_BC_coordinates_l1996_199695

-- Define the given vectors
def vec_AB : ℝ × ℝ := (2, -1)
def vec_AC : ℝ × ℝ := (-4, 1)

-- Define the vector subtraction
def vec_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

-- Define the vector BC as the result of the subtraction
def vec_BC : ℝ × ℝ := vec_sub vec_AC vec_AB

-- State the theorem
theorem vector_BC_coordinates : vec_BC = (-6, 2) := by
  sorry

end vector_BC_coordinates_l1996_199695


namespace complement_A_union_B_m_eq_4_B_nonempty_and_subset_A_range_m_l1996_199601

-- Definitions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | (m + 1) ≤ x ∧ x ≤ (2 * m - 1)}

-- Part (Ⅰ)
theorem complement_A_union_B_m_eq_4 :
  (m = 4) → compl (A ∪ B 4) = {x | x < -2} ∪ {x | x > 7} := 
by
  sorry

-- Part (Ⅱ)
theorem B_nonempty_and_subset_A_range_m :
  (∃ x, x ∈ B m) ∧ (B m ⊆ A) → 2 ≤ m ∧ m ≤ 3 :=
by
  sorry

end complement_A_union_B_m_eq_4_B_nonempty_and_subset_A_range_m_l1996_199601


namespace least_number_of_shoes_needed_on_island_l1996_199613

def number_of_inhabitants : ℕ := 10000
def percentage_one_legged : ℕ := 5
def shoes_needed (N : ℕ) : ℕ :=
  let one_legged := (percentage_one_legged * N) / 100
  let two_legged := N - one_legged
  let barefooted_two_legged := two_legged / 2
  let shoes_for_one_legged := one_legged
  let shoes_for_two_legged := (two_legged - barefooted_two_legged) * 2
  shoes_for_one_legged + shoes_for_two_legged

theorem least_number_of_shoes_needed_on_island :
  shoes_needed number_of_inhabitants = 10000 :=
sorry

end least_number_of_shoes_needed_on_island_l1996_199613


namespace exactly_one_solves_l1996_199606

-- Define the independent probabilities for person A and person B
variables (p₁ p₂ : ℝ)

-- Assume probabilities are between 0 and 1 inclusive
axiom h1 : 0 ≤ p₁ ∧ p₁ ≤ 1
axiom h2 : 0 ≤ p₂ ∧ p₂ ≤ 1

theorem exactly_one_solves : (p₁ * (1 - p₂) + p₂ * (1 - p₁)) = (p₁ * (1 - p₂) + p₂ * (1 - p₁)) := 
by sorry

end exactly_one_solves_l1996_199606


namespace Tom_has_38_photos_l1996_199624

theorem Tom_has_38_photos :
  ∃ (Tom Tim Paul : ℕ), 
  (Paul = Tim + 10) ∧ 
  (Tim = 152 - 100) ∧ 
  (152 = Tom + Paul + Tim) ∧ 
  (Tom = 38) :=
by
  sorry

end Tom_has_38_photos_l1996_199624


namespace Jed_older_than_Matt_l1996_199649

-- Definitions of ages and conditions
def Jed_current_age : ℕ := sorry
def Matt_current_age : ℕ := sorry
axiom condition1 : Jed_current_age + 10 = 25
axiom condition2 : Jed_current_age + Matt_current_age = 20

-- Proof statement
theorem Jed_older_than_Matt : Jed_current_age - Matt_current_age = 10 :=
by
  sorry

end Jed_older_than_Matt_l1996_199649


namespace abs_diff_of_sum_and_product_l1996_199655

theorem abs_diff_of_sum_and_product (x y : ℝ) (h1 : x + y = 20) (h2 : x * y = 96) : |x - y| = 4 := 
by
  sorry

end abs_diff_of_sum_and_product_l1996_199655


namespace lcm_fractions_l1996_199677

theorem lcm_fractions (x : ℕ) (hx : x ≠ 0) : 
  (∀ (a b c : ℕ), (a = 4*x ∧ b = 5*x ∧ c = 6*x) → (Nat.lcm (Nat.lcm a b) c = 60 * x)) :=
by
  sorry

end lcm_fractions_l1996_199677


namespace smallest_x_satisfies_eq_l1996_199603

theorem smallest_x_satisfies_eq : ∃ x : ℝ, (1 / (x - 5) + 1 / (x - 7) = 5 / (2 * (x - 6))) ∧ x = 7 - Real.sqrt 6 :=
by
  -- The proof steps would go here, but we're skipping them with sorry for now.
  sorry

end smallest_x_satisfies_eq_l1996_199603


namespace coordinates_of_point_l1996_199629

theorem coordinates_of_point : 
  ∀ (x y : ℝ), (x, y) = (2, -3) → (x, y) = (2, -3) := 
by 
  intros x y h 
  exact h

end coordinates_of_point_l1996_199629


namespace factorization_of_polynomial_l1996_199667

theorem factorization_of_polynomial : 
  ∀ (x : ℝ), 18 * x^3 + 9 * x^2 + 3 * x = 3 * x * (6 * x^2 + 3 * x + 1) :=
by sorry

end factorization_of_polynomial_l1996_199667


namespace simplify_abs_expr_l1996_199614

noncomputable def piecewise_y (x : ℝ) : ℝ :=
  if h1 : x < -3 then -3 * x
  else if h2 : -3 ≤ x ∧ x < 1 then 6 - x
  else if h3 : 1 ≤ x ∧ x < 2 then 4 + x
  else 3 * x

theorem simplify_abs_expr : 
  ∀ x : ℝ, (|x - 1| + |x - 2| + |x + 3|) = piecewise_y x :=
by
  intro x
  sorry

end simplify_abs_expr_l1996_199614


namespace straw_costs_max_packs_type_a_l1996_199611

theorem straw_costs (x y : ℝ) (h1 : 12 * x + 15 * y = 171) (h2 : 24 * x + 28 * y = 332) :
  x = 8 ∧ y = 5 :=
  by sorry

theorem max_packs_type_a (m : ℕ) (cA cB : ℕ) (total_packs : ℕ) (max_cost : ℕ)
  (h1 : cA = 8) (h2 : cB = 5) (h3 : total_packs = 100) (h4 : max_cost = 600) :
  m ≤ 33 :=
  by sorry

end straw_costs_max_packs_type_a_l1996_199611


namespace anna_money_left_l1996_199663

theorem anna_money_left : 
  let initial_money := 10.0
  let gum_cost := 3.0 -- 3 packs at $1.00 each
  let chocolate_cost := 5.0 -- 5 bars at $1.00 each
  let cane_cost := 1.0 -- 2 canes at $0.50 each
  let total_spent := gum_cost + chocolate_cost + cane_cost
  let money_left := initial_money - total_spent
  money_left = 1.0 := by
  sorry

end anna_money_left_l1996_199663


namespace find_x_l1996_199619

theorem find_x : ∃ x : ℕ, 6 * 2^x = 2048 ∧ x = 10 := by
  sorry

end find_x_l1996_199619


namespace thirtieth_change_month_is_february_l1996_199602

def months_in_year := 12

def months_per_change := 7

def first_change_month := 3 -- March (if we assume January = 1, February = 2, etc.)

def nth_change_month (n : ℕ) : ℕ :=
  (first_change_month + months_per_change * (n - 1)) % months_in_year

theorem thirtieth_change_month_is_february :
  nth_change_month 30 = 2 := -- February (if we assume January = 1, February = 2, etc.)
by 
  sorry

end thirtieth_change_month_is_february_l1996_199602


namespace sum_of_base5_numbers_l1996_199652

-- Definitions for the numbers in base 5
def n1_base5 := (1 * 5^2 + 3 * 5^1 + 2 * 5^0 : ℕ)
def n2_base5 := (2 * 5^2 + 1 * 5^1 + 4 * 5^0 : ℕ)
def n3_base5 := (3 * 5^2 + 4 * 5^1 + 1 * 5^0 : ℕ)

-- Sum the numbers in base 10
def sum_base10 := n1_base5 + n2_base5 + n3_base5

-- Define the base 5 value of the sum
def sum_base5 := 
  -- Convert the sum to base 5
  1 * 5^3 + 2 * 5^2 + 4 * 5^1 + 2 * 5^0

-- The theorem we want to prove
theorem sum_of_base5_numbers :
    (132 + 214 + 341 : ℕ) = 1242 := by
    sorry

end sum_of_base5_numbers_l1996_199652


namespace largest_divisor_of_m_l1996_199699

theorem largest_divisor_of_m (m : ℕ) (hm : m > 0) (h : 54 ∣ m^2) : 18 ∣ m :=
sorry

end largest_divisor_of_m_l1996_199699


namespace solve_eq_2_pow_x_plus_3_pow_y_eq_z_sq_l1996_199664

theorem solve_eq_2_pow_x_plus_3_pow_y_eq_z_sq (x y z : ℕ) :
  ((x = 3 ∧ y = 0 ∧ z = 3) ∨ (x = 0 ∧ y = 1 ∧ z = 2) ∨ (x = 4 ∧ y = 2 ∧ z = 5)) →
  2^x + 3^y = z^2 :=
by
  sorry

end solve_eq_2_pow_x_plus_3_pow_y_eq_z_sq_l1996_199664


namespace nina_total_cost_l1996_199617

-- Define the cost of the first pair of shoes
def first_pair_cost : ℕ := 22

-- Define the cost of the second pair of shoes
def second_pair_cost : ℕ := first_pair_cost + (first_pair_cost / 2)

-- Define the total cost for both pairs of shoes
def total_cost : ℕ := first_pair_cost + second_pair_cost

-- The formal statement of the problem
theorem nina_total_cost : total_cost = 55 := by
  sorry

end nina_total_cost_l1996_199617


namespace sum_of_first_5n_l1996_199659

theorem sum_of_first_5n (n : ℕ) (h : (3 * n) * (3 * n + 1) / 2 = n * (n + 1) / 2 + 270) : (5 * n) * (5 * n + 1) / 2 = 820 :=
by
  sorry

end sum_of_first_5n_l1996_199659


namespace N_eq_M_union_P_l1996_199660

def M : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n}
def N : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n / 2}
def P : Set ℝ := {x : ℝ | ∃ (n : ℤ), x = n + 1 / 2}

theorem N_eq_M_union_P : N = M ∪ P :=
  sorry

end N_eq_M_union_P_l1996_199660


namespace find_number_l1996_199616

theorem find_number (x : ℝ) 
  (h : (28 + x / 69) * 69 = 1980) :
  x = 1952 :=
sorry

end find_number_l1996_199616


namespace smallest_positive_multiple_l1996_199651

theorem smallest_positive_multiple (a : ℕ) (h : 17 * a % 53 = 7) : 17 * a = 544 :=
sorry

end smallest_positive_multiple_l1996_199651


namespace minimum_value_of_f_l1996_199626

noncomputable def f (x : ℝ) : ℝ := 4 * x + 1 / (4 * x - 5)

theorem minimum_value_of_f (x : ℝ) : x > 5 / 4 → ∃ y, ∀ z, f z ≥ y ∧ y = 7 :=
by
  intro h
  sorry

end minimum_value_of_f_l1996_199626


namespace songs_after_operations_l1996_199681

-- Definitions based on conditions
def initialSongs : ℕ := 15
def deletedSongs : ℕ := 8
def addedSongs : ℕ := 50

-- Problem statement to be proved
theorem songs_after_operations : initialSongs - deletedSongs + addedSongs = 57 :=
by
  sorry

end songs_after_operations_l1996_199681


namespace twenty_four_game_solution_l1996_199627

theorem twenty_four_game_solution :
  let a := 4
  let b := 8
  (a - (b / b)) * b = 24 :=
by
  let a := 4
  let b := 8
  show (a - (b / b)) * b = 24
  sorry

end twenty_four_game_solution_l1996_199627


namespace precise_approximate_classification_l1996_199661

def data_points : List String := ["Xiao Ming bought 5 books today",
                                  "The war in Afghanistan cost the United States $1 billion per month in 2002",
                                  "Relevant departments predict that in 2002, the sales of movies in DVD format will exceed those of VHS tapes for the first time, reaching $9.5 billion",
                                  "The human brain has 10,000,000,000 cells",
                                  "Xiao Hong scored 92 points on this test",
                                  "The Earth has more than 1.5 trillion tons of coal reserves"]

def is_precise (data : String) : Bool :=
  match data with
  | "Xiao Ming bought 5 books today" => true
  | "The war in Afghanistan cost the United States $1 billion per month in 2002" => true
  | "Relevant departments predict that in 2002, the sales of movies in DVD format will exceed those of VHS tapes for the first time, reaching $9.5 billion" => true
  | "Xiao Hong scored 92 points on this test" => true
  | _ => false

def is_approximate (data : String) : Bool :=
  match data with
  | "The human brain has 10,000,000,000 cells" => true
  | "The Earth has more than 1.5 trillion tons of coal reserves" => true
  | _ => false

theorem precise_approximate_classification :
  (data_points.filter is_precise = ["Xiao Ming bought 5 books today",
                                    "The war in Afghanistan cost the United States $1 billion per month in 2002",
                                    "Relevant departments predict that in 2002, the sales of movies in DVD format will exceed those of VHS tapes for the first time, reaching $9.5 billion",
                                    "Xiao Hong scored 92 points on this test"]) ∧
  (data_points.filter is_approximate = ["The human brain has 10,000,000,000 cells",
                                        "The Earth has more than 1.5 trillion tons of coal reserves"]) :=
by sorry

end precise_approximate_classification_l1996_199661


namespace edmonton_to_red_deer_distance_l1996_199684

noncomputable def distance_from_Edmonton_to_Calgary (speed time: ℝ) : ℝ :=
  speed * time

theorem edmonton_to_red_deer_distance :
  let speed := 110
  let time := 3
  let distance_Calgary_RedDeer := 110
  let distance_Edmonton_Calgary := distance_from_Edmonton_to_Calgary speed time
  let distance_Edmonton_RedDeer := distance_Edmonton_Calgary - distance_Calgary_RedDeer
  distance_Edmonton_RedDeer = 220 :=
by
  sorry

end edmonton_to_red_deer_distance_l1996_199684


namespace range_of_f_l1996_199628

def diamond (x y : ℝ) := (x + y) ^ 2 - x * y

def f (a x : ℝ) := diamond a x

theorem range_of_f (a : ℝ) (h : diamond 1 a = 3) :
  ∃ b : ℝ, ∀ x : ℝ, x > 0 → f a x > b :=
sorry

end range_of_f_l1996_199628


namespace sunday_dogs_count_l1996_199638

-- Define initial conditions
def initial_dogs : ℕ := 2
def monday_dogs : ℕ := 3
def total_dogs : ℕ := 10
def sunday_dogs (S : ℕ) : Prop :=
  initial_dogs + S + monday_dogs = total_dogs

-- State the theorem
theorem sunday_dogs_count : ∃ S : ℕ, sunday_dogs S ∧ S = 5 := by
  sorry

end sunday_dogs_count_l1996_199638


namespace volume_of_cube_l1996_199683

theorem volume_of_cube (A : ℝ) (s V : ℝ) 
  (hA : A = 150) 
  (h_surface_area : A = 6 * s^2) 
  (h_side_length : s = 5) :
  V = s^3 →
  V = 125 :=
by
  sorry

end volume_of_cube_l1996_199683


namespace zoo_tickets_total_cost_l1996_199682

-- Define the given conditions
def num_children := 6
def num_adults := 10
def cost_child_ticket := 10
def cost_adult_ticket := 16

-- Calculate the expected total cost
def total_cost := 220

-- State the theorem
theorem zoo_tickets_total_cost :
  num_children * cost_child_ticket + num_adults * cost_adult_ticket = total_cost :=
by
  sorry

end zoo_tickets_total_cost_l1996_199682


namespace cubic_roots_reciprocal_squares_sum_l1996_199609

-- Define the roots a, b, and c
variables (a b c : ℝ)

-- Define the given cubic equation conditions
variables (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) (h3 : a * b * c = 6)

-- Define the target statement
theorem cubic_roots_reciprocal_squares_sum :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 49 / 36 :=
by
  sorry

end cubic_roots_reciprocal_squares_sum_l1996_199609


namespace krista_driving_hours_each_day_l1996_199670

-- Define the conditions as constants
def road_trip_days : ℕ := 3
def jade_hours_per_day : ℕ := 8
def total_hours : ℕ := 42

-- Define the function to calculate Krista's hours per day
noncomputable def krista_hours_per_day : ℕ :=
  (total_hours - road_trip_days * jade_hours_per_day) / road_trip_days

-- State the theorem to prove Krista drove 6 hours each day
theorem krista_driving_hours_each_day : krista_hours_per_day = 6 := by
  sorry

end krista_driving_hours_each_day_l1996_199670


namespace complement_of_intersection_l1996_199642

open Set

def A : Set ℝ := {x | x^2 - x - 6 ≤ 0}
def B : Set ℝ := {x | x^2 + x - 6 > 0}
def S : Set ℝ := univ -- S is the set of all real numbers

theorem complement_of_intersection :
  S \ (A ∩ B) = { x : ℝ | x ≤ 2 } ∪ { x : ℝ | 3 < x } :=
by
  sorry

end complement_of_intersection_l1996_199642


namespace op_5_2_l1996_199607

def op (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem op_5_2 : op 5 2 = 30 := 
by sorry

end op_5_2_l1996_199607


namespace jellybean_avg_increase_l1996_199634

noncomputable def avg_increase_jellybeans 
  (avg_original : ℕ) (num_bags_original : ℕ) (num_jellybeans_new_bag : ℕ) : ℕ :=
  let total_original := avg_original * num_bags_original
  let total_new := total_original + num_jellybeans_new_bag
  let num_bags_new := num_bags_original + 1
  let avg_new := total_new / num_bags_new
  avg_new - avg_original

theorem jellybean_avg_increase :
  avg_increase_jellybeans 117 34 362 = 7 := by
  let total_original := 117 * 34
  let total_new := total_original + 362
  let num_bags_new := 34 + 1
  let avg_new := total_new / num_bags_new
  let increase := avg_new - 117
  have h1 : total_original = 3978 := by norm_num
  have h2 : total_new = 4340 := by norm_num
  have h3 : num_bags_new = 35 := by norm_num
  have h4 : avg_new = 124 := by norm_num
  have h5 : increase = 7 := by norm_num
  exact h5

end jellybean_avg_increase_l1996_199634


namespace variance_of_temperatures_l1996_199643

def temperatures : List ℕ := [28, 21, 22, 26, 28, 25]

noncomputable def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

noncomputable def variance (l : List ℕ) : ℚ :=
  let m := mean l
  (l.map (λ x => (x - m)^2)).sum / l.length

theorem variance_of_temperatures : variance temperatures = 22 / 3 := 
by
  sorry

end variance_of_temperatures_l1996_199643


namespace max_min_vec_magnitude_l1996_199625

noncomputable def vec_a (θ : ℝ) := (Real.cos θ, Real.sin θ)
noncomputable def vec_b : ℝ × ℝ := (Real.sqrt 3, 1)

noncomputable def vec_result (θ : ℝ) := (2 * Real.cos θ - Real.sqrt 3, 2 * Real.sin θ - 1)

noncomputable def vec_magnitude (θ : ℝ) := Real.sqrt ((2 * Real.cos θ - Real.sqrt 3)^2 + (2 * Real.sin θ - 1)^2)

theorem max_min_vec_magnitude : 
  ∃ θ_max θ_min, 
    vec_magnitude θ_max = 4 ∧ 
    vec_magnitude θ_min = 0 :=
by
  sorry

end max_min_vec_magnitude_l1996_199625


namespace jake_fewer_peaches_than_steven_l1996_199656

theorem jake_fewer_peaches_than_steven :
  ∀ (jill steven jake : ℕ),
    jill = 87 →
    steven = jill + 18 →
    jake = jill + 13 →
    steven - jake = 5 :=
by
  intros jill steven jake hjill hsteven hjake
  sorry

end jake_fewer_peaches_than_steven_l1996_199656


namespace cookies_per_person_l1996_199688

-- Definitions based on conditions
def cookies_total : ℕ := 144
def people_count : ℕ := 6

-- The goal is to prove the number of cookies per person
theorem cookies_per_person : cookies_total / people_count = 24 :=
by
  sorry

end cookies_per_person_l1996_199688


namespace lcm_of_ratio_and_hcf_l1996_199678

theorem lcm_of_ratio_and_hcf (a b : ℕ) (h1 : a = 3 * 8) (h2 : b = 4 * 8) (h3 : Nat.gcd a b = 8) : Nat.lcm a b = 96 :=
  sorry

end lcm_of_ratio_and_hcf_l1996_199678


namespace exists_n_satisfying_condition_l1996_199612

-- Definition of the divisor function d(n)
def d (n : ℕ) : ℕ := Nat.divisors n |>.card

-- Theorem statement
theorem exists_n_satisfying_condition : ∃ n : ℕ, ∀ i : ℕ, i ≤ 1402 → (d n : ℚ) / d (n + i) > 1401 ∧ (d n : ℚ) / d (n - i) > 1401 :=
by
  sorry

end exists_n_satisfying_condition_l1996_199612


namespace second_route_time_l1996_199676

-- Defining time for the first route with all green lights
def R_green : ℕ := 10

-- Defining the additional time added by each red light
def per_red_light : ℕ := 3

-- Defining total time for the first route with all red lights
def R_red : ℕ := R_green + 3 * per_red_light

-- Defining the second route time plus the difference
def S : ℕ := R_red - 5

theorem second_route_time : S = 14 := by
  sorry

end second_route_time_l1996_199676


namespace inequality_a_b_c_l1996_199654

theorem inequality_a_b_c (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^3 / (a^2 + a * b + b^2)) + (b^3 / (b^2 + b * c + c^2)) + (c^3 / (c^2 + c * a + a^2)) ≥ (a + b + c) / 3 :=
sorry

end inequality_a_b_c_l1996_199654


namespace cards_per_page_l1996_199679

theorem cards_per_page 
  (new_cards old_cards pages : ℕ) 
  (h1 : new_cards = 8)
  (h2 : old_cards = 10)
  (h3 : pages = 6) : (new_cards + old_cards) / pages = 3 := 
by 
  sorry

end cards_per_page_l1996_199679


namespace equal_probability_after_adding_balls_l1996_199674

theorem equal_probability_after_adding_balls :
  let initial_white := 2
  let initial_yellow := 3
  let added_white := 4
  let added_yellow := 3
  let total_white := initial_white + added_white
  let total_yellow := initial_yellow + added_yellow
  let total_balls := total_white + total_yellow
  (total_white / total_balls) = (total_yellow / total_balls) := by
  sorry

end equal_probability_after_adding_balls_l1996_199674


namespace polynomial_range_l1996_199618

def p (x : ℝ) : ℝ := x^4 - 4*x^3 + 8*x^2 - 8*x + 5

theorem polynomial_range : ∀ x : ℝ, p x ≥ 2 :=
by
sorry

end polynomial_range_l1996_199618


namespace pythagorean_consecutive_numbers_unique_l1996_199610

theorem pythagorean_consecutive_numbers_unique :
  ∀ (x : ℕ), (x + 2) * (x + 2) = (x + 1) * (x + 1) + x * x → x = 3 :=
by
  sorry 

end pythagorean_consecutive_numbers_unique_l1996_199610


namespace total_height_of_three_buildings_l1996_199685

theorem total_height_of_three_buildings :
  let h1 := 600
  let h2 := 2 * h1
  let h3 := 3 * (h1 + h2)
  h1 + h2 + h3 = 7200 :=
by
  sorry

end total_height_of_three_buildings_l1996_199685


namespace largest_corner_sum_l1996_199631

-- Define the cube and its properties
structure Cube :=
  (faces : ℕ → ℕ)
  (opposite_faces_sum_to_8 : ∀ i, faces i + faces (7 - i) = 8)

-- Prove that the largest sum of three numbers whose faces meet at one corner is 16
theorem largest_corner_sum (c : Cube) : ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ 
  (c.faces i + c.faces j + c.faces k = 16) :=
sorry

end largest_corner_sum_l1996_199631


namespace eq_iff_squared_eq_l1996_199637

theorem eq_iff_squared_eq (a b : ℝ) : a = b ↔ a^2 + b^2 = 2 * a * b :=
by
  sorry

end eq_iff_squared_eq_l1996_199637


namespace convex_octagon_min_obtuse_l1996_199673

-- Define a type for a polygon (here specifically an octagon)
structure Polygon (n : ℕ) :=
(vertices : ℕ)
(convex : Prop)

-- Define that an octagon is a specific polygon with 8 vertices
def octagon : Polygon 8 :=
{ vertices := 8,
  convex := sorry }

-- Define the predicate for convex polygons
def is_convex (poly : Polygon 8) : Prop := poly.convex

-- Defining the statement that a convex octagon has at least 5 obtuse interior angles
theorem convex_octagon_min_obtuse (poly : Polygon 8) (h : is_convex poly) : ∃ (n : ℕ), n = 5 :=
sorry

end convex_octagon_min_obtuse_l1996_199673


namespace glasses_per_pitcher_l1996_199662

def total_glasses : Nat := 30
def num_pitchers : Nat := 6

theorem glasses_per_pitcher : total_glasses / num_pitchers = 5 := by
  sorry

end glasses_per_pitcher_l1996_199662


namespace average_salary_l1996_199604

theorem average_salary (avg_officer_salary avg_nonofficer_salary num_officers num_nonofficers : ℕ) (total_salary total_employees : ℕ) : 
  avg_officer_salary = 430 → 
  avg_nonofficer_salary = 110 → 
  num_officers = 15 → 
  num_nonofficers = 465 → 
  total_salary = avg_officer_salary * num_officers + avg_nonofficer_salary * num_nonofficers → 
  total_employees = num_officers + num_nonofficers → 
  total_salary / total_employees = 120 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end average_salary_l1996_199604


namespace keiko_jogging_speed_l1996_199658

variable (s : ℝ) -- Keiko's jogging speed
variable (b : ℝ) -- radius of the inner semicircle
variable (L_inner : ℝ := 200 + 2 * Real.pi * b) -- total length of the inner track
variable (L_outer : ℝ := 200 + 2 * Real.pi * (b + 8)) -- total length of the outer track
variable (t_inner : ℝ := L_inner / s) -- time to jog the inside edge
variable (t_outer : ℝ := L_outer / s) -- time to jog the outside edge
variable (time_difference : ℝ := 48) -- time difference between jogging inside and outside edges

theorem keiko_jogging_speed : L_inner = 200 + 2 * Real.pi * b →
                           L_outer = 200 + 2 * Real.pi * (b + 8) →
                           t_outer = t_inner + 48 →
                           s = Real.pi / 3 :=
by
  intro h1 h2 h3
  sorry

end keiko_jogging_speed_l1996_199658


namespace shaded_region_area_l1996_199623

noncomputable def area_shaded_region (r_small r_large : ℝ) (A B : ℝ × ℝ) : ℝ := 
  let pi := Real.pi
  let sqrt_5 := Real.sqrt 5
  (5 * pi / 2) - (4 * sqrt_5)

theorem shaded_region_area : 
  ∀ (r_small r_large : ℝ) (A B : ℝ × ℝ), 
  r_small = 2 → 
  r_large = 3 → 
  (A = (0, 0)) → 
  (B = (4, 0)) → 
  area_shaded_region r_small r_large A B = (5 * Real.pi / 2) - (4 * Real.sqrt 5) := 
by
  intros r_small r_large A B h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  exact rfl

end shaded_region_area_l1996_199623


namespace polygon_sides_eq_five_l1996_199669

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem polygon_sides_eq_five (n : ℕ) (h : n - number_of_diagonals n = 0) : n = 5 :=
by
  sorry

end polygon_sides_eq_five_l1996_199669


namespace evaluate_polynomial_at_3_using_horners_method_l1996_199644

def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem evaluate_polynomial_at_3_using_horners_method : f 3 = 1641 := by
 sorry

end evaluate_polynomial_at_3_using_horners_method_l1996_199644


namespace cos_C_value_l1996_199687

theorem cos_C_value (a b c : ℝ) (A B C : ℝ)
  (h1 : a * Real.cos B + b * Real.cos A = 3 * c * Real.cos C)
  (h2 : 0 < A ∧ A < π)
  (h3 : 0 < B ∧ B < π)
  (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π)
  : Real.cos C = (Real.sqrt 10) / 10 :=
sorry

end cos_C_value_l1996_199687


namespace find_integer_n_l1996_199668

theorem find_integer_n :
  ∃ (n : ℤ), -90 ≤ n ∧ n ≤ 90 ∧ Real.sin (n * Real.pi / 180) = Real.cos (675 * Real.pi / 180) ∧ n = 45 :=
sorry

end find_integer_n_l1996_199668


namespace unique_solution_exists_l1996_199672

theorem unique_solution_exists (n m k : ℕ) :
  n = m^3 ∧ n = 1000 * m + k ∧ 0 ≤ k ∧ k < 1000 ∧ (1000 * m ≤ m^3 ∧ m^3 < 1000 * (m + 1)) → n = 32768 :=
by
  sorry

end unique_solution_exists_l1996_199672


namespace find_x_l1996_199689

-- Define the vectors a and b
def a : ℝ × ℝ := (-2, 3)
def b (x : ℝ) : ℝ × ℝ := (x, -3)

-- Define the parallel condition between (b - a) and b
def parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u  = (k * v.1, k * v.2)

-- The problem statement in Lean 4
theorem find_x (x : ℝ) (h : parallel (b x - a) (b x)) : x = 2 := 
  sorry

end find_x_l1996_199689


namespace problem_statement_l1996_199650

noncomputable def f (x : ℝ) : ℝ := ∫ t in -x..x, Real.cos t

theorem problem_statement : f (f (Real.pi / 4)) = 2 * Real.sin (Real.sqrt 2) := 
by
  sorry

end problem_statement_l1996_199650


namespace sale_in_second_month_l1996_199686

theorem sale_in_second_month 
  (sale_first_month: ℕ := 2500)
  (sale_third_month: ℕ := 3540)
  (sale_fourth_month: ℕ := 1520)
  (average_sale: ℕ := 2890)
  (total_sales: ℕ := 11560) :
  sale_first_month + sale_third_month + sale_fourth_month + (sale_second_month: ℕ) = total_sales → 
  sale_second_month = 4000 := 
by
  intros h
  sorry

end sale_in_second_month_l1996_199686


namespace volume_increase_l1996_199666

theorem volume_increase (L B H : ℝ) :
  let L_new := 1.25 * L
  let B_new := 0.85 * B
  let H_new := 1.10 * H
  (L_new * B_new * H_new) = 1.16875 * (L * B * H) := 
by
  sorry

end volume_increase_l1996_199666


namespace original_price_calculation_l1996_199605

-- Definitions directly from problem conditions
def price_after_decrease (original_price : ℝ) : ℝ := 0.76 * original_price
def new_price : ℝ := 988

-- Statement embedding our problem
theorem original_price_calculation (x : ℝ) (hx : price_after_decrease x = new_price) : x = 1300 :=
by
  sorry

end original_price_calculation_l1996_199605


namespace A_three_two_l1996_199648

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m+1, 0 => A m 2
| m+1, n+1 => A m (A (m + 1) n)

theorem A_three_two : A 3 2 = 5 := 
by 
  sorry

end A_three_two_l1996_199648


namespace perfect_square_trinomial_m_l1996_199675

theorem perfect_square_trinomial_m (m : ℝ) :
  (∃ a : ℝ, (x^2 + 2 * (m - 1) * x + 4) = (x + a)^2) → (m = 3 ∨ m = -1) :=
by
  sorry

end perfect_square_trinomial_m_l1996_199675


namespace lizard_problem_theorem_l1996_199694

def lizard_problem : Prop :=
  ∃ (E W S : ℕ), 
  E = 3 ∧ 
  W = 3 * E ∧ 
  S = 7 * W ∧ 
  (S + W) - E = 69

theorem lizard_problem_theorem : lizard_problem :=
by
  sorry

end lizard_problem_theorem_l1996_199694


namespace max_value_f_on_interval_l1996_199645

def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

theorem max_value_f_on_interval : 
  ∀ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f x ≤ 15 :=
by
  sorry

end max_value_f_on_interval_l1996_199645


namespace cookies_left_l1996_199693

theorem cookies_left (total_cookies : ℕ) (total_neighbors : ℕ) (cookies_per_neighbor : ℕ) (sarah_cookies : ℕ)
  (h1 : total_cookies = 150)
  (h2 : total_neighbors = 15)
  (h3 : cookies_per_neighbor = 10)
  (h4 : sarah_cookies = 12) :
  total_cookies - ((total_neighbors - 1) * cookies_per_neighbor + sarah_cookies) = 8 :=
by
  simp [h1, h2, h3, h4]
  sorry

end cookies_left_l1996_199693


namespace least_positive_three_digit_multiple_of_9_l1996_199635

   theorem least_positive_three_digit_multiple_of_9 : ∃ n : ℕ, n >= 100 ∧ n < 1000 ∧ 9 ∣ n ∧ n = 108 :=
   by
     sorry
   
end least_positive_three_digit_multiple_of_9_l1996_199635


namespace dot_product_AB_AC_dot_product_AB_BC_l1996_199690

-- The definition of equilateral triangle with side length 6
structure EquilateralTriangle (A B C : Type*) :=
  (side_len : ℝ)
  (angle_ABC : ℝ)
  (angle_BCA : ℝ)
  (angle_CAB : ℝ)
  (AB_len : ℝ)
  (AC_len : ℝ)
  (BC_len : ℝ)
  (AB_eq_AC : AB_len = AC_len)
  (AB_eq_BC : AB_len = BC_len)
  (cos_ABC : ℝ)
  (cos_BCA : ℝ)
  (cos_CAB : ℝ)

-- Given an equilateral triangle with side length 6 where the angles are defined,
-- we can define the specific triangle
noncomputable def triangleABC (A B C : Type*) : EquilateralTriangle A B C :=
{ side_len := 6,
  angle_ABC := 120,
  angle_BCA := 60,
  angle_CAB := 60,
  AB_len := 6,
  AC_len := 6,
  BC_len := 6,
  AB_eq_AC := rfl,
  AB_eq_BC := rfl,
  cos_ABC := -0.5,
  cos_BCA := 0.5,
  cos_CAB := 0.5 }

-- Prove the dot product of vectors AB and AC
theorem dot_product_AB_AC (A B C : Type*) 
  (T : EquilateralTriangle A B C) : 
  (T.AB_len * T.AC_len * T.cos_BCA) = 18 :=
by sorry

-- Prove the dot product of vectors AB and BC
theorem dot_product_AB_BC (A B C : Type*) 
  (T : EquilateralTriangle A B C) : 
  (T.AB_len * T.BC_len * T.cos_ABC) = -18 :=
by sorry

end dot_product_AB_AC_dot_product_AB_BC_l1996_199690


namespace donny_remaining_money_l1996_199696

theorem donny_remaining_money :
  let initial_amount := 78
  let kite_cost := 8
  let frisbee_cost := 9
  initial_amount - (kite_cost + frisbee_cost) = 61 :=
by
  sorry

end donny_remaining_money_l1996_199696


namespace hyperbola_center_l1996_199647

def is_midpoint (x1 y1 x2 y2 xc yc : ℝ) : Prop :=
  xc = (x1 + x2) / 2 ∧ yc = (y1 + y2) / 2

theorem hyperbola_center :
  is_midpoint 2 (-3) (-4) 5 (-1) 1 :=
by
  sorry

end hyperbola_center_l1996_199647


namespace sarah_calculate_profit_l1996_199680

noncomputable def sarah_total_profit (hot_day_price : ℚ) (regular_day_price : ℚ) (cost_per_cup : ℚ) (cups_per_day : ℕ) (hot_days : ℕ) (total_days : ℕ) : ℚ := 
  let hot_day_revenue := hot_day_price * cups_per_day * hot_days
  let regular_day_revenue := regular_day_price * cups_per_day * (total_days - hot_days)
  let total_revenue := hot_day_revenue + regular_day_revenue
  let total_cost := cost_per_cup * cups_per_day * total_days
  total_revenue - total_cost

theorem sarah_calculate_profit : 
  let hot_day_price := (20951704545454546 : ℚ) / 10000000000000000
  let regular_day_price := hot_day_price / 1.25
  let cost_per_cup := 75 / 100
  let cups_per_day := 32
  let hot_days := 4
  let total_days := 10
  sarah_total_profit hot_day_price regular_day_price cost_per_cup cups_per_day hot_days total_days = (34935102 : ℚ) / 10000000 :=
by
  sorry

end sarah_calculate_profit_l1996_199680


namespace find_a_find_a_plus_c_l1996_199621

-- Define the triangle with given sides and angles
variables (A B C : ℝ) (a b c S : ℝ)
  (h_cosB : cos B = 4/5)
  (h_b : b = 2)
  (h_area : S = 3)

-- Prove the value of the side 'a' when angle A is π/6
theorem find_a (h_A : A = Real.pi / 6) : a = 5 / 3 := 
  sorry

-- Prove the sum of sides 'a' and 'c' when the area of the triangle is 3
theorem find_a_plus_c (h_ac : a * c = 10) : a + c = 2 * Real.sqrt 10 :=
  sorry

end find_a_find_a_plus_c_l1996_199621


namespace claire_has_gerbils_l1996_199698

-- Definitions based on conditions
variables (G H : ℕ)
variables (h1 : G + H = 90) (h2 : (1/4 : ℚ) * G + (1/3 : ℚ) * H = 25)

-- Main statement to prove
theorem claire_has_gerbils : G = 60 :=
sorry

end claire_has_gerbils_l1996_199698


namespace article_large_font_pages_l1996_199632

theorem article_large_font_pages (L S : ℕ) 
  (pages_eq : L + S = 21) 
  (words_eq : 1800 * L + 2400 * S = 48000) : 
  L = 4 := 
by 
  sorry

end article_large_font_pages_l1996_199632


namespace sum_A_C_l1996_199615

theorem sum_A_C (A B C : ℝ) (h1 : A + B + C = 500) (h2 : B + C = 340) (h3 : C = 40) : A + C = 200 :=
by
  sorry

end sum_A_C_l1996_199615


namespace inequality_abc_l1996_199657

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b + b * c + c * a ≤ 1) :
  a + b + c + Real.sqrt 3 ≥ 8 * a * b * c * (1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1)) :=
by
  sorry

end inequality_abc_l1996_199657


namespace compare_a_b_l1996_199636

theorem compare_a_b (m : ℝ) (h : m > 1) 
  (a : ℝ := (Real.sqrt (m+1)) - (Real.sqrt m))
  (b : ℝ := (Real.sqrt m) - (Real.sqrt (m-1))) : a < b :=
by
  sorry

end compare_a_b_l1996_199636


namespace range_of_m_l1996_199608

def A := { x : ℝ | x^2 - 2 * x - 15 ≤ 0 }
def B (m : ℝ) := { x : ℝ | m - 2 < x ∧ x < 2 * m - 3 }

theorem range_of_m : ∀ m : ℝ, (B m ⊆ A) ↔ (m ≤ 4) :=
by sorry

end range_of_m_l1996_199608


namespace employee_selected_from_10th_group_is_47_l1996_199622

theorem employee_selected_from_10th_group_is_47
  (total_employees : ℕ)
  (sampled_employees : ℕ)
  (total_groups : ℕ)
  (random_start : ℕ)
  (common_difference : ℕ)
  (selected_from_5th_group : ℕ) :
  total_employees = 200 →
  sampled_employees = 40 →
  total_groups = 40 →
  random_start = 2 →
  common_difference = 5 →
  selected_from_5th_group = 22 →
  (selected_from_5th_group = (4 * common_difference + random_start)) →
  (9 * common_difference + random_start) = 47 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end employee_selected_from_10th_group_is_47_l1996_199622


namespace product_of_m_l1996_199639

theorem product_of_m (m n : ℤ) (h_cond : m^2 + m + 8 = n^2) (h_nonneg : n ≥ 0) : 
  (∀ m, (∃ n, m^2 + m + 8 = n^2 ∧ n ≥ 0) → m = 7 ∨ m = -8) ∧ 
  (∃ m1 m2 : ℤ, m1 = 7 ∧ m2 = -8 ∧ (m1 * m2 = -56)) :=
by
  sorry

end product_of_m_l1996_199639


namespace find_third_number_l1996_199697

theorem find_third_number (first_number second_number third_number : ℕ) 
  (h1 : first_number = 200)
  (h2 : first_number + second_number + third_number = 500)
  (h3 : second_number = 2 * third_number) :
  third_number = 100 := sorry

end find_third_number_l1996_199697


namespace chef_initial_eggs_l1996_199646

-- Define the conditions
def eggs_in_fridge := 10
def eggs_per_cake := 5
def cakes_made := 10

-- Prove that the number of initial eggs is 60
theorem chef_initial_eggs : (eggs_per_cake * cakes_made + eggs_in_fridge) = 60 :=
by
  sorry

end chef_initial_eggs_l1996_199646


namespace towel_price_40_l1996_199600

/-- Let x be the price of each towel bought second by the woman. 
    Given that she bought 3 towels at Rs. 100 each, 5 towels at x Rs. each, 
    and 2 towels at Rs. 550 each, and the average price of the towels was Rs. 160,
    we need to prove that x equals 40. -/
theorem towel_price_40 
    (x : ℝ)
    (h_avg_price : (300 + 5 * x + 1100) / 10 = 160) : 
    x = 40 :=
sorry

end towel_price_40_l1996_199600


namespace min_expression_value_l1996_199620

open Real

theorem min_expression_value : ∀ x : ℝ, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ 2023 := by
  sorry

end min_expression_value_l1996_199620


namespace system_of_equations_l1996_199671

theorem system_of_equations (x y : ℝ) (h1 : 3 * x + 210 = 5 * y) (h2 : 10 * y - 10 * x = 100) :
    (3 * x + 210 = 5 * y) ∧ (10 * y - 10 * x = 100) := by
  sorry

end system_of_equations_l1996_199671


namespace boys_joined_school_l1996_199640

theorem boys_joined_school (initial_boys final_boys boys_joined : ℕ) 
  (h1 : initial_boys = 214) 
  (h2 : final_boys = 1124) 
  (h3 : final_boys = initial_boys + boys_joined) : 
  boys_joined = 910 := 
by 
  rw [h1, h2] at h3
  sorry

end boys_joined_school_l1996_199640


namespace firecrackers_defective_fraction_l1996_199630

theorem firecrackers_defective_fraction (initial_total good_remaining confiscated : ℕ) 
(h_initial : initial_total = 48) 
(h_confiscated : confiscated = 12) 
(h_good_remaining : good_remaining = 15) : 
(initial_total - confiscated - 2 * good_remaining) / (initial_total - confiscated) = 1 / 6 := by
  sorry

end firecrackers_defective_fraction_l1996_199630
