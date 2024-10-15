import Mathlib

namespace NUMINAMATH_GPT_top_three_probability_l336_33666

-- Definitions for the real-world problem
def total_ways_to_choose_three_cards : ℕ :=
  52 * 51 * 50

def favorable_ways_to_choose_three_specific_suits : ℕ :=
  13 * 13 * 13 * 6

def probability_top_three_inclusive (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

-- The mathematically equivalent proof problem's Lean statement
theorem top_three_probability:
  probability_top_three_inclusive total_ways_to_choose_three_cards favorable_ways_to_choose_three_specific_suits = 2197 / 22100 :=
by
  sorry

end NUMINAMATH_GPT_top_three_probability_l336_33666


namespace NUMINAMATH_GPT_minimum_value_of_f_l336_33659

noncomputable def f (x : ℝ) : ℝ := (1 / (Real.cos x)^2) + (1 / (Real.sin x)^2)

theorem minimum_value_of_f : ∀ x : ℝ, ∃ y : ℝ, y = f x ∧ y = 4 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l336_33659


namespace NUMINAMATH_GPT_find_a_l336_33601

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a (a : ℝ) 
  (h : (6 * a * (-1) + 6) = 4) : 
  a = 10 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_l336_33601


namespace NUMINAMATH_GPT_medication_price_reduction_l336_33637

variable (a : ℝ)

theorem medication_price_reduction (h : 0.60 * x = a) : x = 5/3 * a := by
  sorry

end NUMINAMATH_GPT_medication_price_reduction_l336_33637


namespace NUMINAMATH_GPT_find_number_l336_33653

theorem find_number (x : ℝ) (h : 0.05 * x = 12.75) : x = 255 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l336_33653


namespace NUMINAMATH_GPT_slice_of_bread_area_l336_33610

theorem slice_of_bread_area (total_area : ℝ) (number_of_parts : ℕ) (h1 : total_area = 59.6) (h2 : number_of_parts = 4) : 
  total_area / number_of_parts = 14.9 :=
by
  rw [h1, h2]
  norm_num


end NUMINAMATH_GPT_slice_of_bread_area_l336_33610


namespace NUMINAMATH_GPT_find_k_l336_33618

def vec2 := ℝ × ℝ

-- Definitions
def i : vec2 := (1, 0)
def j : vec2 := (0, 1)
def a : vec2 := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)
def b (k : ℝ) : vec2 := (k * i.1 - 4 * j.1, k * i.2 - 4 * j.2)

-- Dot product definition for 2D vectors
def dot_product (u v : vec2) : ℝ := u.1 * v.1 + u.2 * v.2

-- Theorem
theorem find_k (k : ℝ) : dot_product a (b k) = 0 → k = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l336_33618


namespace NUMINAMATH_GPT_average_of_data_set_is_five_l336_33606

def data_set : List ℕ := [2, 5, 5, 6, 7]

def sum_of_data_set : ℕ := data_set.sum
def count_of_data_set : ℕ := data_set.length

theorem average_of_data_set_is_five :
  (sum_of_data_set / count_of_data_set) = 5 :=
by
  sorry

end NUMINAMATH_GPT_average_of_data_set_is_five_l336_33606


namespace NUMINAMATH_GPT_fewer_VIP_tickets_sold_l336_33615

variable (V G : ℕ)

-- Definitions: total number of tickets sold and the total revenue from tickets sold
def total_tickets : Prop := V + G = 320
def total_revenue : Prop := 45 * V + 20 * G = 7500

-- Definition of the number of fewer VIP tickets than general admission tickets
def fewer_VIP_tickets : Prop := G - V = 232

-- The theorem to be proven
theorem fewer_VIP_tickets_sold (h1 : total_tickets V G) (h2 : total_revenue V G) : fewer_VIP_tickets V G :=
sorry

end NUMINAMATH_GPT_fewer_VIP_tickets_sold_l336_33615


namespace NUMINAMATH_GPT_perpendicular_condition_l336_33656

theorem perpendicular_condition (a : ℝ) :
  (a = 1) ↔ (∀ x : ℝ, (a*x + 1 - ((a - 2)*x - 1)) * ((a * x + 1 - (a * x + 1))) = 0) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_condition_l336_33656


namespace NUMINAMATH_GPT_min_value_f_min_value_f_sqrt_min_value_f_2_min_m_l336_33691

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (1 / 2) * x^2 - a * Real.log x + b

theorem min_value_f 
  (a b : ℝ) 
  (a_non_pos : a ≤ 1) : 
  f 1 a b = (1 / 2) + b :=
sorry

theorem min_value_f_sqrt 
  (a b : ℝ) 
  (a_pos_range : 1 < a ∧ a < 4) : 
  f (Real.sqrt a) a b = (a / 2) - a * Real.log (Real.sqrt a) + b :=
sorry

theorem min_value_f_2 
  (a b : ℝ) 
  (a_ge_4 : 4 ≤ a) : 
  f 2 a b = 2 - a * Real.log 2 + b :=
sorry

theorem min_m 
  (a : ℝ) 
  (a_range : -2 ≤ a ∧ a < 0):
  ∀x1 x2 : ℝ, (0 < x1 ∧ x1 ≤ 2) ∧ (0 < x2 ∧ x2 ≤ 2) →
  ∃m : ℝ, m = 12 ∧ abs (f x1 a 0 - f x2 a 0) ≤ m ^ abs (1 / x1 - 1 / x2) :=
sorry

end NUMINAMATH_GPT_min_value_f_min_value_f_sqrt_min_value_f_2_min_m_l336_33691


namespace NUMINAMATH_GPT_henry_change_l336_33608

theorem henry_change (n : ℕ) (p m : ℝ) (h_n : n = 4) (h_p : p = 0.75) (h_m : m = 10) : 
  m - (n * p) = 7 := 
by 
  sorry

end NUMINAMATH_GPT_henry_change_l336_33608


namespace NUMINAMATH_GPT_geometric_arithmetic_sequences_sum_l336_33660

theorem geometric_arithmetic_sequences_sum (a b : ℕ → ℝ) (S_n : ℕ → ℝ) 
  (q d : ℝ) (h1 : 0 < q) 
  (h2 : a 1 = 1) (h3 : b 1 = 1) 
  (h4 : a 5 + b 3 = 21) 
  (h5 : a 3 + b 5 = 13) :
  (∀ n, a n = 2^(n-1)) ∧ (∀ n, b n = 2*n - 1) ∧ (∀ n, S_n n = 3 - (2*n + 3)/(2^n)) := 
sorry

end NUMINAMATH_GPT_geometric_arithmetic_sequences_sum_l336_33660


namespace NUMINAMATH_GPT_find_julios_bonus_l336_33668

def commission (customers: ℕ) : ℕ :=
  customers * 1

def total_commission (week1: ℕ) (week2: ℕ) (week3: ℕ) : ℕ :=
  commission week1 + commission week2 + commission week3

noncomputable def julios_bonus (total_earnings salary total_commission: ℕ) : ℕ :=
  total_earnings - salary - total_commission

theorem find_julios_bonus :
  let week1 := 35
  let week2 := 2 * week1
  let week3 := 3 * week1
  let salary := 500
  let total_earnings := 760
  let total_comm := total_commission week1 week2 week3
  julios_bonus total_earnings salary total_comm = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_julios_bonus_l336_33668


namespace NUMINAMATH_GPT_find_value_l336_33695

def set_condition (s : Set ℕ) : Prop := s = {0, 1, 2}

def one_relationship_correct (a b c : ℕ) : Prop :=
  (a ≠ 2 ∧ b ≠ 2 ∧ c = 0)
  ∨ (a ≠ 2 ∧ b = 2 ∧ c = 0)
  ∨ (a ≠ 2 ∧ b = 2 ∧ c ≠ 0)
  ∨ (a = 2 ∧ b ≠ 2 ∧ c ≠ 0)
  ∨ (a = 2 ∧ b = 2 ∧ c = 0)
  ∨ (a = 2 ∧ b ≠ 2 ∧ c = 0)
  ∨ (a ≠ 2 ∧ b ≠ 2 ∧ c ≠ 0)
  ∨ (a = 2 ∧ b ≠ 2 ∧ c = 1)
  ∨ (a = 2 ∧ b = 0 ∧ c = 1)
  ∨ (a = 2 ∧ b = 0 ∧ c ≠ 0)
  ∨ (a ≠ 2 ∧ b = 0 ∧ c ≠ 0)

theorem find_value (a b c : ℕ) (h1 : set_condition {a, b, c}) (h2 : one_relationship_correct a b c) :
  100 * c + 10 * b + a = 102 :=
sorry

end NUMINAMATH_GPT_find_value_l336_33695


namespace NUMINAMATH_GPT_C_investment_l336_33696

theorem C_investment (A B C_profit total_profit : ℝ) (hA : A = 24000) (hB : B = 32000) (hC_profit : C_profit = 36000) (h_total_profit : total_profit = 92000) (x : ℝ) (h : x / (A + B + x) = C_profit / total_profit) : x = 36000 := 
by
  sorry

end NUMINAMATH_GPT_C_investment_l336_33696


namespace NUMINAMATH_GPT_find_num_of_boys_l336_33622

-- Define the constants for number of girls and total number of kids
def num_of_girls : ℕ := 3
def total_kids : ℕ := 9

-- The theorem stating the number of boys based on the given conditions
theorem find_num_of_boys (g t : ℕ) (h1 : g = num_of_girls) (h2 : t = total_kids) :
  t - g = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_num_of_boys_l336_33622


namespace NUMINAMATH_GPT_double_seven_eighth_l336_33673

theorem double_seven_eighth (n : ℕ) (h : n = 48) : 2 * (7 / 8 * n) = 84 := by
  sorry

end NUMINAMATH_GPT_double_seven_eighth_l336_33673


namespace NUMINAMATH_GPT_wall_building_time_l336_33680

theorem wall_building_time (m1 m2 d1 d2 k : ℕ) (h1 : m1 = 12) (h2 : d1 = 6) (h3 : m2 = 18) (h4 : k = 72) 
  (condition : m1 * d1 = k) (rate_const : m2 * d2 = k) : d2 = 4 := by
  sorry

end NUMINAMATH_GPT_wall_building_time_l336_33680


namespace NUMINAMATH_GPT_tegwen_family_total_children_l336_33611

variable (Tegwen : Type)

-- Variables representing the number of girls and boys
variable (g b : ℕ)

-- Conditions from the problem
variable (h1 : b = g - 1)
variable (h2 : g = (3/2:ℚ) * (b - 1))

-- Proposition that the total number of children is 11
theorem tegwen_family_total_children : g + b = 11 := by
  sorry

end NUMINAMATH_GPT_tegwen_family_total_children_l336_33611


namespace NUMINAMATH_GPT_boat_length_in_steps_l336_33632

theorem boat_length_in_steps (L E S : ℝ) 
  (h1 : 250 * E = L + 250 * S) 
  (h2 : 50 * E = L - 50 * S) :
  L = 83 * E :=
by sorry

end NUMINAMATH_GPT_boat_length_in_steps_l336_33632


namespace NUMINAMATH_GPT_johns_daily_calorie_intake_l336_33607

variable (breakfast lunch dinner shake : ℕ)
variable (num_shakes meals_per_day : ℕ)
variable (lunch_inc : ℕ)
variable (dinner_mult : ℕ)

-- Define the conditions from the problem
def john_calories_per_day 
  (breakfast := 500)
  (lunch := breakfast + lunch_inc)
  (dinner := lunch * dinner_mult)
  (shake := 300)
  (num_shakes := 3)
  (lunch_inc := breakfast / 4)
  (dinner_mult := 2)
  : ℕ :=
  breakfast + lunch + dinner + (shake * num_shakes)

theorem johns_daily_calorie_intake : john_calories_per_day = 3275 := by
  sorry

end NUMINAMATH_GPT_johns_daily_calorie_intake_l336_33607


namespace NUMINAMATH_GPT_q_minus_r_max_value_l336_33621

theorem q_minus_r_max_value :
  ∃ (q r : ℕ), q > 99 ∧ q < 1000 ∧ r > 99 ∧ r < 1000 ∧ 
    q = 100 * (q / 100) + 10 * ((q / 10) % 10) + (q % 10) ∧ 
    r = 100 * (q % 10) + 10 * ((q / 10) % 10) + (q / 100) ∧ 
    q - r = 297 :=
by sorry

end NUMINAMATH_GPT_q_minus_r_max_value_l336_33621


namespace NUMINAMATH_GPT_xiao_hong_home_to_school_distance_l336_33675

-- Definition of conditions
def distance_from_drop_to_school := 1000 -- in meters
def time_from_home_to_school_walking := 22.5 -- in minutes
def time_from_home_to_school_biking := 40 -- in minutes
def walking_speed := 80 -- in meters per minute
def bike_speed_slowdown := 800 -- in meters per minute

-- The main theorem statement
theorem xiao_hong_home_to_school_distance :
  ∃ d : ℝ, d = 12000 ∧ 
            distance_from_drop_to_school = 1000 ∧
            time_from_home_to_school_walking = 22.5 ∧
            time_from_home_to_school_biking = 40 ∧
            walking_speed = 80 ∧
            bike_speed_slowdown = 800 := 
sorry

end NUMINAMATH_GPT_xiao_hong_home_to_school_distance_l336_33675


namespace NUMINAMATH_GPT_sara_initial_pears_l336_33682

theorem sara_initial_pears (given_to_dan : ℕ) (left_with_sara : ℕ) (total : ℕ) :
  given_to_dan = 28 ∧ left_with_sara = 7 ∧ total = given_to_dan + left_with_sara → total = 35 :=
by
  sorry

end NUMINAMATH_GPT_sara_initial_pears_l336_33682


namespace NUMINAMATH_GPT_pages_read_per_hour_l336_33670

theorem pages_read_per_hour (lunch_time : ℕ) (book_pages : ℕ) (round_trip_time : ℕ)
  (h1 : lunch_time = round_trip_time)
  (h2 : book_pages = 4000)
  (h3 : round_trip_time = 2 * 4) :
  book_pages / (2 * lunch_time) = 250 :=
by
  sorry

end NUMINAMATH_GPT_pages_read_per_hour_l336_33670


namespace NUMINAMATH_GPT_range_of_t_l336_33625

noncomputable def a_n (t : ℝ) (n : ℕ) : ℝ :=
  if n > 8 then ((1 / 3) - t) * (n:ℝ) + 2 else t ^ (n - 7)

theorem range_of_t (t : ℝ) :
  (∀ (n : ℕ), n ≠ 0 → a_n t n > a_n t (n + 1)) →
  (1/2 < t ∧ t < 1) :=
by
  intros h
  -- The proof would go here.
  sorry

end NUMINAMATH_GPT_range_of_t_l336_33625


namespace NUMINAMATH_GPT_min_value_fraction_l336_33658

theorem min_value_fraction (a b : ℝ) (h1 : 2 * a + b = 3) (h2 : a > 0) (h3 : b > 0) (h4 : ∃ n : ℕ, b = n) : 
  (∃ a b : ℝ, 2 * a + b = 3 ∧ a > 0 ∧ b > 0 ∧ (∃ n : ℕ, b = n) ∧ ((1/(2*a) + 2/b) = 2)) := 
by
  sorry

end NUMINAMATH_GPT_min_value_fraction_l336_33658


namespace NUMINAMATH_GPT_operation_example_l336_33664

def operation (a b : ℤ) : ℤ := 2 * a * b - b^2

theorem operation_example : operation 1 (-3) = -15 := by
  sorry

end NUMINAMATH_GPT_operation_example_l336_33664


namespace NUMINAMATH_GPT_random_event_is_crane_among_chickens_l336_33620

-- Definitions of the idioms as events
def coveringTheSkyWithOneHand : Prop := false
def fumingFromAllSevenOrifices : Prop := false
def stridingLikeAMeteor : Prop := false
def standingOutLikeACraneAmongChickens : Prop := ¬false

-- The theorem stating that Standing out like a crane among chickens is a random event
theorem random_event_is_crane_among_chickens :
  ¬coveringTheSkyWithOneHand ∧ ¬fumingFromAllSevenOrifices ∧ ¬stridingLikeAMeteor → standingOutLikeACraneAmongChickens :=
by 
  sorry

end NUMINAMATH_GPT_random_event_is_crane_among_chickens_l336_33620


namespace NUMINAMATH_GPT_depth_of_first_hole_l336_33631

theorem depth_of_first_hole :
  (45 * 8 * (80 * 6 * 40) / (45 * 8) : ℝ) = 53.33 := by
  -- This is where you would provide the proof, but it will be skipped with 'sorry'
  sorry

end NUMINAMATH_GPT_depth_of_first_hole_l336_33631


namespace NUMINAMATH_GPT_centroid_of_triangle_l336_33649

-- Definitions and conditions
def is_lattice_point (p : ℤ × ℤ) : Prop := 
  true -- Placeholder for a more specific definition if necessary

def triangle (A B C : ℤ × ℤ) : Prop := 
  true -- Placeholder for defining a triangle with vertices at integer grid points

def no_other_nodes_on_sides (A B C : ℤ × ℤ) : Prop := 
  true -- Placeholder to assert no other integer grid points on the sides

def exactly_one_node_inside (A B C O : ℤ × ℤ) : Prop := 
  true -- Placeholder to assert exactly one integer grid point inside the triangle

def medians_intersection_is_point_O (A B C O : ℤ × ℤ) : Prop := 
  true -- Placeholder to assert \(O\) is the intersection point of the medians

-- Theorem statement
theorem centroid_of_triangle 
  (A B C O : ℤ × ℤ)
  (h1 : is_lattice_point A)
  (h2 : is_lattice_point B)
  (h3 : is_lattice_point C)
  (h4 : triangle A B C)
  (h5 : no_other_nodes_on_sides A B C)
  (h6 : exactly_one_node_inside A B C O) : 
  medians_intersection_is_point_O A B C O :=
sorry

end NUMINAMATH_GPT_centroid_of_triangle_l336_33649


namespace NUMINAMATH_GPT_max_value_phi_l336_33657

theorem max_value_phi (φ : ℝ) (hφ : -Real.pi / 2 < φ ∧ φ < Real.pi / 2) :
  (∃ k : ℤ, φ = 2 * k * Real.pi + Real.pi / 2 - Real.pi / 3) →
  φ = Real.pi / 6 :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_max_value_phi_l336_33657


namespace NUMINAMATH_GPT_andrey_gifts_l336_33644

theorem andrey_gifts (n a : ℕ) (h : n * (n - 2) = a * (n - 1) + 16) : n = 18 :=
sorry

end NUMINAMATH_GPT_andrey_gifts_l336_33644


namespace NUMINAMATH_GPT_gcd_lcm_sum_l336_33694

-- Definitions
def gcd_42_70 := Nat.gcd 42 70
def lcm_8_32 := Nat.lcm 8 32

-- Theorem statement
theorem gcd_lcm_sum : gcd_42_70 + lcm_8_32 = 46 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_sum_l336_33694


namespace NUMINAMATH_GPT_average_income_PQ_l336_33638

/-
Conditions:
1. The average monthly income of Q and R is Rs. 5250.
2. The average monthly income of P and R is Rs. 6200.
3. The monthly income of P is Rs. 3000.
-/

def avg_income_QR := 5250
def avg_income_PR := 6200
def income_P := 3000

theorem average_income_PQ :
  ∃ (Q R : ℕ), ((Q + R) / 2 = avg_income_QR) ∧ ((income_P + R) / 2 = avg_income_PR) ∧ 
               (∀ (p q : ℕ), p = income_P → q = (Q + income_P) / 2 → q = 2050) :=
by
  sorry

end NUMINAMATH_GPT_average_income_PQ_l336_33638


namespace NUMINAMATH_GPT_star_compound_l336_33605

noncomputable def star (A B : ℝ) : ℝ := (A + B) / 4

theorem star_compound : star (star 3 11) 6 = 2.375 := by
  sorry

end NUMINAMATH_GPT_star_compound_l336_33605


namespace NUMINAMATH_GPT_carpet_shaded_area_l336_33674

theorem carpet_shaded_area
  (side_length_carpet : ℝ)
  (S : ℝ)
  (T : ℝ)
  (h1 : side_length_carpet = 12)
  (h2 : 12 / S = 4)
  (h3 : S / T = 2) :
  let area_big_square := S^2
  let area_small_squares := 4 * T^2
  area_big_square + area_small_squares = 18 := by
  sorry

end NUMINAMATH_GPT_carpet_shaded_area_l336_33674


namespace NUMINAMATH_GPT_part1_a_eq_zero_part2_range_of_a_l336_33672

noncomputable def f (x : ℝ) := abs (x + 1)
noncomputable def g (x : ℝ) (a : ℝ) := 2 * abs x + a

theorem part1_a_eq_zero :
  ∀ x, 0 < x + 1 → 0 < 2 * abs x → a = 0 →
  f x ≥ g x a ↔ (-1 / 3 : ℝ) ≤ x ∧ x ≤ 1 :=
sorry

theorem part2_range_of_a :
  ∃ x, f x ≥ g x a ↔ a ≤ 1 :=
sorry

end NUMINAMATH_GPT_part1_a_eq_zero_part2_range_of_a_l336_33672


namespace NUMINAMATH_GPT_arnold_protein_intake_l336_33619

theorem arnold_protein_intake :
  (∀ p q s : ℕ,  p = 18 / 2 ∧ q = 21 ∧ s = 56 → (p + q + s = 86)) := by
  sorry

end NUMINAMATH_GPT_arnold_protein_intake_l336_33619


namespace NUMINAMATH_GPT_gym_class_total_students_l336_33627

theorem gym_class_total_students (group1_members group2_members : ℕ) 
  (h1 : group1_members = 34) (h2 : group2_members = 37) :
  group1_members + group2_members = 71 :=
by
  sorry

end NUMINAMATH_GPT_gym_class_total_students_l336_33627


namespace NUMINAMATH_GPT_unique_position_of_chess_piece_l336_33665

theorem unique_position_of_chess_piece (x y : ℕ) (h : x^2 + x * y - 2 * y^2 = 13) : (x = 5) ∧ (y = 4) :=
sorry

end NUMINAMATH_GPT_unique_position_of_chess_piece_l336_33665


namespace NUMINAMATH_GPT_minimum_value_of_expression_l336_33634

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) :
  3 * x + 5 + 2 / x^5 ≥ 10 + 3 * (2 / 5) ^ (1 / 5) := by
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l336_33634


namespace NUMINAMATH_GPT_sqrt_computation_l336_33686

open Real

theorem sqrt_computation : sqrt ((5: ℝ)^2 * (7: ℝ)^4) = 245 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_sqrt_computation_l336_33686


namespace NUMINAMATH_GPT_divisibility_56786730_polynomial_inequality_l336_33688

theorem divisibility_56786730 (m n : ℤ) : 56786730 ∣ m * n * (m^60 - n^60) :=
sorry

theorem polynomial_inequality (m n : ℤ) : m^5 + 3 * m^4 * n - 5 * m^3 * n^2 - 15 * m^2 * n^3 + 4 * m * n^4 + 12 * n^5 ≠ 33 :=
sorry

end NUMINAMATH_GPT_divisibility_56786730_polynomial_inequality_l336_33688


namespace NUMINAMATH_GPT_range_of_f_l336_33629

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 6)

theorem range_of_f (x : ℝ) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  ∃ y, y = f x ∧ (y ≥ -3 / 2 ∧ y ≤ 3) :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_f_l336_33629


namespace NUMINAMATH_GPT_identify_quadratic_l336_33623

def is_quadratic (eq : String) : Prop :=
  eq = "x^2 - 2x + 1 = 0"

theorem identify_quadratic :
  is_quadratic "x^2 - 2x + 1 = 0" :=
by
  sorry

end NUMINAMATH_GPT_identify_quadratic_l336_33623


namespace NUMINAMATH_GPT_faye_pencils_allocation_l336_33654

theorem faye_pencils_allocation (pencils total_pencils rows : ℕ) (h_pencils : total_pencils = 6) (h_rows : rows = 2) (h_allocation : pencils = total_pencils / rows) : pencils = 3 := by
  sorry

end NUMINAMATH_GPT_faye_pencils_allocation_l336_33654


namespace NUMINAMATH_GPT_eliot_account_balance_l336_33602

-- Definitions for the conditions
variables {A E : ℝ}

--- Conditions rephrased into Lean:
-- 1. Al has more money than Eliot.
def al_more_than_eliot (A E : ℝ) : Prop := A > E

-- 2. The difference between their two accounts is 1/12 of the sum of their two accounts.
def difference_condition (A E : ℝ) : Prop := A - E = (1 / 12) * (A + E)

-- 3. If Al's account were to increase by 10% and Eliot's account were to increase by 15%, 
--     then Al would have exactly $22 more than Eliot in his account.
def percentage_increase_condition (A E : ℝ) : Prop := 1.10 * A = 1.15 * E + 22

-- Prove the total statement
theorem eliot_account_balance : 
  ∀ (A E : ℝ), al_more_than_eliot A E → difference_condition A E → percentage_increase_condition A E → E = 146.67 :=
by
  intros A E h1 h2 h3
  sorry

end NUMINAMATH_GPT_eliot_account_balance_l336_33602


namespace NUMINAMATH_GPT_treasure_hunt_distance_l336_33669

theorem treasure_hunt_distance (d : ℝ) : 
  (d < 8) → (d > 7) → (d > 9) → False :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_treasure_hunt_distance_l336_33669


namespace NUMINAMATH_GPT_arithmetic_problem_l336_33616

theorem arithmetic_problem : 
  (888.88 - 555.55 + 111.11) * 2 = 888.88 := 
sorry

end NUMINAMATH_GPT_arithmetic_problem_l336_33616


namespace NUMINAMATH_GPT_batsman_avg_increase_l336_33677

theorem batsman_avg_increase (R : ℕ) (A : ℕ) : 
  (R + 48 = 12 * 26) ∧ (R = 11 * A) → 26 - A = 2 :=
by
  intro h
  have h1 : R + 48 = 312 := h.1
  have h2 : R = 11 * A := h.2
  sorry

end NUMINAMATH_GPT_batsman_avg_increase_l336_33677


namespace NUMINAMATH_GPT_initial_amount_correct_l336_33617

noncomputable def initial_amount (A R T : ℝ) : ℝ :=
  A / (1 + (R * T) / 100)

theorem initial_amount_correct :
  initial_amount 2000 3.571428571428571 4 = 1750 :=
by
  sorry

end NUMINAMATH_GPT_initial_amount_correct_l336_33617


namespace NUMINAMATH_GPT_savings_on_discounted_milk_l336_33633

theorem savings_on_discounted_milk :
  let num_gallons := 8
  let price_per_gallon := 3.20
  let discount_rate := 0.25
  let discount_per_gallon := price_per_gallon * discount_rate
  let discounted_price_per_gallon := price_per_gallon - discount_per_gallon
  let total_cost_without_discount := num_gallons * price_per_gallon
  let total_cost_with_discount := num_gallons * discounted_price_per_gallon
  let savings := total_cost_without_discount - total_cost_with_discount
  savings = 6.40 :=
by
  sorry

end NUMINAMATH_GPT_savings_on_discounted_milk_l336_33633


namespace NUMINAMATH_GPT_k_interval_l336_33613

noncomputable def f (x k : ℝ) : ℝ := x^2 + (1 - k) * x - k

theorem k_interval (k : ℝ) :
  (∃! x : ℝ, 2 < x ∧ x < 3 ∧ f x k = 0) ↔ (2 < k ∧ k < 3) :=
by
  sorry

end NUMINAMATH_GPT_k_interval_l336_33613


namespace NUMINAMATH_GPT_sum_of_sequence_l336_33612

theorem sum_of_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) :
  (∀ n, a n = (-1 : ℤ)^(n+1) * (2*n - 1)) →
  (S 0 = 0) →
  (∀ n, S (n+1) = S n + a (n+1)) →
  (∀ n, S (n+1) = (-1 : ℤ)^(n+1) * (n+1)) :=
by
  intros h_a h_S0 h_S
  sorry

end NUMINAMATH_GPT_sum_of_sequence_l336_33612


namespace NUMINAMATH_GPT_monotonicity_decreasing_range_l336_33689

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + Real.pi / 4)

theorem monotonicity_decreasing_range (ω : ℝ) :
  (∀ x y : ℝ, (π / 2 < x ∧ x < π ∧ π / 2 < y ∧ y < π ∧ x < y) → f ω x > f ω y) ↔ (1 / 2 ≤ ω ∧ ω ≤ 5 / 4) :=
sorry

end NUMINAMATH_GPT_monotonicity_decreasing_range_l336_33689


namespace NUMINAMATH_GPT_fish_fishermen_problem_l336_33600

theorem fish_fishermen_problem (h: ℕ) (r: ℕ) (w_h: ℕ) (w_r: ℕ) (claimed_weight: ℕ) (total_real_weight: ℕ) 
  (total_fishermen: ℕ) :
  -- conditions
  (claimed_weight = 60) →
  (total_real_weight = 120) →
  (total_fishermen = 10) →
  (w_h = 30) →
  (w_r < 60 / 7) →
  (h + r = total_fishermen) →
  (2 * w_h * h + r * claimed_weight = claimed_weight * total_fishermen) →
  -- prove the number of regular fishermen
  (r = 7 ∨ r = 8) :=
sorry

end NUMINAMATH_GPT_fish_fishermen_problem_l336_33600


namespace NUMINAMATH_GPT_find_radius_of_cone_l336_33692

def slant_height : ℝ := 10
def curved_surface_area : ℝ := 157.07963267948966

theorem find_radius_of_cone
    (l : ℝ) (CSA : ℝ) (h1 : l = slant_height) (h2 : CSA = curved_surface_area) :
    ∃ r : ℝ, r = 5 := 
by
  sorry

end NUMINAMATH_GPT_find_radius_of_cone_l336_33692


namespace NUMINAMATH_GPT_max_a_value_l336_33626

theorem max_a_value (a : ℝ) (h : ∀ x : ℝ, |x - a| + |x - 3| ≥ 2 * a) : a ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_a_value_l336_33626


namespace NUMINAMATH_GPT_min_value_of_a_l336_33630

variables (a b c d : ℕ)

-- Conditions
def conditions : Prop :=
  a > b ∧ b > c ∧ c > d ∧
  a + b + c + d = 2004 ∧
  a^2 - b^2 + c^2 - d^2 = 2004

-- Theorem: minimum value of a
theorem min_value_of_a (h : conditions a b c d) : a = 503 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_l336_33630


namespace NUMINAMATH_GPT_log_inequality_l336_33693

theorem log_inequality (a b c : ℝ) (h1 : b^2 - a * c < 0) :
  ∀ x y : ℝ, a * (Real.log x)^2 + 2 * b * (Real.log x) * (Real.log y) + c * (Real.log y)^2 = 1 
  → a * 1^2 + 2 * b * 1 * (-1) + c * (-1)^2 = 1 → 
  -1 / Real.sqrt (a * c - b^2) ≤ Real.log (x * y) ∧ Real.log (x * y) ≤ 1 / Real.sqrt (a * c - b^2) :=
by
  sorry

end NUMINAMATH_GPT_log_inequality_l336_33693


namespace NUMINAMATH_GPT_not_prime_a_l336_33663

theorem not_prime_a 
  (a b : ℕ) 
  (h₁ : 0 < a)
  (h₂ : 0 < b)
  (h₃ : ∃ k : ℤ, (5 * a^4 + a^2) = k * (b^4 + 3 * b^2 + 4))
  : ¬ Nat.Prime a := 
sorry

end NUMINAMATH_GPT_not_prime_a_l336_33663


namespace NUMINAMATH_GPT_set_union_inter_example_l336_33698

open Set

theorem set_union_inter_example :
  let A := ({1, 2} : Set ℕ)
  let B := ({1, 2, 3} : Set ℕ)
  let C := ({2, 3, 4} : Set ℕ)
  (A ∩ B) ∪ C = ({1, 2, 3, 4} : Set ℕ) := by
    let A := ({1, 2} : Set ℕ)
    let B := ({1, 2, 3} : Set ℕ)
    let C := ({2, 3, 4} : Set ℕ)
    sorry

end NUMINAMATH_GPT_set_union_inter_example_l336_33698


namespace NUMINAMATH_GPT_men_l336_33624

-- Given conditions
variable (W M : ℕ)
variable (B : ℕ) [DecidableEq ℕ] -- number of boys
variable (total_earnings : ℕ)

def earnings : ℕ := 5 * M + W * M + 8 * W

-- Total earnings of men, women, and boys is Rs. 150.
def conditions : Prop := 
  5 * M = W * M ∧ 
  W * M = 8 * W ∧ 
  earnings = total_earnings

-- Prove men's wages (total wages for 5 men) is Rs. 50.
theorem men's_wages (hm : total_earnings = 150) (hb : W = 8) : 
  5 * M = 50 :=
by
  sorry

end NUMINAMATH_GPT_men_l336_33624


namespace NUMINAMATH_GPT_identify_heaviest_and_lightest_13_weighings_l336_33699

theorem identify_heaviest_and_lightest_13_weighings (coins : Fin 10 → ℝ) (h_distinct : Function.Injective coins) :
  ∃ f : (Fin 13 → ((Fin 10) × (Fin 10) × ℝ)), true :=
by
  sorry

end NUMINAMATH_GPT_identify_heaviest_and_lightest_13_weighings_l336_33699


namespace NUMINAMATH_GPT_appropriate_term_for_assessment_l336_33628

-- Definitions
def price : Type := String
def value : Type := String
def cost : Type := String
def expense : Type := String

-- Context for assessment of the project
def assessment_context : Type := Π (word : String), word ∈ ["price", "value", "cost", "expense"] → Prop

-- Main Lean statement
theorem appropriate_term_for_assessment (word : String) (h : word ∈ ["price", "value", "cost", "expense"]) :
  word = "value" :=
sorry

end NUMINAMATH_GPT_appropriate_term_for_assessment_l336_33628


namespace NUMINAMATH_GPT_length_GH_l336_33639

theorem length_GH (AB BC : ℝ) (hAB : AB = 10) (hBC : BC = 5) (DG DH GH : ℝ)
  (hDG : DG = DH) (hArea_DGH : 1 / 2 * DG * DH = 1 / 5 * (AB * BC)) :
  GH = 2 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_length_GH_l336_33639


namespace NUMINAMATH_GPT_abc_product_l336_33603

/-- Given a b c + a b + b c + a c + a + b + c = 164 -/
theorem abc_product :
  ∃ (a b c : ℕ), a * b * c + a * b + b * c + a * c + a + b + c = 164 ∧ a * b * c = 80 :=
by
  sorry

end NUMINAMATH_GPT_abc_product_l336_33603


namespace NUMINAMATH_GPT_binomial_510_510_l336_33642

-- Define binomial coefficient
def binomial (n k : ℕ) : ℕ :=
  if h : k ≤ n then Nat.choose n k else 0

theorem binomial_510_510 : binomial 510 510 = 1 :=
  by
    -- Skip the proof with sorry
    sorry

end NUMINAMATH_GPT_binomial_510_510_l336_33642


namespace NUMINAMATH_GPT_find_v2_poly_l336_33645

theorem find_v2_poly (x : ℤ) (v0 v1 v2 : ℤ) 
  (h1 : x = -4)
  (h2 : v0 = 1) 
  (h3 : v1 = v0 * x)
  (h4 : v2 = v1 * x + 6) :
  v2 = 22 :=
by
  -- To be filled with proof (example problem requirement specifies proof is not needed)
  sorry

end NUMINAMATH_GPT_find_v2_poly_l336_33645


namespace NUMINAMATH_GPT_find_remainder_in_division_l336_33683

theorem find_remainder_in_division
  (D : ℕ)
  (r : ℕ) -- the remainder when using the incorrect divisor
  (R : ℕ) -- the remainder when using the correct divisor
  (h1 : D = 12 * 63 + r)
  (h2 : D = 21 * 36 + R)
  : R = 0 :=
by
  sorry

end NUMINAMATH_GPT_find_remainder_in_division_l336_33683


namespace NUMINAMATH_GPT_prod2025_min_sum_l336_33646

theorem prod2025_min_sum : ∃ (a b : ℕ), a * b = 2025 ∧ a > 0 ∧ b > 0 ∧ (∀ (x y : ℕ), x * y = 2025 → x > 0 → y > 0 → x + y ≥ a + b) ∧ a + b = 90 :=
sorry

end NUMINAMATH_GPT_prod2025_min_sum_l336_33646


namespace NUMINAMATH_GPT_b5_b9_equal_16_l336_33661

-- Define the arithmetic sequence and conditions
variables {a : ℕ → ℝ} (h_arith : ∀ n m, a m = a n + (m - n) * (a 1 - a 0))
variable (h_non_zero : ∀ n, a n ≠ 0)
variable (h_cond : 2 * a 3 - (a 7)^2 + 2 * a 11 = 0)

-- Define the geometric sequence and condition
variables {b : ℕ → ℝ} (h_geom : ∀ n, b (n + 1) = b n * (b 1 / b 0))
variable (h_b7 : b 7 = a 7)

-- State the theorem to prove
theorem b5_b9_equal_16 : b 5 * b 9 = 16 :=
sorry

end NUMINAMATH_GPT_b5_b9_equal_16_l336_33661


namespace NUMINAMATH_GPT_final_ratio_l336_33652

-- Define initial conditions
def initial_milk_ratio : ℕ := 1
def initial_water_ratio : ℕ := 5
def total_parts : ℕ := initial_milk_ratio + initial_water_ratio
def can_capacity : ℕ := 8
def additional_milk : ℕ := 2
def initial_volume : ℕ := can_capacity - additional_milk
def part_volume : ℕ := initial_volume / total_parts

-- Define initial quantities
def initial_milk_quantity : ℕ := part_volume * initial_milk_ratio
def initial_water_quantity : ℕ := part_volume * initial_water_ratio

-- Define final quantities
def final_milk_quantity : ℕ := initial_milk_quantity + additional_milk
def final_water_quantity : ℕ := initial_water_quantity

-- Hypothesis: final ratios of milk and water
def final_ratio_of_milk_to_water : ℕ × ℕ := (final_milk_quantity, final_water_quantity)

-- Final ratio should be 3:5
theorem final_ratio (h : final_ratio_of_milk_to_water = (3, 5)) : final_ratio_of_milk_to_water = (3, 5) :=
  by
  sorry

end NUMINAMATH_GPT_final_ratio_l336_33652


namespace NUMINAMATH_GPT_sumOfTrianglesIs34_l336_33690

def triangleOp (a b c : ℕ) : ℕ := a * b - c

theorem sumOfTrianglesIs34 : 
  triangleOp 3 5 2 + triangleOp 4 6 3 = 34 := 
by
  sorry

end NUMINAMATH_GPT_sumOfTrianglesIs34_l336_33690


namespace NUMINAMATH_GPT_major_axis_length_l336_33614

theorem major_axis_length (r : ℝ) (minor_axis major_axis : ℝ) 
  (hr : r = 2) 
  (h_minor : minor_axis = 2 * r)
  (h_major : major_axis = 1.25 * minor_axis) :
  major_axis = 5 :=
by
  sorry

end NUMINAMATH_GPT_major_axis_length_l336_33614


namespace NUMINAMATH_GPT_max_value_l336_33651

theorem max_value (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) : 
  8 * x + 3 * y + 15 * z ≤ Real.sqrt 298 :=
sorry

end NUMINAMATH_GPT_max_value_l336_33651


namespace NUMINAMATH_GPT_costPrice_of_bat_is_152_l336_33687

noncomputable def costPriceOfBatForA (priceC : ℝ) (profitA : ℝ) (profitB : ℝ) : ℝ :=
  priceC / (1 + profitB) / (1 + profitA)

theorem costPrice_of_bat_is_152 :
  costPriceOfBatForA 228 0.20 0.25 = 152 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_costPrice_of_bat_is_152_l336_33687


namespace NUMINAMATH_GPT_triangle_height_l336_33650

theorem triangle_height (base height area : ℝ) 
(h_base : base = 3) (h_area : area = 9) 
(h_area_eq : area = (base * height) / 2) :
  height = 6 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_height_l336_33650


namespace NUMINAMATH_GPT_justin_reading_ratio_l336_33648

theorem justin_reading_ratio
  (pages_total : ℝ)
  (pages_first_day : ℝ)
  (pages_left : ℝ)
  (days_remaining : ℝ) :
  pages_total = 130 → 
  pages_first_day = 10 → 
  pages_left = pages_total - pages_first_day →
  days_remaining = 6 →
  (∃ R : ℝ, 60 * R = pages_left) → 
  ∃ R : ℝ, 60 * R = pages_left ∧ R = 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_justin_reading_ratio_l336_33648


namespace NUMINAMATH_GPT_find_certain_number_l336_33636

theorem find_certain_number (x : ℝ) (h : ((7 * (x + 5)) / 5) - 5 = 33) : x = 22 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l336_33636


namespace NUMINAMATH_GPT_mass_of_man_is_120_l336_33684

def length_of_boat : ℝ := 3
def breadth_of_boat : ℝ := 2
def height_water_rise : ℝ := 0.02
def density_of_water : ℝ := 1000
def volume_displaced : ℝ := length_of_boat * breadth_of_boat * height_water_rise
def mass_of_man := density_of_water * volume_displaced

theorem mass_of_man_is_120 : mass_of_man = 120 :=
by
  -- insert the detailed proof here
  sorry

end NUMINAMATH_GPT_mass_of_man_is_120_l336_33684


namespace NUMINAMATH_GPT_y_exceeds_x_by_35_percent_l336_33643

theorem y_exceeds_x_by_35_percent {x y : ℝ} (h : x = 0.65 * y) : ((y - x) / x) * 100 = 35 :=
by
  sorry

end NUMINAMATH_GPT_y_exceeds_x_by_35_percent_l336_33643


namespace NUMINAMATH_GPT_max_articles_produced_l336_33640

variables (a b c d p q r s z : ℝ)
variables (h1 : d = (a^2 * b * c) / z)
variables (h2 : p * q * r ≤ s)

theorem max_articles_produced : 
  p * q * r * (a / z) = s * (a / z) :=
by
  sorry

end NUMINAMATH_GPT_max_articles_produced_l336_33640


namespace NUMINAMATH_GPT_minimum_employment_age_l336_33609

/-- This structure represents the conditions of the problem -/
structure EmploymentConditions where
  jane_current_age : ℕ  -- Jane's current age
  years_until_dara_half_age : ℕ  -- Years until Dara is half Jane's age
  years_until_dara_min_age : ℕ  -- Years until Dara reaches minimum employment age

/-- The proof problem statement -/
theorem minimum_employment_age (conds : EmploymentConditions)
  (h_jane : conds.jane_current_age = 28)
  (h_half_age : conds.years_until_dara_half_age = 6)
  (h_min_age : conds.years_until_dara_min_age = 14) :
  let jane_in_six := conds.jane_current_age + conds.years_until_dara_half_age
  let dara_in_six := jane_in_six / 2
  let dara_now := dara_in_six - conds.years_until_dara_half_age
  let M := dara_now + conds.years_until_dara_min_age
  M = 25 :=
by
  sorry

end NUMINAMATH_GPT_minimum_employment_age_l336_33609


namespace NUMINAMATH_GPT_isosceles_right_triangle_area_l336_33697

theorem isosceles_right_triangle_area (p : ℝ) : 
  ∃ (A : ℝ), A = (3 - 2 * Real.sqrt 2) * p^2 
  → (∃ (x : ℝ), 2 * x + x * Real.sqrt 2 = 2 * p ∧ A = 1 / 2 * x^2) := 
sorry

end NUMINAMATH_GPT_isosceles_right_triangle_area_l336_33697


namespace NUMINAMATH_GPT_sum_of_first_9_terms_l336_33679

-- Define the arithmetic sequence {a_n} and the sum S_n of the first n terms
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  S 0 = 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

-- Define the conditions given in the problem
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
axiom arith_seq : arithmetic_sequence a
axiom sum_terms : sum_of_first_n_terms a S
axiom S3 : S 3 = 30
axiom S6 : S 6 = 100

-- Goal: Prove that S 9 = 170
theorem sum_of_first_9_terms : S 9 = 170 :=
sorry -- Placeholder for the proof

end NUMINAMATH_GPT_sum_of_first_9_terms_l336_33679


namespace NUMINAMATH_GPT_problem_l336_33681

noncomputable def vector_a (ω φ x : ℝ) : ℝ × ℝ := (Real.sin (ω / 2 * x + φ), 1)
noncomputable def vector_b (ω φ x : ℝ) : ℝ × ℝ := (1, Real.cos (ω / 2 * x + φ))
noncomputable def f (ω φ x : ℝ) : ℝ := 
  let a := vector_a ω φ x
  let b := vector_b ω φ x
  (a.1 + b.1) * (a.1 - b.1) + (a.2 + b.2) * (a.2 - b.2)

theorem problem 
  (ω φ : ℝ) (hω : ω > 0) (hφ : 0 < φ ∧ φ < π / 4)
  (h_period : Function.Periodic (f ω φ) 4)
  (h_point1 : f ω φ 1 = 1 / 2) : 
  ω = π / 2 ∧ ∀ x, -1 ≤ x ∧ x ≤ 1 → -1 ≤ f (π / 2) (π / 12) x ∧ f (π / 2) (π / 12) x ≤ 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_problem_l336_33681


namespace NUMINAMATH_GPT_original_profit_percentage_l336_33685

theorem original_profit_percentage (C S : ℝ) (hC : C = 70)
(h1 : S - 14.70 = 1.30 * (C * 0.80)) :
  (S - C) / C * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_original_profit_percentage_l336_33685


namespace NUMINAMATH_GPT_cost_of_large_fries_l336_33641

noncomputable def cost_of_cheeseburger : ℝ := 3.65
noncomputable def cost_of_milkshake : ℝ := 2
noncomputable def cost_of_coke : ℝ := 1
noncomputable def cost_of_cookie : ℝ := 0.5
noncomputable def tax : ℝ := 0.2
noncomputable def toby_initial_amount : ℝ := 15
noncomputable def toby_remaining_amount : ℝ := 7
noncomputable def split_bill : ℝ := 2

theorem cost_of_large_fries : 
  let total_meal_cost := (split_bill * (toby_initial_amount - toby_remaining_amount))
  let total_cost_so_far := (2 * cost_of_cheeseburger) + cost_of_milkshake + cost_of_coke + (3 * cost_of_cookie) + tax
  total_meal_cost - total_cost_so_far = 4 := 
by
  sorry

end NUMINAMATH_GPT_cost_of_large_fries_l336_33641


namespace NUMINAMATH_GPT_integers_with_product_72_and_difference_4_have_sum_20_l336_33676

theorem integers_with_product_72_and_difference_4_have_sum_20 :
  ∃ (x y : ℕ), (x * y = 72) ∧ (x - y = 4) ∧ (x + y = 20) :=
sorry

end NUMINAMATH_GPT_integers_with_product_72_and_difference_4_have_sum_20_l336_33676


namespace NUMINAMATH_GPT_common_point_eq_l336_33655

theorem common_point_eq (a b c d : ℝ) (h₀ : a ≠ b) 
  (h₁ : ∃ x y : ℝ, y = a * x + a ∧ y = b * x + b ∧ y = c * x + d) : 
  d = c :=
by
  sorry

end NUMINAMATH_GPT_common_point_eq_l336_33655


namespace NUMINAMATH_GPT_ellipse_chord_slope_relation_l336_33678

theorem ellipse_chord_slope_relation
    (a b : ℝ) (h : a > b) (h1 : b > 0)
    (A B M : ℝ × ℝ)
    (hM_midpoint : M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2))
    (hAB_slope : A.1 ≠ B.1)
    (K_AB K_OM : ℝ)
    (hK_AB : K_AB = (B.2 - A.2) / (B.1 - A.1))
    (hK_OM : K_OM = (M.2 - 0) / (M.1 - 0)) :
  K_AB * K_OM = - (b ^ 2) / (a ^ 2) := 
  sorry

end NUMINAMATH_GPT_ellipse_chord_slope_relation_l336_33678


namespace NUMINAMATH_GPT_maximum_value_of_expression_is_4_l336_33662

noncomputable def maximimum_integer_value (x : ℝ) : ℝ :=
    (5 * x^2 + 10 * x + 12) / (5 * x^2 + 10 * x + 2)

theorem maximum_value_of_expression_is_4 :
    ∃ x : ℝ, ∀ y : ℝ, maximimum_integer_value y ≤ 4 ∧ maximimum_integer_value x = 4 := 
by 
  -- Proof omitted for now
  sorry

end NUMINAMATH_GPT_maximum_value_of_expression_is_4_l336_33662


namespace NUMINAMATH_GPT_rin_craters_difference_l336_33604

theorem rin_craters_difference (d da r : ℕ) (h1 : d = 35) (h2 : da = d - 10) (h3 : r = 75) :
  r - (d + da) = 15 :=
by
  sorry

end NUMINAMATH_GPT_rin_craters_difference_l336_33604


namespace NUMINAMATH_GPT_extreme_value_at_3_increasing_on_interval_l336_33635

def f (a : ℝ) (x : ℝ) : ℝ := 2*x^3 - 3*(a+1)*x^2 + 6*a*x + 8

theorem extreme_value_at_3 (a : ℝ) : (∃ x, x = 3 ∧ 6*x^2 - 6*(a+1)*x + 6*a = 0) → a = 3 :=
by
  sorry

theorem increasing_on_interval (a : ℝ) : (∀ x, x < 0 → 6*(x-a)*(x-1) > 0) → 0 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_extreme_value_at_3_increasing_on_interval_l336_33635


namespace NUMINAMATH_GPT_spurs_team_players_l336_33667

theorem spurs_team_players (total_basketballs : ℕ) (basketballs_per_player : ℕ) (h : total_basketballs = 242) (h1 : basketballs_per_player = 11) : total_basketballs / basketballs_per_player = 22 :=
by { sorry }

end NUMINAMATH_GPT_spurs_team_players_l336_33667


namespace NUMINAMATH_GPT_triangle_inequality_l336_33647

theorem triangle_inequality (a b c p S r : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : c + a > b)
  (hp : p = (a + b + c) / 2)
  (hS : S = Real.sqrt (p * (p - a) * (p - b) * (p - c)))
  (hr : r = S / p):
  1 / (p - a) ^ 2 + 1 / (p - b) ^ 2 + 1 / (p - c) ^ 2 ≥ 1 / r ^ 2 :=
sorry

end NUMINAMATH_GPT_triangle_inequality_l336_33647


namespace NUMINAMATH_GPT_geom_seq_min_value_l336_33671

theorem geom_seq_min_value :
  let a1 := 2
  ∃ r : ℝ, ∀ a2 a3,
    a2 = 2 * r ∧ 
    a3 = 2 * r^2 →
    3 * a2 + 6 * a3 = -3/2 := by
  sorry

end NUMINAMATH_GPT_geom_seq_min_value_l336_33671
