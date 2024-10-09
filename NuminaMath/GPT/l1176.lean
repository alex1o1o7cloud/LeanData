import Mathlib

namespace probability_of_staying_in_dark_l1176_117690

theorem probability_of_staying_in_dark (revolutions_per_minute : ℕ) (time_in_seconds : ℕ) (dark_time : ℕ) :
  revolutions_per_minute = 2 →
  time_in_seconds = 60 →
  dark_time = 5 →
  (5 / 6 : ℝ) = 5 / 6 :=
by
  intros
  sorry

end probability_of_staying_in_dark_l1176_117690


namespace div_poly_odd_power_l1176_117648

theorem div_poly_odd_power (a b : ℤ) (n : ℕ) : (a + b) ∣ (a^(2*n+1) + b^(2*n+1)) :=
sorry

end div_poly_odd_power_l1176_117648


namespace find_natural_numbers_l1176_117670

theorem find_natural_numbers (a b : ℕ) (p : ℕ) (hp : Nat.Prime p)
  (h : a^3 - b^3 = 633 * p) : a = 16 ∧ b = 13 :=
by
  sorry

end find_natural_numbers_l1176_117670


namespace tile_coverage_fraction_l1176_117680

structure Room where
  rect_length : ℝ
  rect_width : ℝ
  tri_base : ℝ
  tri_height : ℝ
  
structure Tiles where
  square_tiles : ℕ
  triangular_tiles : ℕ
  triangle_base : ℝ
  triangle_height : ℝ
  tile_area : ℝ
  triangular_tile_area : ℝ
  
noncomputable def fractionalTileCoverage (room : Room) (tiles : Tiles) : ℝ :=
  let rect_area := room.rect_length * room.rect_width
  let tri_area := (room.tri_base * room.tri_height) / 2
  let total_room_area := rect_area + tri_area
  let total_tile_area := (tiles.square_tiles * tiles.tile_area) + (tiles.triangular_tiles * tiles.triangular_tile_area)
  total_tile_area / total_room_area

theorem tile_coverage_fraction
  (room : Room) (tiles : Tiles)
  (h1 : room.rect_length = 12)
  (h2 : room.rect_width = 20)
  (h3 : room.tri_base = 10)
  (h4 : room.tri_height = 8)
  (h5 : tiles.square_tiles = 40)
  (h6 : tiles.triangular_tiles = 4)
  (h7 : tiles.tile_area = 1)
  (h8 : tiles.triangular_tile_area = (1 * 1) / 2) :
  fractionalTileCoverage room tiles = 3 / 20 :=
by 
  sorry

end tile_coverage_fraction_l1176_117680


namespace sqrt_equiv_1715_l1176_117624

noncomputable def sqrt_five_squared_times_seven_sixth : ℕ := 
  Nat.sqrt (5^2 * 7^6)

theorem sqrt_equiv_1715 : sqrt_five_squared_times_seven_sixth = 1715 := by
  sorry

end sqrt_equiv_1715_l1176_117624


namespace sugar_more_than_flour_l1176_117647

def flour_needed : Nat := 9
def sugar_needed : Nat := 11
def flour_added : Nat := 4
def sugar_added : Nat := 0

def flour_remaining : Nat := flour_needed - flour_added
def sugar_remaining : Nat := sugar_needed - sugar_added

theorem sugar_more_than_flour : sugar_remaining - flour_remaining = 6 :=
by
  sorry

end sugar_more_than_flour_l1176_117647


namespace lattice_point_exists_l1176_117678

noncomputable def exists_distant_lattice_point : Prop :=
∃ (X Y : ℤ), ∀ (x y : ℤ), gcd x y = 1 → (X - x) ^ 2 + (Y - y) ^ 2 ≥ 1995 ^ 2

theorem lattice_point_exists : exists_distant_lattice_point :=
sorry

end lattice_point_exists_l1176_117678


namespace evaluate_expression_l1176_117663

variable {a b c : ℝ}

theorem evaluate_expression
  (h : a / (35 - a) + b / (75 - b) + c / (85 - c) = 5) :
  7 / (35 - a) + 15 / (75 - b) + 17 / (85 - c) = 8 / 5 := by
  sorry

end evaluate_expression_l1176_117663


namespace max_product_ge_993_squared_l1176_117691

theorem max_product_ge_993_squared (a : Fin 1985 → Fin 1985) (hperm : ∀ n : Fin 1985, ∃ k : Fin 1985, a k = n ∧ ∃ m : Fin 1985, a m = n) :
  ∃ k : Fin 1985, a k * k ≥ 993^2 :=
sorry

end max_product_ge_993_squared_l1176_117691


namespace total_number_of_toys_l1176_117653

theorem total_number_of_toys (average_cost_Dhoni_toys : ℕ) (number_Dhoni_toys : ℕ) 
    (price_David_toy : ℕ) (new_avg_cost : ℕ) 
    (h1 : average_cost_Dhoni_toys = 10) (h2 : number_Dhoni_toys = 5) 
    (h3 : price_David_toy = 16) (h4 : new_avg_cost = 11) : 
    (number_Dhoni_toys + 1) = 6 := 
by
  sorry

end total_number_of_toys_l1176_117653


namespace min_value_of_inverse_sum_l1176_117688

theorem min_value_of_inverse_sum {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : (1/x) + (1/y) ≥ 4 :=
by
  sorry

end min_value_of_inverse_sum_l1176_117688


namespace selling_price_of_cycle_l1176_117631

theorem selling_price_of_cycle (cp : ℝ) (loss_percentage : ℝ) (sp : ℝ) : 
  cp = 1400 → loss_percentage = 20 → sp = cp - (loss_percentage / 100) * cp → sp = 1120 :=
by 
  intro h1 h2 h3
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end selling_price_of_cycle_l1176_117631


namespace hexagon_perimeter_l1176_117630

theorem hexagon_perimeter
  (A B C D E F : Type)  -- vertices of the hexagon
  (angle_A : ℝ) (angle_C : ℝ) (angle_E : ℝ)  -- nonadjacent angles
  (angle_B : ℝ) (angle_D : ℝ) (angle_F : ℝ)  -- adjacent angles
  (area_hexagon : ℝ)
  (side_length : ℝ)
  (h1 : angle_A = 120) (h2 : angle_C = 120) (h3 : angle_E = 120)
  (h4 : angle_B = 60) (h5 : angle_D = 60) (h6 : angle_F = 60)
  (h7 : area_hexagon = 24)
  (h8 : ∃ s, ∀ (u v : Type), side_length = s) :
  6 * side_length = 24 / (Real.sqrt 3 ^ (1/4)) :=
by
  sorry

end hexagon_perimeter_l1176_117630


namespace average_speed_rest_of_trip_l1176_117652

variable (v : ℝ) -- The average speed for the rest of the trip
variable (d1 : ℝ := 30 * 5) -- Distance for the first part of the trip
variable (t1 : ℝ := 5) -- Time for the first part of the trip
variable (t_total : ℝ := 7.5) -- Total time for the trip
variable (avg_total : ℝ := 34) -- Average speed for the entire trip

def total_distance := avg_total * t_total
def d2 := total_distance - d1
def t2 := t_total - t1

theorem average_speed_rest_of_trip : 
  v = 42 :=
by
  let distance_rest := d2
  let time_rest := t2
  have v_def : v = distance_rest / time_rest := by sorry
  have v_value : v = 42 := by sorry
  exact v_value

end average_speed_rest_of_trip_l1176_117652


namespace dig_second_hole_l1176_117622

theorem dig_second_hole (w1 h1 d1 w2 d2 : ℕ) (extra_workers : ℕ) (h2 : ℕ) :
  w1 = 45 ∧ h1 = 8 ∧ d1 = 30 ∧ extra_workers = 65 ∧
  w2 = w1 + extra_workers ∧ d2 = 55 →
  360 * d2 / d1 = w2 * h2 →
  h2 = 6 :=
by
  intros h cond
  sorry

end dig_second_hole_l1176_117622


namespace pete_mileage_l1176_117627

def steps_per_flip : Nat := 100000
def flips : Nat := 50
def final_reading : Nat := 25000
def steps_per_mile : Nat := 2000

theorem pete_mileage :
  let total_steps := (steps_per_flip * flips) + final_reading
  let total_miles := total_steps.toFloat / steps_per_mile.toFloat
  total_miles = 2512.5 :=
by
  sorry

end pete_mileage_l1176_117627


namespace range_of_a_l1176_117611

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + 2 * x + a ≥ 0) : a ≥ 1 :=
sorry

end range_of_a_l1176_117611


namespace candidates_appeared_l1176_117650

-- Define the number of appeared candidates in state A and state B
variables (X : ℝ)

-- The conditions given in the problem
def condition1 : Prop := (0.07 * X = 0.06 * X + 83)

-- The claim that needs to be proved
def claim : Prop := (X = 8300)

-- The theorem statement in Lean 4
theorem candidates_appeared (X : ℝ) (h1 : condition1 X) : claim X := by
  -- Proof is omitted
  sorry

end candidates_appeared_l1176_117650


namespace least_k_for_divisibility_l1176_117666

theorem least_k_for_divisibility (k : ℕ) : (k ^ 4) % 1260 = 0 ↔ k ≥ 210 :=
sorry

end least_k_for_divisibility_l1176_117666


namespace hash_nesting_example_l1176_117671

def hash (N : ℝ) : ℝ :=
  0.5 * N + 2

theorem hash_nesting_example : hash (hash (hash (hash 20))) = 5 :=
by
  sorry

end hash_nesting_example_l1176_117671


namespace find_max_value_l1176_117617

-- We define the conditions as Lean definitions and hypotheses
def is_distinct_digits (A B C D E F : ℕ) : Prop :=
  (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (A ≠ F) ∧
  (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (B ≠ F) ∧
  (C ≠ D) ∧ (C ≠ E) ∧ (C ≠ F) ∧
  (D ≠ E) ∧ (D ≠ F) ∧
  (E ≠ F)

def all_digits_in_range (A B C D E F : ℕ) : Prop :=
  (1 ≤ A) ∧ (A ≤ 8) ∧
  (1 ≤ B) ∧ (B ≤ 8) ∧
  (1 ≤ C) ∧ (C ≤ 8) ∧
  (1 ≤ D) ∧ (D ≤ 8) ∧
  (1 ≤ E) ∧ (E ≤ 8) ∧
  (1 ≤ F) ∧ (F ≤ 8)

def divisible_by_99 (n : ℕ) : Prop :=
  (n % 99 = 0)

theorem find_max_value (A B C D E F : ℕ) :
  is_distinct_digits A B C D E F →
  all_digits_in_range A B C D E F →
  divisible_by_99 (100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F) →
  100000 * A + 10000 * B + 1000 * C + 100 * D + 10 * E + F = 87653412 :=
sorry

end find_max_value_l1176_117617


namespace total_fruit_cost_is_173_l1176_117687

-- Define the cost of a single orange and a single apple
def orange_cost := 2
def apple_cost := 3
def banana_cost := 1

-- Define the number of fruits each person has
def louis_oranges := 5
def louis_apples := 3

def samantha_oranges := 8
def samantha_apples := 7

def marley_oranges := 2 * louis_oranges
def marley_apples := 3 * samantha_apples

def edward_oranges := 3 * louis_oranges
def edward_bananas := 4

-- Define the cost of fruits for each person
def louis_cost := (louis_oranges * orange_cost) + (louis_apples * apple_cost)
def samantha_cost := (samantha_oranges * orange_cost) + (samantha_apples * apple_cost)
def marley_cost := (marley_oranges * orange_cost) + (marley_apples * apple_cost)
def edward_cost := (edward_oranges * orange_cost) + (edward_bananas * banana_cost)

-- Define the total cost for all four people
def total_cost := louis_cost + samantha_cost + marley_cost + edward_cost

-- Statement to prove that the total cost is $173
theorem total_fruit_cost_is_173 : total_cost = 173 :=
by
  sorry

end total_fruit_cost_is_173_l1176_117687


namespace min_value_of_fraction_l1176_117657

theorem min_value_of_fraction (n : ℕ) (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 2) :
  (1 / (1 + a^n) + 1 / (1 + b^n)) = 1 :=
sorry

end min_value_of_fraction_l1176_117657


namespace find_c_l1176_117673

noncomputable def f (c x : ℝ) : ℝ :=
  c * x^3 + 17 * x^2 - 4 * c * x + 45

theorem find_c (h : f c (-5) = 0) : c = 94 / 21 :=
by sorry

end find_c_l1176_117673


namespace factorize_expression_l1176_117603

theorem factorize_expression (a b : ℝ) : 3 * a ^ 2 - 3 * b ^ 2 = 3 * (a + b) * (a - b) :=
by
  sorry

end factorize_expression_l1176_117603


namespace patty_weeks_without_chores_correct_l1176_117642

noncomputable def patty_weeks_without_chores : ℕ := by
  let cookie_per_chore := 3
  let chores_per_week_per_sibling := 4
  let siblings := 2
  let dollars := 15
  let cookie_pack_size := 24
  let cookie_pack_cost := 3

  let packs := dollars / cookie_pack_cost
  let total_cookies := packs * cookie_pack_size
  let weekly_cookies_needed := chores_per_week_per_sibling * cookie_per_chore * siblings

  exact total_cookies / weekly_cookies_needed

theorem patty_weeks_without_chores_correct : patty_weeks_without_chores = 5 := sorry

end patty_weeks_without_chores_correct_l1176_117642


namespace sample_size_is_correct_l1176_117656

-- Define the conditions
def num_classes := 40
def students_per_class := 50
def selected_students := 150

-- Define the statement to prove the sample size
theorem sample_size_is_correct : selected_students = 150 := by 
  -- Proof is skipped with sorry
  sorry

end sample_size_is_correct_l1176_117656


namespace negation_true_l1176_117613

theorem negation_true (a : ℝ) : ¬ (∀ a : ℝ, a ≤ 2 → a^2 < 4) :=
sorry

end negation_true_l1176_117613


namespace complement_inter_of_A_and_B_l1176_117621

open Set

variable (U A B : Set ℕ)

theorem complement_inter_of_A_and_B:
  U = {1, 2, 3, 4, 5}
  ∧ A = {1, 2, 3}
  ∧ B = {2, 3, 4} 
  → U \ (A ∩ B) = {1, 4, 5} :=
by
  sorry

end complement_inter_of_A_and_B_l1176_117621


namespace vector_addition_l1176_117618

-- Let vectors a and b be defined as
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, -3)

-- Theorem statement to prove
theorem vector_addition : a + 2 • b = (4, -5) :=
by
  sorry

end vector_addition_l1176_117618


namespace ellipse_equation_l1176_117620

theorem ellipse_equation (a : ℝ) (x y : ℝ) (h : (x, y) = (-3, 2)) :
  (∃ a : ℝ, ∀ x y : ℝ, x^2 / 15 + y^2 / 10 = 1) ↔ (x, y) ∈ { p : ℝ × ℝ | p.1^2 / 15 + p.2^2 / 10 = 1 } :=
by
  have h1 : 15 = a^2 := by
    sorry
  have h2 : 10 = a^2 - 5 := by
    sorry
  sorry

end ellipse_equation_l1176_117620


namespace circle_tangent_to_ellipse_l1176_117607

theorem circle_tangent_to_ellipse {r : ℝ} 
  (h1: ∀ p: ℝ × ℝ, p ≠ (0, 0) → ((p.1 - r)^2 + p.2^2 = r^2 → p.1^2 + 4 * p.2^2 = 8))
  (h2: ∃ p: ℝ × ℝ, p ≠ (0, 0) ∧ ((p.1 - r)^2 + p.2^2 = r^2 ∧ p.1^2 + 4 * p.2^2 = 8)):
  r = Real.sqrt (3 / 2) :=
by
  sorry

end circle_tangent_to_ellipse_l1176_117607


namespace preimage_exists_l1176_117683

-- Define the mapping function f
def f (x y : ℚ) : ℚ × ℚ :=
  (x + 2 * y, 2 * x - y)

-- Define the statement
theorem preimage_exists (x y : ℚ) :
  f x y = (3, 1) → (x, y) = (-1/3, 5/3) :=
by
  sorry

end preimage_exists_l1176_117683


namespace megan_final_balance_same_as_starting_balance_l1176_117651

theorem megan_final_balance_same_as_starting_balance :
  let starting_balance : ℝ := 125
  let increased_balance := starting_balance * (1 + 0.25)
  let final_balance := increased_balance * (1 - 0.20)
  final_balance = starting_balance :=
by
  sorry

end megan_final_balance_same_as_starting_balance_l1176_117651


namespace oldest_son_cookies_l1176_117640

def youngest_son_cookies : Nat := 2
def total_cookies : Nat := 54
def days : Nat := 9

theorem oldest_son_cookies : ∃ x : Nat, 9 * (x + youngest_son_cookies) = total_cookies ∧ x = 4 := by
  sorry

end oldest_son_cookies_l1176_117640


namespace prob_same_gender_eq_two_fifths_l1176_117638

-- Define the number of male and female students
def num_male_students : ℕ := 3
def num_female_students : ℕ := 2

-- Define the total number of students
def total_students : ℕ := num_male_students + num_female_students

-- Define the probability calculation
def probability_same_gender := (num_male_students * (num_male_students - 1) / 2 + num_female_students * (num_female_students - 1) / 2) / (total_students * (total_students - 1) / 2)

theorem prob_same_gender_eq_two_fifths :
  probability_same_gender = 2 / 5 :=
by
  -- Proof is omitted
  sorry

end prob_same_gender_eq_two_fifths_l1176_117638


namespace largest_four_digit_sum_23_l1176_117659

theorem largest_four_digit_sum_23 : ∃ (n : ℕ), (∃ (a b c d : ℕ), n = a * 1000 + b * 100 + c * 10 + d ∧ a + b + c + d = 23 ∧ 1000 ≤ n ∧ n < 10000) ∧ n = 9950 :=
  sorry

end largest_four_digit_sum_23_l1176_117659


namespace find_f_of_2_l1176_117676

-- Definitions based on problem conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def g (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  f (x) + 9

-- The main statement to proof that f(2) = 6 under the given conditions
theorem find_f_of_2 (f : ℝ → ℝ)
  (hf : is_odd_function f)
  (hg : ∀ x, g f x = f x + 9)
  (h : g f (-2) = 3) :
  f 2 = 6 := 
sorry

end find_f_of_2_l1176_117676


namespace number_of_candidates_l1176_117635

theorem number_of_candidates (n : ℕ) (h : n * (n - 1) = 42) : n = 7 :=
sorry

end number_of_candidates_l1176_117635


namespace last_digit_of_product_of_consecutive_numbers_l1176_117660

theorem last_digit_of_product_of_consecutive_numbers (n : ℕ) (k : ℕ) (h1 : k > 5)
    (h2 : n = (k + 1) * (k + 2) * (k + 3) * (k + 4))
    (h3 : n % 10 ≠ 0) : n % 10 = 4 :=
sorry -- Proof not provided as per instructions.

end last_digit_of_product_of_consecutive_numbers_l1176_117660


namespace factorization_correct_l1176_117667

theorem factorization_correct (c d : ℤ) (h : 25 * x^2 - 160 * x - 144 = (5 * x + c) * (5 * x + d)) : c + 2 * d = -2 := 
sorry

end factorization_correct_l1176_117667


namespace no_positive_integer_solution_l1176_117698

theorem no_positive_integer_solution (a b c d : ℕ) (h1 : a^2 + b^2 = c^2 - d^2) (h2 : a * b = c * d) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : false := 
by 
  sorry

end no_positive_integer_solution_l1176_117698


namespace round_trip_time_l1176_117699

def boat_speed := 9 -- speed of the boat in standing water (kmph)
def stream_speed := 6 -- speed of the stream (kmph)
def distance := 210 -- distance to the place (km)

def upstream_speed := boat_speed - stream_speed
def downstream_speed := boat_speed + stream_speed

def time_upstream := distance / upstream_speed
def time_downstream := distance / downstream_speed
def total_time := time_upstream + time_downstream

theorem round_trip_time : total_time = 84 := by
  sorry

end round_trip_time_l1176_117699


namespace isosceles_triangle_properties_l1176_117601

noncomputable def isosceles_triangle_sides (a : ℝ) : ℝ × ℝ × ℝ :=
  let x := a * Real.sqrt 3
  let y := 2 * x / 3
  let z := (x + y) / 2
  (x, z, z)

theorem isosceles_triangle_properties (a x y z : ℝ) 
  (h1 : x * y = 2 * a ^ 2) 
  (h2 : x + y = 2 * z) 
  (h3 : y ^ 2 + (x / 2) ^ 2 = z ^ 2) : 
  x = a * Real.sqrt 3 ∧ 
  z = 5 * a * Real.sqrt 3 / 6 :=
by
-- Proof goes here
sorry

end isosceles_triangle_properties_l1176_117601


namespace correct_option_D_l1176_117644

theorem correct_option_D (x : ℝ) : (x - 1)^2 = x^2 + 1 - 2 * x :=
by sorry

end correct_option_D_l1176_117644


namespace find_base_c_l1176_117693

theorem find_base_c (c : ℕ) : (c^3 - 7*c^2 - 18*c - 8 = 0) → c = 10 :=
by
  sorry

end find_base_c_l1176_117693


namespace ratio_josh_to_doug_l1176_117674

theorem ratio_josh_to_doug (J D B : ℕ) (h1 : J + D + B = 68) (h2 : J = 2 * B) (h3 : D = 32) : J / D = 3 / 4 := 
by
  sorry

end ratio_josh_to_doug_l1176_117674


namespace evaluate_fraction_l1176_117602

theorem evaluate_fraction : 1 + 3 / (4 + 5 / (6 + 7 / 8)) = 85 / 52 :=
by sorry

end evaluate_fraction_l1176_117602


namespace additional_trams_proof_l1176_117665

-- Definitions for the conditions
def initial_tram_count : Nat := 12
def total_distance : Nat := 60
def initial_interval : Nat := total_distance / initial_tram_count
def reduced_interval : Nat := initial_interval - (initial_interval / 5)
def final_tram_count : Nat := total_distance / reduced_interval
def additional_trams_needed : Nat := final_tram_count - initial_tram_count

-- The theorem we need to prove
theorem additional_trams_proof : additional_trams_needed = 3 :=
by
  sorry

end additional_trams_proof_l1176_117665


namespace sequence_properties_l1176_117695

-- Define the arithmetic-geometric sequence and its sum
def a_n (n : ℕ) : ℕ := 2^(n-1)
def S_n (n : ℕ) : ℕ := 2^n - 1
def T_n (n : ℕ) : ℕ := 2^(n+1) - n - 2

theorem sequence_properties : 
(S_n 3 = 7) ∧ (S_n 6 = 63) → 
(∀ n: ℕ, a_n n = 2^(n-1)) ∧ 
(∀ n: ℕ, S_n n = 2^n - 1) ∧ 
(∀ n: ℕ, T_n n = 2^(n+1) - n - 2) :=
by
  sorry

end sequence_properties_l1176_117695


namespace cost_per_serving_is_3_62_l1176_117658

noncomputable def cost_per_serving : ℝ :=
  let beef_cost := 4 * 6
  let chicken_cost := (2.2 * 5) * 0.85
  let carrots_cost := 2 * 1.50
  let potatoes_cost := (1.5 * 1.80) * 0.85
  let onions_cost := 1 * 3
  let discounted_carrots := carrots_cost * 0.80
  let discounted_potatoes := potatoes_cost * 0.80
  let total_cost_before_tax := beef_cost + chicken_cost + discounted_carrots + discounted_potatoes + onions_cost
  let sales_tax := total_cost_before_tax * 0.07
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  total_cost_after_tax / 12

theorem cost_per_serving_is_3_62 : cost_per_serving = 3.62 :=
by
  sorry

end cost_per_serving_is_3_62_l1176_117658


namespace find_a8_l1176_117637

noncomputable def geometric_sequence (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * q^(n-1)

noncomputable def sum_geom (a_1 q : ℝ) (n : ℕ) : ℝ := a_1 * (1 - q^n) / (1 - q)

theorem find_a8 (a_1 q a_2 a_5 a_8 : ℝ) (S : ℕ → ℝ) 
  (Hsum : ∀ n, S n = sum_geom a_1 q n)
  (H1 : 2 * S 9 = S 3 + S 6)
  (H2 : a_2 = geometric_sequence a_1 q 2)
  (H3 : a_5 = geometric_sequence a_1 q 5)
  (H4 : a_2 + a_5 = 4)
  (H5 : a_8 = geometric_sequence a_1 q 8) :
  a_8 = 2 :=
sorry

end find_a8_l1176_117637


namespace quick_calc_formula_l1176_117649

variables (a b A B C : ℤ)

theorem quick_calc_formula (h1 : (100 - a) * (100 - b) = (A + B - 100) * 100 + C)
                           (h2 : (100 + a) * (100 + b) = (A + B - 100) * 100 + C) :
  A = 100 ∨ A = 100 ∧ B = 100 ∨ B = 100 ∧ C = a * b :=
sorry

end quick_calc_formula_l1176_117649


namespace initial_number_of_persons_l1176_117628

-- Define the given conditions
def initial_weights (N : ℕ) : ℝ := 65 * N
def new_person_weight : ℝ := 80
def increased_average_weight : ℝ := 2.5
def weight_increase (N : ℕ) : ℝ := increased_average_weight * N

-- Mathematically equivalent proof problem
theorem initial_number_of_persons 
    (N : ℕ)
    (h : weight_increase N = new_person_weight - 65) : N = 6 :=
by
  -- Place proof here when necessary
  sorry

end initial_number_of_persons_l1176_117628


namespace frog_reaches_vertical_side_l1176_117605

def P (x y : ℕ) : ℝ := 
  if (x = 3 ∧ y = 3) then 0 -- blocked cell
  else if (x = 0 ∨ x = 5) then 1 -- vertical boundary
  else if (y = 0 ∨ y = 5) then 0 -- horizontal boundary
  else sorry -- inner probabilities to be calculated

theorem frog_reaches_vertical_side : P 2 2 = 5 / 8 :=
by sorry

end frog_reaches_vertical_side_l1176_117605


namespace proof_standard_deviation_l1176_117632

noncomputable def standard_deviation (average_age : ℝ) (max_diff_ages : ℕ) : ℝ := sorry

theorem proof_standard_deviation :
  let average_age := 31
  let max_diff_ages := 19
  standard_deviation average_age max_diff_ages = 9 := 
by
  sorry

end proof_standard_deviation_l1176_117632


namespace simplify_expression_l1176_117669

theorem simplify_expression (a: ℤ) (h₁: a ≠ 0) (h₂: a ≠ 1) (h₃: a ≠ -3) :
  (2 * a = 4) → a = 2 :=
by
  sorry

end simplify_expression_l1176_117669


namespace seating_arrangements_l1176_117623

theorem seating_arrangements :
  ∀ (chairs people : ℕ), 
  chairs = 8 → 
  people = 3 → 
  (∃ gaps : ℕ, gaps = 4) → 
  (∀ pos, pos = Nat.choose 3 4) → 
  pos = 24 :=
by
  intros chairs people h1 h2 h3 h4
  have gaps := 4
  have pos := Nat.choose 4 3
  sorry

end seating_arrangements_l1176_117623


namespace bookstore_price_change_l1176_117685

theorem bookstore_price_change (P : ℝ) (x : ℝ) (h : P > 0) : 
  (P * (1 + x / 100) * (1 - x / 100)) = 0.75 * P → x = 50 :=
by
  sorry

end bookstore_price_change_l1176_117685


namespace men_in_business_class_l1176_117616

theorem men_in_business_class (total_passengers : ℕ) (percentage_men : ℝ)
  (fraction_business_class : ℝ) (num_men_in_business_class : ℕ) 
  (h1 : total_passengers = 160) 
  (h2 : percentage_men = 0.75) 
  (h3 : fraction_business_class = 1 / 4) 
  (h4 : num_men_in_business_class = total_passengers * percentage_men * fraction_business_class) : 
  num_men_in_business_class = 30 := 
  sorry

end men_in_business_class_l1176_117616


namespace find_xyz_l1176_117686

theorem find_xyz (x y z : ℝ) (h₁ : x + 1 / y = 5) (h₂ : y + 1 / z = 2) (h₃ : z + 2 / x = 10 / 3) : x * y * z = (21 + Real.sqrt 433) / 2 :=
by
  sorry

end find_xyz_l1176_117686


namespace triangle_area_l1176_117610

theorem triangle_area (A B C : ℝ × ℝ) (hA : A = (0, 0)) (hB : B = (0, 8)) (hC : C = (10, 15)) : 
  let base := 8
  let height := 10
  let area := 1 / 2 * base * height
  area = 40.0 :=
by
  sorry

end triangle_area_l1176_117610


namespace values_of_m_and_n_l1176_117694

theorem values_of_m_and_n (m n : ℕ) (h_cond1 : 2 * m + 3 = 5 * n - 2) (h_cond2 : 5 * n - 2 < 15) : m = 5 ∧ n = 3 :=
by
  sorry

end values_of_m_and_n_l1176_117694


namespace solve_for_x_l1176_117629

-- Definitions for the problem conditions
def perimeter_triangle := 14 + 12 + 12
def perimeter_rectangle (x : ℝ) := 2 * x + 16

-- Lean 4 statement for the proof problem 
theorem solve_for_x (x : ℝ) : 
  perimeter_triangle = perimeter_rectangle x → 
  x = 11 := 
by 
  -- standard placeholders
  sorry

end solve_for_x_l1176_117629


namespace chewbacca_gum_packs_l1176_117608

theorem chewbacca_gum_packs (x : ℕ) :
  (30 - 2 * x) * (40 + 4 * x) = 1200 → x = 5 :=
by
  -- This is where the proof would go. We'll leave it as sorry for now.
  sorry

end chewbacca_gum_packs_l1176_117608


namespace even_numbers_with_specific_square_properties_l1176_117636

theorem even_numbers_with_specific_square_properties (n : ℕ) :
  (10^13 ≤ n^2 ∧ n^2 < 10^14 ∧ (n^2 % 100) / 10 = 5) → 
  (2 ∣ n ∧ 273512 > 10^5) := 
sorry

end even_numbers_with_specific_square_properties_l1176_117636


namespace first_year_students_sampled_equals_40_l1176_117600

-- Defining the conditions
def num_first_year_students := 800
def num_second_year_students := 600
def num_third_year_students := 500
def num_sampled_third_year_students := 25
def total_students := num_first_year_students + num_second_year_students + num_third_year_students

-- Proving the number of first-year students sampled
theorem first_year_students_sampled_equals_40 :
  (num_first_year_students * num_sampled_third_year_students) / num_third_year_students = 40 := by
  sorry

end first_year_students_sampled_equals_40_l1176_117600


namespace find_A_minus_C_l1176_117633

theorem find_A_minus_C (A B C : ℤ) 
  (h1 : A = B - 397)
  (h2 : A = 742)
  (h3 : B = C + 693) : 
  A - C = 296 :=
by
  sorry

end find_A_minus_C_l1176_117633


namespace mutually_coprime_divisors_l1176_117604

theorem mutually_coprime_divisors (a x y : ℕ) (h1 : a = 1944) 
  (h2 : ∃ d1 d2 d3, d1 * d2 * d3 = a ∧ gcd x y = 1 ∧ gcd x (x + y) = 1 ∧ gcd y (x + y) = 1) : 
  (x = 1 ∧ y = 2 ∧ x + y = 3) ∨ 
  (x = 1 ∧ y = 8 ∧ x + y = 9) ∨ 
  (x = 1 ∧ y = 3 ∧ x + y = 4) :=
sorry

end mutually_coprime_divisors_l1176_117604


namespace tiles_needed_l1176_117634

def tile_area : ℕ := 3 * 4
def floor_area : ℕ := 36 * 60

theorem tiles_needed : floor_area / tile_area = 180 := by
  sorry

end tiles_needed_l1176_117634


namespace grace_age_is_60_l1176_117664

def Grace : ℕ := 60
def motherAge : ℕ := 80
def grandmotherAge : ℕ := 2 * motherAge
def graceAge : ℕ := (3 / 8) * grandmotherAge

theorem grace_age_is_60 : graceAge = Grace := by
  sorry

end grace_age_is_60_l1176_117664


namespace smallest_four_consecutive_numbers_l1176_117625

theorem smallest_four_consecutive_numbers (n : ℕ) 
  (h : n * (n + 1) * (n + 2) * (n + 3) = 4574880) : n = 43 :=
sorry

end smallest_four_consecutive_numbers_l1176_117625


namespace radius_of_sphere_l1176_117626

theorem radius_of_sphere 
  (shadow_length_sphere : ℝ)
  (stick_height : ℝ)
  (stick_shadow : ℝ)
  (parallel_sun_rays : Prop) 
  (tan_θ : ℝ) 
  (h1 : tan_θ = stick_height / stick_shadow)
  (h2 : tan_θ = shadow_length_sphere / 20) :
  shadow_length_sphere / 20 = 1/4 → shadow_length_sphere = 5 := by
  sorry

end radius_of_sphere_l1176_117626


namespace printing_presses_equivalence_l1176_117684

theorem printing_presses_equivalence :
  ∃ P : ℕ, (500000 / 12) / P = (500000 / 14) / 30 ∧ P = 26 :=
by
  sorry

end printing_presses_equivalence_l1176_117684


namespace amy_total_spending_l1176_117614

def initial_tickets : ℕ := 33
def cost_per_ticket : ℝ := 1.50
def additional_tickets : ℕ := 21
def total_cost : ℝ := 81.00

theorem amy_total_spending :
  (initial_tickets * cost_per_ticket + additional_tickets * cost_per_ticket) = total_cost := 
sorry

end amy_total_spending_l1176_117614


namespace red_blue_pencil_difference_l1176_117641

theorem red_blue_pencil_difference :
  let total_pencils := 36
  let red_fraction := 5 / 9
  let blue_fraction := 5 / 12
  let red_pencils := red_fraction * total_pencils
  let blue_pencils := blue_fraction * total_pencils
  red_pencils - blue_pencils = 5 :=
by
  -- placeholder proof
  sorry

end red_blue_pencil_difference_l1176_117641


namespace coeff_x5_of_expansion_l1176_117645

theorem coeff_x5_of_expansion : 
  (Polynomial.coeff ((Polynomial.C (1 : ℤ)) * (Polynomial.X ^ 2 - Polynomial.X - Polynomial.C 2) ^ 3) 5) = -3 := 
by sorry

end coeff_x5_of_expansion_l1176_117645


namespace al_original_portion_l1176_117668

variables (a b c d : ℝ)

theorem al_original_portion :
  a + b + c + d = 1200 →
  a - 150 + 2 * b + 2 * c + 3 * d = 1800 →
  a = 450 :=
by
  intros h1 h2
  sorry

end al_original_portion_l1176_117668


namespace Carolina_Winning_Probability_Beto_Winning_Probability_Ana_Winning_Probability_l1176_117682

section
  -- Define the types of participants and the colors
  inductive Participant
  | Ana | Beto | Carolina

  inductive Color
  | blue | green

  -- Define the strategies for each participant
  inductive Strategy
  | guessBlue | guessGreen | pass

  -- Probability calculations for each strategy
  def carolinaStrategyProbability : ℚ := 1 / 8
  def betoStrategyProbability : ℚ := 1 / 2
  def anaStrategyProbability : ℚ := 3 / 4

  -- Statements to prove the probabilities
  theorem Carolina_Winning_Probability :
    carolinaStrategyProbability = 1 / 8 :=
  sorry

  theorem Beto_Winning_Probability :
    betoStrategyProbability = 1 / 2 :=
  sorry

  theorem Ana_Winning_Probability :
    anaStrategyProbability = 3 / 4 :=
  sorry
end

end Carolina_Winning_Probability_Beto_Winning_Probability_Ana_Winning_Probability_l1176_117682


namespace correct_operation_l1176_117612

variable (m n : ℝ)

-- Define the statement to be proved
theorem correct_operation : (-2 * m * n) ^ 2 = 4 * m ^ 2 * n ^ 2 :=
by sorry

end correct_operation_l1176_117612


namespace polar_curve_is_circle_l1176_117689

theorem polar_curve_is_circle (θ ρ : ℝ) (h : 4 * Real.sin θ = 5 * ρ) : 
  ∃ c : ℝ×ℝ, ∀ (x y : ℝ), x^2 + y^2 = c.1^2 + c.2^2 :=
by
  sorry

end polar_curve_is_circle_l1176_117689


namespace cuboid_edge_length_l1176_117696

theorem cuboid_edge_length (x : ℝ) (h1 : (2 * (x * 5 + x * 6 + 5 * 6)) = 148) : x = 4 :=
by 
  sorry

end cuboid_edge_length_l1176_117696


namespace students_passed_both_l1176_117677

noncomputable def F_H : ℝ := 32
noncomputable def F_E : ℝ := 56
noncomputable def F_HE : ℝ := 12
noncomputable def total_percentage : ℝ := 100

theorem students_passed_both : (total_percentage - (F_H + F_E - F_HE)) = 24 := by
  sorry

end students_passed_both_l1176_117677


namespace number_of_TVs_in_shop_c_l1176_117679

theorem number_of_TVs_in_shop_c 
  (a b d e : ℕ) 
  (avg : ℕ) 
  (num_shops : ℕ) 
  (total_TVs_in_other_shops : ℕ) 
  (total_TVs : ℕ) 
  (sum_shops : a + b + d + e = total_TVs_in_other_shops) 
  (avg_sets : avg = total_TVs / num_shops) 
  (number_shops : num_shops = 5)
  (avg_value : avg = 48)
  (T_a : a = 20) 
  (T_b : b = 30) 
  (T_d : d = 80) 
  (T_e : e = 50) 
  : (total_TVs - total_TVs_in_other_shops = 60) := 
by 
  sorry

end number_of_TVs_in_shop_c_l1176_117679


namespace walnuts_left_in_burrow_l1176_117662

-- Define the initial quantities
def boy_initial_walnuts : Nat := 6
def boy_dropped_walnuts : Nat := 1
def initial_burrow_walnuts : Nat := 12
def girl_added_walnuts : Nat := 5
def girl_eaten_walnuts : Nat := 2

-- Define the resulting quantity and the proof goal
theorem walnuts_left_in_burrow : boy_initial_walnuts - boy_dropped_walnuts + initial_burrow_walnuts + girl_added_walnuts - girl_eaten_walnuts = 20 :=
by
  sorry

end walnuts_left_in_burrow_l1176_117662


namespace cookies_in_jar_l1176_117692

-- Let C be the total number of cookies in the jar.
def C : ℕ := sorry

-- Conditions
def adults_eat_one_third (C : ℕ) : ℕ := C / 3
def children_get_each (C : ℕ) : ℕ := 20
def num_children : ℕ := 4

-- Proof statement
theorem cookies_in_jar (C : ℕ) (h1 : C / 3 = adults_eat_one_third C)
  (h2 : children_get_each C * num_children = 80)
  (h3 : 2 * (C / 3) = 80) :
  C = 120 :=
sorry

end cookies_in_jar_l1176_117692


namespace manicure_cost_per_person_l1176_117681

-- Definitions based on given conditions
def fingers_per_person : ℕ := 10
def total_fingers : ℕ := 210
def total_revenue : ℕ := 200  -- in dollars
def non_clients : ℕ := 11

-- Statement we want to prove
theorem manicure_cost_per_person :
  (total_revenue : ℚ) / (total_fingers / fingers_per_person - non_clients) = 9.52 :=
by
  sorry

end manicure_cost_per_person_l1176_117681


namespace reflect_y_axis_correct_l1176_117643

-- Define the initial coordinates of the point M
def M_orig : ℝ × ℝ := (3, 2)

-- Define the reflection function across the y-axis
def reflect_y_axis (M : ℝ × ℝ) : ℝ × ℝ :=
  (-M.1, M.2)

-- Prove that reflecting M_orig across the y-axis results in the coordinates (-3, 2)
theorem reflect_y_axis_correct : reflect_y_axis M_orig = (-3, 2) :=
  by
    -- Provide the missing steps of the proof
    sorry

end reflect_y_axis_correct_l1176_117643


namespace solve_for_M_l1176_117646

theorem solve_for_M (a b M : ℝ) (h : (a + 2 * b) ^ 2 = (a - 2 * b) ^ 2 + M) : M = 8 * a * b :=
by sorry

end solve_for_M_l1176_117646


namespace find_t_l1176_117606

noncomputable def ellipse_eq (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1

def F1 : ℝ × ℝ := (-1, 0)
def F2 : ℝ × ℝ := (1, 0)

def tangent_point (t : ℝ) : ℝ × ℝ := (t, 0)

theorem find_t :
  (∀ (A : ℝ × ℝ), ellipse_eq A.1 A.2 → 
    ∃ (C : ℝ × ℝ),
      tangent_point 2 = C ∧
      -- C is tangent to the extended line of F1A
      -- C is tangent to the extended line of F1F2
      -- C is tangent to segment AF2
      true
  ) :=
sorry

end find_t_l1176_117606


namespace find_pairs_l1176_117619

noncomputable def possibleValues (α β : ℝ) : Prop :=
  (∃ (n l : ℤ), α = 2*n*Real.pi ∧ β = -(Real.pi/3) + 2*l*Real.pi) ∨
  (∃ (n l : ℤ), α = 2*n*Real.pi ∧ β = (Real.pi/3) + 2*l*Real.pi)

theorem find_pairs (α β : ℝ) (h1 : Real.sin (α - β) = Real.sin α - Real.sin β)
  (h2 : Real.cos (α - β) = Real.cos α - Real.cos β) :
  possibleValues α β :=
sorry

end find_pairs_l1176_117619


namespace gcd_204_85_l1176_117609

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  sorry

end gcd_204_85_l1176_117609


namespace false_implies_exists_nonpositive_l1176_117697

variable (f : ℝ → ℝ)

theorem false_implies_exists_nonpositive (h : ¬ ∀ x > 0, f x > 0) : ∃ x > 0, f x ≤ 0 :=
by sorry

end false_implies_exists_nonpositive_l1176_117697


namespace sin_neg_three_halves_pi_l1176_117615

theorem sin_neg_three_halves_pi : Real.sin (-3 * Real.pi / 2) = 1 := sorry

end sin_neg_three_halves_pi_l1176_117615


namespace volume_of_right_triangle_pyramid_l1176_117655

noncomputable def pyramid_volume (H α β : ℝ) : ℝ :=
  (H^3 * Real.sin (2 * α)) / (3 * (Real.tan β)^2)

theorem volume_of_right_triangle_pyramid (H α β : ℝ) (alpha_acute : 0 < α ∧ α < π / 2) (H_pos : 0 < H) (beta_acute : 0 < β ∧ β < π / 2) :
  pyramid_volume H α β = (H^3 * Real.sin (2 * α)) / (3 * (Real.tan β)^2) := 
sorry

end volume_of_right_triangle_pyramid_l1176_117655


namespace find_abs_diff_of_average_and_variance_l1176_117675

noncomputable def absolute_difference (x y : ℝ) (a1 a2 a3 a4 a5 : ℝ) : ℝ :=
  |x - y|

theorem find_abs_diff_of_average_and_variance (x y : ℝ) (h1 : (x + y + 30 + 29 + 31) / 5 = 30)
  (h2 : ((x - 30)^2 + (y - 30)^2 + (30 - 30)^2 + (29 - 30)^2 + (31 - 30)^2) / 5 = 2) :
  absolute_difference x y 30 30 29 31 = 4 :=
by
  sorry

end find_abs_diff_of_average_and_variance_l1176_117675


namespace no_prime_divisible_by_91_l1176_117639

theorem no_prime_divisible_by_91 : ¬ ∃ p : ℕ, p > 1 ∧ Prime p ∧ 91 ∣ p :=
by
  sorry

end no_prime_divisible_by_91_l1176_117639


namespace calc_op_l1176_117672

def op (a b : ℕ) := (a + b) * (a - b)

theorem calc_op : (op 5 2)^2 = 441 := 
by 
  sorry

end calc_op_l1176_117672


namespace rainfall_second_week_january_l1176_117661

-- Define the conditions
def total_rainfall_2_weeks (rainfall_first_week rainfall_second_week : ℝ) : Prop :=
  rainfall_first_week + rainfall_second_week = 20

def rainfall_second_week_is_1_5_times_first (rainfall_first_week rainfall_second_week : ℝ) : Prop :=
  rainfall_second_week = 1.5 * rainfall_first_week

-- Define the statement to prove
theorem rainfall_second_week_january (rainfall_first_week rainfall_second_week : ℝ) :
  total_rainfall_2_weeks rainfall_first_week rainfall_second_week →
  rainfall_second_week_is_1_5_times_first rainfall_first_week rainfall_second_week →
  rainfall_second_week = 12 :=
by
  sorry

end rainfall_second_week_january_l1176_117661


namespace probability_hit_10_or_7_ring_probability_below_7_ring_l1176_117654

noncomputable def P_hit_10_ring : ℝ := 0.21
noncomputable def P_hit_9_ring : ℝ := 0.23
noncomputable def P_hit_8_ring : ℝ := 0.25
noncomputable def P_hit_7_ring : ℝ := 0.28
noncomputable def P_below_7_ring : ℝ := 0.03

theorem probability_hit_10_or_7_ring :
  P_hit_10_ring + P_hit_7_ring = 0.49 :=
  by sorry

theorem probability_below_7_ring :
  P_below_7_ring = 0.03 :=
  by sorry

end probability_hit_10_or_7_ring_probability_below_7_ring_l1176_117654
