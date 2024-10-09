import Mathlib

namespace right_triangle_num_array_l1697_169714

theorem right_triangle_num_array (n : ℕ) (hn : 0 < n) 
    (a : ℕ → ℕ → ℝ) 
    (h1 : a 1 1 = 1/4)
    (hd : ∀ i j, 0 < j → j <= i → a (i+1) 1 = a i 1 + 1/4)
    (hq : ∀ i j, 2 < i → 0 < j → j ≤ i → a i (j+1) = a i j * (1/2)) :
  a n 3 = n / 16 := 
by 
  sorry

end right_triangle_num_array_l1697_169714


namespace transform_unit_square_l1697_169721

-- Define the unit square vertices in the xy-plane
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (0, 1)

-- Transformation functions from the xy-plane to the uv-plane
def transform_u (x y : ℝ) : ℝ := x^2 - y^2
def transform_v (x y : ℝ) : ℝ := x * y

-- Vertex transformation results
def O_image : ℝ × ℝ := (transform_u 0 0, transform_v 0 0)  -- (0,0)
def A_image : ℝ × ℝ := (transform_u 1 0, transform_v 1 0)  -- (1,0)
def B_image : ℝ × ℝ := (transform_u 1 1, transform_v 1 1)  -- (0,1)
def C_image : ℝ × ℝ := (transform_u 0 1, transform_v 0 1)  -- (-1,0)

-- The Lean 4 theorem statement
theorem transform_unit_square :
  O_image = (0, 0) ∧
  A_image = (1, 0) ∧
  B_image = (0, 1) ∧
  C_image = (-1, 0) :=
  by sorry

end transform_unit_square_l1697_169721


namespace intersection_of_M_and_N_l1697_169738

def set_M : Set ℝ := {x : ℝ | x^2 - x ≥ 0}
def set_N : Set ℝ := {x : ℝ | x < 2}

theorem intersection_of_M_and_N :
  set_M ∩ set_N = {x : ℝ | x ≤ 0 ∨ (1 ≤ x ∧ x < 2)} :=
by
  sorry

end intersection_of_M_and_N_l1697_169738


namespace diamond_of_2_and_3_l1697_169739

def diamond (a b : ℕ) : ℕ := a^3 * b^2 - b + 2

theorem diamond_of_2_and_3 : diamond 2 3 = 71 := by
  sorry

end diamond_of_2_and_3_l1697_169739


namespace correct_statement_l1697_169729

def correct_input_format_1 (s : String) : Prop :=
  s = "INPUT a, b, c"

def correct_input_format_2 (s : String) : Prop :=
  s = "INPUT x="

def correct_output_format_1 (s : String) : Prop :=
  s = "PRINT A="

def correct_output_format_2 (s : String) : Prop :=
  s = "PRINT 3*2"

theorem correct_statement : (correct_input_format_1 "INPUT a; b; c" = false) ∧
                            (correct_input_format_2 "INPUT x=3" = false) ∧
                            (correct_output_format_1 "PRINT“A=4”" = false) ∧
                            (correct_output_format_2 "PRINT 3*2" = true) :=
by sorry

end correct_statement_l1697_169729


namespace servings_in_bottle_l1697_169730

theorem servings_in_bottle (total_revenue : ℕ) (price_per_serving : ℕ) (h1 : total_revenue = 98) (h2 : price_per_serving = 8) : Nat.floor (total_revenue / price_per_serving) = 12 :=
by
  sorry

end servings_in_bottle_l1697_169730


namespace initial_volume_shampoo_l1697_169787

theorem initial_volume_shampoo (V : ℝ) 
  (replace_rate : ℝ)
  (use_rate : ℝ)
  (t : ℝ) 
  (hot_sauce_fraction : ℝ) 
  (hot_sauce_amount : ℝ) : 
  replace_rate = 1/2 → 
  use_rate = 1 → 
  t = 4 → 
  hot_sauce_fraction = 0.25 → 
  hot_sauce_amount = t * replace_rate → 
  hot_sauce_amount = hot_sauce_fraction * V → 
  V = 8 :=
by 
  intro h_replace_rate h_use_rate h_t h_hot_sauce_fraction h_hot_sauce_amount h_hot_sauce_amount_eq
  sorry

end initial_volume_shampoo_l1697_169787


namespace minimum_handshakes_l1697_169790

-- Definitions
def people : ℕ := 30
def handshakes_per_person : ℕ := 3

-- Theorem statement
theorem minimum_handshakes : (people * handshakes_per_person) / 2 = 45 :=
by
  sorry

end minimum_handshakes_l1697_169790


namespace num_terms_arithmetic_sequence_is_15_l1697_169701

theorem num_terms_arithmetic_sequence_is_15 :
  ∃ n : ℕ, (∀ (a : ℤ), a = -58 + (n - 1) * 7 → a = 44) ∧ n = 15 :=
by {
  sorry
}

end num_terms_arithmetic_sequence_is_15_l1697_169701


namespace sum_of_consecutive_even_integers_l1697_169786

theorem sum_of_consecutive_even_integers
  (a1 a2 a3 a4 : ℤ)
  (h1 : a2 = a1 + 2)
  (h2 : a3 = a1 + 4)
  (h3 : a4 = a1 + 6)
  (h_sum : a1 + a3 = 146) :
  a1 + a2 + a3 + a4 = 296 :=
by sorry

end sum_of_consecutive_even_integers_l1697_169786


namespace largest_alternating_geometric_four_digit_number_l1697_169706

theorem largest_alternating_geometric_four_digit_number :
  ∃ (a b c d : ℕ), 
  (9 = 2 * b) ∧ (b = 2 * c) ∧ (a = 3) ∧ (9 * d = b * c) ∧ 
  (a > b) ∧ (b < c) ∧ (c > d) ∧ (1000 * a + 100 * b + 10 * c + d = 9632) := sorry

end largest_alternating_geometric_four_digit_number_l1697_169706


namespace three_digit_numbers_l1697_169794

theorem three_digit_numbers (n : ℕ) :
  n = 4 ↔ ∃ (x y : ℕ), 
  (100 ≤ 101 * x + 10 * y ∧ 101 * x + 10 * y < 1000) ∧ 
  (x ≠ 0 ∧ x ≠ 5) ∧ 
  (2 * x + y = 15) ∧ 
  (y < 10) :=
by { sorry }

end three_digit_numbers_l1697_169794


namespace sequence_arithmetic_mean_l1697_169765

theorem sequence_arithmetic_mean (a b c d e f g : ℝ)
  (h1 : b = (a + c) / 2)
  (h2 : c = (b + d) / 2)
  (h3 : d = (c + e) / 2)
  (h4 : e = (d + f) / 2)
  (h5 : f = (e + g) / 2) :
  d = (a + g) / 2 :=
sorry

end sequence_arithmetic_mean_l1697_169765


namespace geometric_sequence_third_term_l1697_169778

theorem geometric_sequence_third_term (a1 a5 a3 : ℕ) (r : ℝ) 
  (h1 : a1 = 4) 
  (h2 : a5 = 1296) 
  (h3 : a5 = a1 * r^4)
  (h4 : a3 = a1 * r^2) : 
  a3 = 36 := 
by 
  sorry

end geometric_sequence_third_term_l1697_169778


namespace rainfall_thursday_l1697_169784

theorem rainfall_thursday : 
  let monday_rain := 0.9
  let tuesday_rain := monday_rain - 0.7
  let wednesday_rain := tuesday_rain * 1.5
  let thursday_rain := wednesday_rain * 0.8
  thursday_rain = 0.24 :=
by
  sorry

end rainfall_thursday_l1697_169784


namespace part1_part2_l1697_169724

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a^2| + |x - a - 1|

-- Question 1: Prove that f(x) ≥ 3/4
theorem part1 (x a : ℝ) : f x a ≥ 3 / 4 := 
sorry

-- Question 2: Given f(4) < 13, find the range of a
theorem part2 (a : ℝ) (h : f 4 a < 13) : -2 < a ∧ a < 3 := 
sorry

end part1_part2_l1697_169724


namespace tim_kittens_count_l1697_169788

def initial_kittens : Nat := 6
def kittens_given_to_jessica : Nat := 3
def kittens_received_from_sara : Nat := 9

theorem tim_kittens_count : initial_kittens - kittens_given_to_jessica + kittens_received_from_sara = 12 :=
by
  sorry

end tim_kittens_count_l1697_169788


namespace sequence_problem_l1697_169797

/-- Given sequence a_n with specific values for a_2 and a_4 and the assumption that a_(n+1)
    is a geometric sequence, prove that a_6 equals 63. -/
theorem sequence_problem 
  {a : ℕ → ℝ} 
  (h1 : a 2 = 3) 
  (h2 : a 4 = 15) 
  (h3 : ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n ∧ q^2 = 4) : 
  a 6 = 63 := by
  sorry

end sequence_problem_l1697_169797


namespace find_common_ratio_l1697_169756

variable {a : ℕ → ℝ} {q : ℝ}

-- Define that a is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Define the sum of the first n terms of the sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

theorem find_common_ratio
  (h1 : is_geometric_sequence a q)
  (h2 : 0 < q)
  (h3 : a 1 * a 3 = 1)
  (h4 : sum_first_n_terms a 3 = 7) :
  q = 1 / 2 :=
sorry

end find_common_ratio_l1697_169756


namespace decreasing_interval_l1697_169755

theorem decreasing_interval (f : ℝ → ℝ) (hf : ∀ x, f x = x^2 - 2 * x) :
  {x | deriv f x < 0} = {x | x < 1} :=
by
  sorry

end decreasing_interval_l1697_169755


namespace factorial_sum_power_of_two_l1697_169768

theorem factorial_sum_power_of_two (a b c : ℕ) (hac : 0 < a) (hbc : 0 < b) (hcc : 0 < c) :
  a! + b! = 2 ^ c! ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 2 ∧ b = 2 ∧ c = 2) :=
by
  sorry

end factorial_sum_power_of_two_l1697_169768


namespace sum_of_coordinates_B_l1697_169760

theorem sum_of_coordinates_B :
  ∃ (x y : ℝ), (3, 5) = ((x + 6) / 2, (y + 8) / 2) ∧ x + y = 2 := by
  sorry

end sum_of_coordinates_B_l1697_169760


namespace field_perimeter_l1697_169702

noncomputable def outer_perimeter (posts : ℕ) (post_width_inches : ℝ) (spacing_feet : ℝ) : ℝ :=
  let posts_per_side := posts / 4
  let gaps_per_side := posts_per_side - 1
  let post_width_feet := post_width_inches / 12
  let side_length := gaps_per_side * spacing_feet + posts_per_side * post_width_feet
  4 * side_length

theorem field_perimeter : 
  outer_perimeter 32 5 4 = 125 + 1/3 := 
by
  sorry

end field_perimeter_l1697_169702


namespace minimum_value_of_expression_l1697_169791

noncomputable def minimum_value_expression (x y z : ℝ) : ℝ :=
  1 / ((1 - x) * (1 - y) * (1 - z)) + 1 / ((1 + x) * (1 + y) * (1 + z))

theorem minimum_value_of_expression : ∀ (x y z : ℝ), -1 < x ∧ x < 0 ∧ -1 < y ∧ y < 0 ∧ -1 < z ∧ z < 0 → 
  minimum_value_expression x y z ≥ 2 := 
by
  intro x y z h
  sorry

end minimum_value_of_expression_l1697_169791


namespace machine_transportation_l1697_169776

theorem machine_transportation (x y : ℕ) 
  (h1 : x + 6 - y = 10) 
  (h2 : 400 * x + 800 * (20 - x) + 300 * (6 - y) + 500 * y = 16000) : 
  x = 5 ∧ y = 1 := 
sorry

end machine_transportation_l1697_169776


namespace find_pairs_l1697_169719

def regions_divided (h s : ℕ) : ℕ :=
  1 + s * (s + 1) / 2 + h * (s + 1)

theorem find_pairs (h s : ℕ) :
  regions_divided h s = 1992 →
  (h = 995 ∧ s = 1) ∨ (h = 176 ∧ s = 10) ∨ (h = 80 ∧ s = 21) :=
by
  sorry

end find_pairs_l1697_169719


namespace determine_C_for_identity_l1697_169747

theorem determine_C_for_identity :
  (∀ (x : ℝ), (1/2 * (Real.sin x)^2 + C = -1/4 * Real.cos (2 * x))) → C = -1/4 :=
by
  sorry

end determine_C_for_identity_l1697_169747


namespace how_many_cheburashkas_erased_l1697_169745

theorem how_many_cheburashkas_erased 
  (total_krakozyabras : ℕ)
  (characters_per_row_initial : ℕ) 
  (total_characters_initial : ℕ)
  (total_cheburashkas : ℕ)
  (total_rows : ℕ := 2)
  (total_krakozyabras := 29) :
  total_cheburashkas = 11 :=
by
  sorry

end how_many_cheburashkas_erased_l1697_169745


namespace shifted_linear_func_is_2x_l1697_169708

-- Define the initial linear function
def linear_func (x : ℝ) : ℝ := 2 * x - 3

-- Define the shifted linear function
def shifted_linear_func (x : ℝ) : ℝ := linear_func x + 3

theorem shifted_linear_func_is_2x (x : ℝ) : shifted_linear_func x = 2 * x := by
  -- Proof would go here, but we use sorry to skip it
  sorry

end shifted_linear_func_is_2x_l1697_169708


namespace holds_for_even_positive_l1697_169715

variable {n : ℕ}
variable (p : ℕ → Prop)

-- Conditions
axiom base_case : p 2
axiom inductive_step : ∀ k, p k → p (k + 2)

-- Theorem to prove
theorem holds_for_even_positive (n : ℕ) (h : n > 0) (h_even : n % 2 = 0) : p n :=
sorry

end holds_for_even_positive_l1697_169715


namespace travel_time_l1697_169732

-- Definitions of the conditions
variables (x : ℝ) (speed_elder speed_younger : ℝ)
variables (time_elder_total time_younger_total : ℝ)

def elder_speed_condition : Prop := speed_elder = x
def younger_speed_condition : Prop := speed_younger = x - 4
def elder_distance : Prop := 42 / speed_elder + 1 = time_elder_total
def younger_distance : Prop := 42 / speed_younger + 1 / 3 = time_younger_total

-- The main theorem we want to prove
theorem travel_time : ∀ (x : ℝ), 
  elder_speed_condition x speed_elder → 
  younger_speed_condition x speed_younger → 
  elder_distance speed_elder time_elder_total → 
  younger_distance speed_younger time_younger_total → 
  time_elder_total = time_younger_total ∧ time_elder_total = (10 / 3) :=
sorry

end travel_time_l1697_169732


namespace books_sold_in_store_on_saturday_l1697_169734

namespace BookshopInventory

def initial_inventory : ℕ := 743
def saturday_online_sales : ℕ := 128
def sunday_online_sales : ℕ := 162
def shipment_received : ℕ := 160
def final_inventory : ℕ := 502

-- Define the total number of books sold
def total_books_sold (S : ℕ) : ℕ := S + saturday_online_sales + 2 * S + sunday_online_sales

-- Net change in inventory equals total books sold minus shipment received
def net_change_in_inventory (S : ℕ) : ℕ := total_books_sold S - shipment_received

-- Prove that the difference between initial and final inventories equals the net change in inventory
theorem books_sold_in_store_on_saturday : ∃ S : ℕ, net_change_in_inventory S = initial_inventory - final_inventory ∧ S = 37 :=
by
  sorry

end BookshopInventory

end books_sold_in_store_on_saturday_l1697_169734


namespace number_of_subsets_l1697_169727

-- Define the set
def my_set : Set ℕ := {1, 2, 3}

-- Theorem statement
theorem number_of_subsets : Finset.card (Finset.powerset {1, 2, 3}) = 8 :=
by
  sorry

end number_of_subsets_l1697_169727


namespace total_numbers_l1697_169792

theorem total_numbers (n : ℕ) (a : ℕ → ℝ) 
  (h1 : (a 0 + a 1 + a 2 + a 3) / 4 = 25)
  (h2 : (a (n - 3) + a (n - 2) + a (n - 1)) / 3 = 35)
  (h3 : a 3 = 25)
  (h4 : (Finset.sum (Finset.range n) a) / n = 30) :
  n = 6 :=
sorry

end total_numbers_l1697_169792


namespace lottery_probability_correct_l1697_169777

/-- The binomial coefficient function -/
def binom (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of matching MegaBall and WinnerBalls in the lottery -/
noncomputable def lottery_probability : ℚ :=
  let megaBall_prob := (1 : ℚ) / 30
  let winnerBalls_prob := (1 : ℚ) / binom 45 6
  megaBall_prob * winnerBalls_prob

theorem lottery_probability_correct : lottery_probability = (1 : ℚ) / 244351800 := by
  sorry

end lottery_probability_correct_l1697_169777


namespace inequality_problem_l1697_169785

variable {a b : ℕ}

theorem inequality_problem (a : ℕ) (b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_neq_1_a : a ≠ 1) (h_neq_1_b : b ≠ 1) :
  ((a^5 - 1:ℚ) / (a^4 - 1)) * ((b^5 - 1) / (b^4 - 1)) > (25 / 64 : ℚ) * (a + 1) * (b + 1) :=
by
  sorry

end inequality_problem_l1697_169785


namespace max_value_l1697_169711

theorem max_value (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 2) : 
  2 * x * y + 2 * y * z * Real.sqrt 3 ≤ 4 :=
sorry

end max_value_l1697_169711


namespace max_value_of_expr_l1697_169799

-- Define the initial conditions and expression 
def initial_ones (n : ℕ) := List.replicate n 1

-- Given that we place "+" or ")(" between consecutive ones
def max_possible_value (n : ℕ) : ℕ := sorry

theorem max_value_of_expr : max_possible_value 2013 = 3 ^ 671 := 
sorry

end max_value_of_expr_l1697_169799


namespace sum_of_palindromic_primes_less_than_70_l1697_169709

def is_prime (n : ℕ) : Prop := Nat.Prime n

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ acc d => acc * 10 + d) 0

def is_palindromic_prime (n : ℕ) : Prop :=
  is_prime n ∧ is_prime (reverse_digits n)

theorem sum_of_palindromic_primes_less_than_70 :
  let palindromic_primes := [11, 13, 31, 37]
  (∀ p ∈ palindromic_primes, is_palindromic_prime p ∧ p < 70) →
  palindromic_primes.sum = 92 :=
by
  sorry

end sum_of_palindromic_primes_less_than_70_l1697_169709


namespace bus_speed_l1697_169723

theorem bus_speed (x y z : ℕ) (hx : x < 10) (hy : y < 10) (hz : z < 10)
    (h1 : 9 * (11 * y - x) = 5 * z)
    (h2 : z = 9) :
    ∀ speed, speed = 45 :=
by
  sorry

end bus_speed_l1697_169723


namespace positive_difference_perimeters_l1697_169742

theorem positive_difference_perimeters :
  let w1 := 3
  let h1 := 2
  let w2 := 6
  let h2 := 1
  let P1 := 2 * (w1 + h1)
  let P2 := 2 * (w2 + h2)
  P2 - P1 = 4 := by
  sorry

end positive_difference_perimeters_l1697_169742


namespace mary_chopped_tables_l1697_169741

-- Define the constants based on the conditions
def chairs_sticks := 6
def tables_sticks := 9
def stools_sticks := 2
def burn_rate := 5

-- Define the quantities of items Mary chopped up
def chopped_chairs := 18
def chopped_stools := 4
def warm_hours := 34
def sticks_from_chairs := chopped_chairs * chairs_sticks
def sticks_from_stools := chopped_stools * stools_sticks
def total_needed_sticks := warm_hours * burn_rate
def sticks_from_tables (chopped_tables : ℕ) := chopped_tables * tables_sticks

-- Define the proof goal
theorem mary_chopped_tables : ∃ chopped_tables, sticks_from_chairs + sticks_from_stools + sticks_from_tables chopped_tables = total_needed_sticks ∧ chopped_tables = 6 :=
by
  sorry

end mary_chopped_tables_l1697_169741


namespace problem_part1_problem_part2_l1697_169728

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6)

theorem problem_part1 :
  f (Real.pi / 12) = 3 * Real.sqrt 3 / 2 :=
by
  sorry

theorem problem_part2 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < Real.pi / 2) :
  Real.sin θ = 4 / 5 →
  f (5 * Real.pi / 12 - θ) = 72 / 25 :=
by
  sorry

end problem_part1_problem_part2_l1697_169728


namespace factorial_equation_solution_unique_l1697_169769

theorem factorial_equation_solution_unique :
  ∀ a b c : ℕ, (0 < a ∧ 0 < b ∧ 0 < c) →
  (a.factorial * b.factorial = a.factorial + b.factorial + c.factorial) →
  (a = 3 ∧ b = 3 ∧ c = 4) := 
by
  intros a b c h_positive h_eq
  sorry

end factorial_equation_solution_unique_l1697_169769


namespace sales_neither_notebooks_nor_markers_l1697_169754

theorem sales_neither_notebooks_nor_markers (percent_notebooks percent_markers percent_staplers : ℝ) 
  (h1 : percent_notebooks = 25)
  (h2 : percent_markers = 40)
  (h3 : percent_staplers = 15) : 
  percent_staplers + (100 - (percent_notebooks + percent_markers + percent_staplers)) = 35 :=
by
  sorry

end sales_neither_notebooks_nor_markers_l1697_169754


namespace max_value_expression_l1697_169707

theorem max_value_expression (θ : ℝ) : 
  2 ≤ 5 + 3 * Real.sin θ ∧ 5 + 3 * Real.sin θ ≤ 8 → 
  (∃ θ, (14 / (5 + 3 * Real.sin θ)) = 7) := 
sorry

end max_value_expression_l1697_169707


namespace cubic_inequality_l1697_169753

theorem cubic_inequality (x : ℝ) : x^3 - 4*x^2 + 4*x < 0 ↔ x < 0 :=
by
  sorry

end cubic_inequality_l1697_169753


namespace find_a_range_l1697_169775

def f (a x : ℝ) : ℝ := x^2 + a * x

theorem find_a_range (a : ℝ) :
  (∃ x : ℝ, f a (f a x) ≤ f a x) → (a ≤ 0 ∨ a ≥ 2) :=
by
  sorry

end find_a_range_l1697_169775


namespace solve_ab_c_eq_l1697_169795

theorem solve_ab_c_eq (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_eq : 11^a + 3^b = c^2) :
  a = 4 ∧ b = 5 ∧ c = 122 :=
by
  sorry

end solve_ab_c_eq_l1697_169795


namespace Jason_spent_correct_amount_l1697_169770

def flute_cost : ℝ := 142.46
def music_stand_cost : ℝ := 8.89
def song_book_cost : ℝ := 7.00
def total_cost : ℝ := 158.35

theorem Jason_spent_correct_amount :
  flute_cost + music_stand_cost + song_book_cost = total_cost :=
by
  sorry

end Jason_spent_correct_amount_l1697_169770


namespace div_by_6_for_all_k_l1697_169713

def b_n_sum_of_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6

theorem div_by_6_for_all_k : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 50 → (b_n_sum_of_squares k) % 6 = 0 :=
by
  intros k hk
  sorry

end div_by_6_for_all_k_l1697_169713


namespace minimum_throws_to_ensure_same_sum_twice_l1697_169762

-- Define a fair six-sided die.
def fair_six_sided_die := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

-- Define the sum of four fair six-sided dice.
def sum_of_four_dice (d1 d2 d3 d4 : fair_six_sided_die) : ℕ :=
  d1.val + d2.val + d3.val + d4.val

-- Proof problem: Prove that the minimum number of throws required to ensure the same sum is rolled twice is 22.
theorem minimum_throws_to_ensure_same_sum_twice : 22 = 22 := by
  sorry

end minimum_throws_to_ensure_same_sum_twice_l1697_169762


namespace part1_part2_l1697_169783

noncomputable def A_m (m : ℕ) (k : ℕ) : ℕ := (2 * k - 1) * m + k

theorem part1 (m : ℕ) (hm : m ≥ 2) :
  ∃ a : ℕ, 1 ≤ a ∧ a < m ∧ (∃ k : ℕ, 2^a = A_m m k) ∨ (∃ k : ℕ, 2^a + 1 = A_m m k) :=
sorry

theorem part2 {m : ℕ} (hm : m ≥ 2) 
  (a b : ℕ) (ha : ∃ k, 2^a = A_m m k) (hb : ∃ k, 2^b + 1 = A_m m k)
  (hmin_a : ∀ x, (∃ k, 2^x = A_m m k) → a ≤ x) 
  (hmin_b : ∀ y, (∃ k, 2^y + 1 = A_m m k) → b ≤ y) :
  a = 2 * b + 1 :=
sorry

end part1_part2_l1697_169783


namespace common_z_values_l1697_169710

theorem common_z_values (z : ℝ) :
  (∃ x : ℝ, x^2 + z^2 = 9 ∧ x^2 = 4*z - 5) ↔ (z = -2 + 3*Real.sqrt 2 ∨ z = -2 - 3*Real.sqrt 2) := 
sorry

end common_z_values_l1697_169710


namespace cubic_km_to_cubic_m_l1697_169752

theorem cubic_km_to_cubic_m (km_to_m : 1 = 1000) : (1 : ℝ) ^ 3 = (1000 : ℝ) ^ 3 :=
by sorry

end cubic_km_to_cubic_m_l1697_169752


namespace smallest_A_divided_by_6_has_third_of_original_factors_l1697_169781

theorem smallest_A_divided_by_6_has_third_of_original_factors:
  ∃ A: ℕ, A > 0 ∧ (∃ a b: ℕ, A = 2^a * 3^b ∧ (a + 1) * (b + 1) = 3 * a * b) ∧ A = 12 :=
by
  sorry

end smallest_A_divided_by_6_has_third_of_original_factors_l1697_169781


namespace fill_table_with_numbers_l1697_169737

-- Define the main theorem based on the conditions and question.
theorem fill_table_with_numbers (numbers : Finset ℤ) (table : ℕ → ℕ → ℤ)
  (h_numbers_card : numbers.card = 100)
  (h_sum_1x3_horizontal : ∀ i j, (table i j + table i (j + 1) + table i (j + 2) ∈ numbers))
  (h_sum_1x3_vertical : ∀ i j, (table i j + table (i + 1) j + table (i + 2) j ∈ numbers)):
  ∃ (t : ℕ → ℕ → ℤ), (∀ k, 1 ≤ k ∧ k ≤ 6 → ∃ i j, t i j = k) :=
sorry

end fill_table_with_numbers_l1697_169737


namespace trig_identity_example_l1697_169746

theorem trig_identity_example :
  (Real.sin (36 * Real.pi / 180) * Real.cos (6 * Real.pi / 180) -
   Real.sin (54 * Real.pi / 180) * Real.cos (84 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end trig_identity_example_l1697_169746


namespace debby_ate_candy_l1697_169748

theorem debby_ate_candy (initial_candy : ℕ) (remaining_candy : ℕ) (debby_initial : initial_candy = 12) (debby_remaining : remaining_candy = 3) : initial_candy - remaining_candy = 9 :=
by
  sorry

end debby_ate_candy_l1697_169748


namespace gcd_45345_34534_l1697_169764

theorem gcd_45345_34534 : Nat.gcd 45345 34534 = 71 := by
  sorry

end gcd_45345_34534_l1697_169764


namespace figure_total_area_l1697_169761

theorem figure_total_area (a : ℝ) (h : a^2 - (3/2 * a^2) = 0.6) : 
  5 * a^2 = 6 :=
by
  sorry

end figure_total_area_l1697_169761


namespace optimal_tablet_combination_exists_l1697_169731

/-- Define the daily vitamin requirement structure --/
structure Vitamins (A B C D : ℕ)

theorem optimal_tablet_combination_exists {x y : ℕ} :
  (∃ (x y : ℕ), 
    (3 * x ≥ 3) ∧ (x + y ≥ 9) ∧ (x + 3 * y ≥ 15) ∧ (2 * y ≥ 2) ∧
    (x + y = 9) ∧ 
    (20 * x + 60 * y = 3) ∧ 
    (x + 2 * y = 12) ∧ 
    (x = 6 ∧ y = 3)) := 
  by
  sorry

end optimal_tablet_combination_exists_l1697_169731


namespace solve_inequality_l1697_169740

theorem solve_inequality :
  {x : ℝ | (x - 3)*(x - 4)*(x - 5) / ((x - 2)*(x - 6)*(x - 7)) > 0} =
  {x : ℝ | x < 2} ∪ {x : ℝ | 4 < x ∧ x < 5} ∪ {x : ℝ | 6 < x ∧ x < 7} ∪ {x : ℝ | 7 < x} :=
by
  sorry

end solve_inequality_l1697_169740


namespace pet_shop_legs_l1697_169772

theorem pet_shop_legs :
  let birds := 3
  let dogs := 5
  let snakes := 4
  let spiders := 1
  let bird_legs := 2
  let dog_legs := 4
  let snake_legs := 0
  let spider_legs := 8
  birds * bird_legs + dogs * dog_legs + snakes * snake_legs + spiders * spider_legs = 34 := 
by
  let birds := 3
  let dogs := 5
  let snakes := 4
  let spiders := 1
  let bird_legs := 2
  let dog_legs := 4
  let snake_legs := 0
  let spider_legs := 8
  sorry

end pet_shop_legs_l1697_169772


namespace ending_number_is_54_l1697_169780

def first_even_after_15 : ℕ := 16
def evens_between (a b : ℕ) : ℕ := (b - first_even_after_15) / 2 + 1

theorem ending_number_is_54 (n : ℕ) (h : evens_between 15 n = 20) : n = 54 :=
by {
  sorry
}

end ending_number_is_54_l1697_169780


namespace number_of_players_l1697_169705
-- Importing the necessary library

-- Define the number of games formula for the tournament
def number_of_games (n : ℕ) : ℕ := n * (n - 1) * 2

-- The theorem to prove the number of players given the conditions
theorem number_of_players (n : ℕ) (h : number_of_games n = 306) : n = 18 :=
by
  sorry

end number_of_players_l1697_169705


namespace symmetric_line_equation_l1697_169717

theorem symmetric_line_equation (x y : ℝ) (h : 4 * x - 3 * y + 5 = 0):
  4 * x + 3 * y + 5 = 0 :=
sorry

end symmetric_line_equation_l1697_169717


namespace range_of_function_l1697_169763

theorem range_of_function :
  ∃ (S : Set ℝ), (∀ x : ℝ, (1 / 2)^(x^2 - 2) ∈ S) ∧ S = Set.Ioc 0 4 := by
  sorry

end range_of_function_l1697_169763


namespace avg_weight_of_13_children_l1697_169796

-- Definitions based on conditions:
def boys_avg_weight := 160
def boys_count := 8
def girls_avg_weight := 130
def girls_count := 5

-- Calculation to determine the total weights
def boys_total_weight := boys_avg_weight * boys_count
def girls_total_weight := girls_avg_weight * girls_count

-- Combined total weight
def total_weight := boys_total_weight + girls_total_weight

-- Average weight calculation
def children_count := boys_count + girls_count
def avg_weight := total_weight / children_count

-- The theorem to prove:
theorem avg_weight_of_13_children : avg_weight = 148 := by
  sorry

end avg_weight_of_13_children_l1697_169796


namespace problem_l1697_169726

theorem problem (a b : ℤ) (ha : a = 4) (hb : b = -3) : -2 * a - b^2 + 2 * a * b = -41 := 
by
  -- Provide proof here
  sorry

end problem_l1697_169726


namespace polynomial_integer_roots_l1697_169703

theorem polynomial_integer_roots
  (b c : ℤ)
  (x1 x2 x1' x2' : ℤ)
  (h_eq1 : x1 * x2 > 0)
  (h_eq2 : x1' * x2' > 0)
  (h_eq3 : x1^2 + b * x1 + c = 0)
  (h_eq4 : x2^2 + b * x2 + c = 0)
  (h_eq5 : x1'^2 + c * x1' + b = 0)
  (h_eq6 : x2'^2 + c * x2' + b = 0)
  : x1 < 0 ∧ x2 < 0 ∧ b - 1 ≤ c ∧ c ≤ b + 1 ∧ 
    ((b = 5 ∧ c = 6) ∨ (b = 6 ∧ c = 5) ∨ (b = 4 ∧ c = 4)) := 
sorry

end polynomial_integer_roots_l1697_169703


namespace jenny_eggs_in_each_basket_l1697_169722

theorem jenny_eggs_in_each_basket (n : ℕ) (h1 : 30 % n = 0) (h2 : 45 % n = 0) (h3 : n ≥ 5) : n = 15 :=
sorry

end jenny_eggs_in_each_basket_l1697_169722


namespace smallest_positive_n_l1697_169712

theorem smallest_positive_n (n : ℕ) (h : 19 * n ≡ 789 [MOD 11]) : n = 1 := 
by
  sorry

end smallest_positive_n_l1697_169712


namespace probability_four_ones_in_five_rolls_l1697_169751

-- Define the probability of rolling a 1 on a fair six-sided die
def prob_one_roll_one : ℚ := 1 / 6

-- Define the probability of not rolling a 1 on a fair six-sided die
def prob_one_roll_not_one : ℚ := 5 / 6

-- Define the number of successes needed, here 4 ones in 5 rolls
def num_successes : ℕ := 4

-- Define the total number of trials, here 5 rolls
def num_trials : ℕ := 5

-- Binomial probability calculation for 4 successes in 5 trials with probability of success prob_one_roll_one
def binomial_prob (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_four_ones_in_five_rolls : binomial_prob num_trials num_successes prob_one_roll_one = 25 / 7776 := 
by
  sorry

end probability_four_ones_in_five_rolls_l1697_169751


namespace solution_set_l1697_169774

def f (x : ℝ) : ℝ := |x - 2| - |x - 5|

theorem solution_set :
  {x | f x ≥ x^2 - 8 * x + 15} = {2} ∪ {x | x > 6} :=
by
  sorry

end solution_set_l1697_169774


namespace inequality_solution_l1697_169725

theorem inequality_solution {x : ℝ} :
  {x | (2 * x - 8) * (x - 4) / x ≥ 0} = {x | x < 0} ∪ {x | x > 0} :=
by
  sorry

end inequality_solution_l1697_169725


namespace find_n_l1697_169793

theorem find_n (n : ℤ) : -180 ≤ n ∧ n ≤ 180 ∧ (Real.sin (n * Real.pi / 180) = Real.cos (690 * Real.pi / 180)) → n = 60 :=
by
  intro h
  sorry

end find_n_l1697_169793


namespace grid_rows_l1697_169744

theorem grid_rows (R : ℕ) :
  let squares_per_row := 15
  let red_squares := 4 * 6
  let blue_squares := 4 * squares_per_row
  let green_squares := 66
  let total_squares := red_squares + blue_squares + green_squares 
  total_squares = squares_per_row * R →
  R = 10 :=
by
  intros
  sorry

end grid_rows_l1697_169744


namespace rectangle_to_rhombus_l1697_169743

def is_rectangle (A B C D : ℝ × ℝ) : Prop :=
  A.1 = D.1 ∧ D.2 = C.2 ∧ C.1 = B.1 ∧ B.2 = A.2

def is_triangle (A B C : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2) ≠ 0

def is_rhombus (A B C D : ℝ × ℝ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A

theorem rectangle_to_rhombus (A B C D : ℝ × ℝ) (h1 : is_rectangle A B C D) :
  ∃ X Y Z W : ℝ × ℝ, is_triangle A B C ∧ is_triangle A D C ∧ is_rhombus X Y Z W :=
by
  sorry

end rectangle_to_rhombus_l1697_169743


namespace problem1_problem2_l1697_169757

noncomputable def f (x a : ℝ) := |x - 1| + |x - a|

theorem problem1 (x : ℝ) : f x (-1) ≥ 3 ↔ x ≤ -1.5 ∨ x ≥ 1.5 :=
sorry

theorem problem2 (a : ℝ) : (∀ x, f x a ≥ 2) ↔ (a = 3 ∨ a = -1) :=
sorry

end problem1_problem2_l1697_169757


namespace xiao_ming_climb_stairs_8_l1697_169720

def fibonacci (n : ℕ) : ℕ :=
  match n with
  | 0     => 0
  | 1     => 1
  | (n+2) => fibonacci n + fibonacci (n + 1)

theorem xiao_ming_climb_stairs_8 :
  fibonacci 8 = 34 :=
sorry

end xiao_ming_climb_stairs_8_l1697_169720


namespace total_people_served_l1697_169771

variable (total_people : ℕ)
variable (people_not_buy_coffee : ℕ := 10)

theorem total_people_served (H : (2 / 5 : ℚ) * total_people = people_not_buy_coffee) : total_people = 25 := 
by
  sorry

end total_people_served_l1697_169771


namespace saving_is_zero_cents_l1697_169736

-- Define the in-store and online prices
def in_store_price : ℝ := 129.99
def online_payment_per_installment : ℝ := 29.99
def shipping_and_handling : ℝ := 11.99

-- Define the online total price
def online_total_price : ℝ := 4 * online_payment_per_installment + shipping_and_handling

-- Define the saving in cents
def saving_in_cents : ℝ := (in_store_price - online_total_price) * 100

-- State the theorem to prove the number of cents saved
theorem saving_is_zero_cents : saving_in_cents = 0 := by
  sorry

end saving_is_zero_cents_l1697_169736


namespace range_of_a_l1697_169735

-- Define the function f(x) and its condition
def f (x a : ℝ) : ℝ := x^2 + (a + 2) * x + (a - 1)

-- Given condition: f(-1, a) = -2
def condition (a : ℝ) : Prop := f (-1) a = -2

-- Requirement for the domain of g(x) = ln(f(x) + 3) being ℝ
def domain_requirement (a : ℝ) : Prop := ∀ x : ℝ, f x a + 3 > 0

-- Main theorem to prove the range of a
theorem range_of_a : {a : ℝ // condition a ∧ domain_requirement a} = {a : ℝ // -2 < a ∧ a < 2} :=
by sorry

end range_of_a_l1697_169735


namespace travel_allowance_increase_20_l1697_169782

def employees_total : ℕ := 480
def employees_no_increase : ℕ := 336
def employees_salary_increase_percentage : ℕ := 10

def employees_salary_increase : ℕ :=
(employees_salary_increase_percentage * employees_total) / 100

def employees_travel_allowance_increase : ℕ :=
employees_total - (employees_salary_increase + employees_no_increase)

def travel_allowance_increase_percentage : ℕ :=
(employees_travel_allowance_increase * 100) / employees_total

theorem travel_allowance_increase_20 :
  travel_allowance_increase_percentage = 20 :=
by sorry

end travel_allowance_increase_20_l1697_169782


namespace john_total_amount_l1697_169779

def grandpa_amount : ℕ := 30
def grandma_amount : ℕ := 3 * grandpa_amount
def aunt_amount : ℕ := 3 / 2 * grandpa_amount
def uncle_amount : ℕ := 2 / 3 * grandma_amount

def total_amount : ℕ :=
  grandpa_amount + grandma_amount + aunt_amount + uncle_amount

theorem john_total_amount : total_amount = 225 := by sorry

end john_total_amount_l1697_169779


namespace largest_non_factor_product_of_factors_of_100_l1697_169716

theorem largest_non_factor_product_of_factors_of_100 :
  ∃ x y : ℕ, 
  (x ≠ y) ∧ 
  (0 < x ∧ 0 < y) ∧ 
  (x ∣ 100 ∧ y ∣ 100) ∧ 
  ¬(x * y ∣ 100) ∧ 
  (∀ a b : ℕ, 
    (a ≠ b) ∧ 
    (0 < a ∧ 0 < b) ∧ 
    (a ∣ 100 ∧ b ∣ 100) ∧ 
    ¬(a * b ∣ 100) → 
    (x * y) ≥ (a * b)) ∧ 
  (x * y) = 40 :=
by
  sorry

end largest_non_factor_product_of_factors_of_100_l1697_169716


namespace mike_picked_peaches_l1697_169700

def initial_peaches : ℕ := 34
def total_peaches : ℕ := 86

theorem mike_picked_peaches : total_peaches - initial_peaches = 52 :=
by
  sorry

end mike_picked_peaches_l1697_169700


namespace z_pow12_plus_inv_z_pow12_l1697_169718

open Complex

theorem z_pow12_plus_inv_z_pow12 (z: ℂ) (h: z + z⁻¹ = 2 * cos (10 * Real.pi / 180)) :
  z^12 + z⁻¹^12 = -1 := by
  sorry

end z_pow12_plus_inv_z_pow12_l1697_169718


namespace solve_problem_l1697_169749

-- Define the polynomial p(x)
noncomputable def p (x : ℂ) : ℂ := x^2 - x + 1

-- Define the root condition
def is_root (α : ℂ) : Prop := p (p (p (p α))) = 0

-- Define the expression to evaluate
noncomputable def expression (α : ℂ) : ℂ := (p α - 1) * p α * p (p α) * p (p (p α))

-- State the theorem asserting the required equality
theorem solve_problem (α : ℂ) (hα : is_root α) : expression α = -1 :=
sorry

end solve_problem_l1697_169749


namespace number_of_correct_answers_is_95_l1697_169798

variable (x y : ℕ) -- Define x as the number of correct answers and y as the number of wrong answers

-- Define the conditions
axiom h1 : x + y = 150
axiom h2 : 5 * x - 2 * y = 370

-- State the goal we want to prove
theorem number_of_correct_answers_is_95 : x = 95 :=
by
  sorry

end number_of_correct_answers_is_95_l1697_169798


namespace sum_of_gcd_and_lcm_of_180_and_4620_l1697_169758

def gcd_180_4620 : ℕ := Nat.gcd 180 4620
def lcm_180_4620 : ℕ := Nat.lcm 180 4620
def sum_gcd_lcm_180_4620 : ℕ := gcd_180_4620 + lcm_180_4620

theorem sum_of_gcd_and_lcm_of_180_and_4620 :
  sum_gcd_lcm_180_4620 = 13920 :=
by
  sorry

end sum_of_gcd_and_lcm_of_180_and_4620_l1697_169758


namespace quadratic_real_roots_l1697_169766

theorem quadratic_real_roots (k : ℝ) (h : k ≠ 0) : 
  (∃ x1 x2 : ℝ, k * x1^2 - 6 * x1 - 1 = 0 ∧ k * x2^2 - 6 * x2 - 1 = 0 ∧ x1 ≠ x2) ↔ k ≥ -9 := 
by
  sorry

end quadratic_real_roots_l1697_169766


namespace required_bricks_l1697_169759

def brick_volume (length width height : ℝ) : ℝ := length * width * height

def wall_volume (length width height : ℝ) : ℝ := length * width * height

theorem required_bricks : 
  let brick_length := 25
  let brick_width := 11.25
  let brick_height := 6
  let wall_length := 850
  let wall_width := 600
  let wall_height := 22.5
  (wall_volume wall_length wall_width wall_height) / 
  (brick_volume brick_length brick_width brick_height) = 6800 :=
by
  sorry

end required_bricks_l1697_169759


namespace division_remainder_correct_l1697_169789

theorem division_remainder_correct :
  ∃ q r, 987670 = 128 * q + r ∧ 0 ≤ r ∧ r < 128 ∧ r = 22 :=
by
  sorry

end division_remainder_correct_l1697_169789


namespace imaginary_unit_multiplication_l1697_169767

theorem imaginary_unit_multiplication (i : ℂ) (h1 : i * i = -1) : i * (1 + i) = i - 1 :=
by
  sorry

end imaginary_unit_multiplication_l1697_169767


namespace pentagon_area_is_correct_l1697_169704

noncomputable def area_of_pentagon : ℕ :=
  let area_trapezoid := (1 / 2) * (25 + 28) * 30
  let area_triangle := (1 / 2) * 18 * 24
  area_trapezoid + area_triangle

theorem pentagon_area_is_correct (s1 s2 s3 s4 s5 : ℕ) (b1 b2 h1 b3 h2 : ℕ)
  (h₀ : s1 = 18) (h₁ : s2 = 25) (h₂ : s3 = 30) (h₃ : s4 = 28) (h₄ : s5 = 25)
  (h₅ : b1 = 25) (h₆ : b2 = 28) (h₇ : h1 = 30) (h₈ : b3 = 18) (h₉ : h2 = 24) :
  area_of_pentagon = 1011 := by
  -- placeholder for actual proof
  sorry

end pentagon_area_is_correct_l1697_169704


namespace line_does_not_pass_through_third_quadrant_l1697_169773

theorem line_does_not_pass_through_third_quadrant (x y : ℝ) (h : y = -x + 1) :
  ¬(x < 0 ∧ y < 0) :=
sorry

end line_does_not_pass_through_third_quadrant_l1697_169773


namespace pizza_combinations_l1697_169733

/-- The number of unique pizzas that can be made with exactly 5 toppings from a selection of 8 is 56. -/
theorem pizza_combinations : (Nat.choose 8 5) = 56 := by
  sorry

end pizza_combinations_l1697_169733


namespace math_problem_l1697_169750

noncomputable def answer := 21

theorem math_problem 
  (a b c d x : ℝ)
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : |x| = 3) : 
  2 * x^2 - (a * b - c - d) + |a * b + 3| = answer := 
sorry

end math_problem_l1697_169750
