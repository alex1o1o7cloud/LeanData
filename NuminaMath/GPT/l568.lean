import Mathlib

namespace expression_is_integer_expression_modulo_3_l568_56881

theorem expression_is_integer (n : ℕ) (hn : n > 0) : 
  ∃ (k : ℤ), (n^3 + (3/2) * n^2 + (1/2) * n - 1) = k := 
sorry

theorem expression_modulo_3 (n : ℕ) (hn : n > 0) : 
  (n^3 + (3/2) * n^2 + (1/2) * n - 1) % 3 = 2 :=
sorry

end expression_is_integer_expression_modulo_3_l568_56881


namespace difference_between_heads_and_feet_l568_56815

-- Definitions based on the conditions
def penguins := 30
def zebras := 22
def tigers := 8
def zookeepers := 12

-- Counting heads
def heads := penguins + zebras + tigers + zookeepers

-- Counting feet
def feet := (2 * penguins) + (4 * zebras) + (4 * tigers) + (2 * zookeepers)

-- Proving the difference between the number of feet and heads is 132
theorem difference_between_heads_and_feet : (feet - heads) = 132 :=
by
  sorry

end difference_between_heads_and_feet_l568_56815


namespace sandy_marks_l568_56829

def marks_each_correct_sum : ℕ := 3

theorem sandy_marks (x : ℕ) 
  (total_attempts : ℕ := 30)
  (correct_sums : ℕ := 23)
  (marks_per_incorrect_sum : ℕ := 2)
  (total_marks_obtained : ℕ := 55)
  (incorrect_sums : ℕ := total_attempts - correct_sums)
  (lost_marks : ℕ := incorrect_sums * marks_per_incorrect_sum) :
  (correct_sums * x - lost_marks = total_marks_obtained) -> x = marks_each_correct_sum :=
by
  sorry

end sandy_marks_l568_56829


namespace product_of_solutions_is_zero_l568_56863

theorem product_of_solutions_is_zero :
  (∀ x : ℝ, ((x + 3) / (3 * x + 3) = (5 * x + 4) / (8 * x + 4) -> x = 0)) -> true :=
by
  sorry

end product_of_solutions_is_zero_l568_56863


namespace polynomial_solutions_l568_56869

theorem polynomial_solutions (P : Polynomial ℝ) :
  (∀ x : ℝ, P.eval x * P.eval (x + 1) = P.eval (x^2 - x + 3)) →
  (P = 0 ∨ ∃ n : ℕ, P = (Polynomial.C 1) * (Polynomial.X^2 - 2 * Polynomial.X + 3)^n) :=
by
  sorry

end polynomial_solutions_l568_56869


namespace polar_r_eq_3_is_circle_l568_56823

theorem polar_r_eq_3_is_circle :
  ∀ θ : ℝ, ∃ x y : ℝ, (x, y) = (3 * Real.cos θ, 3 * Real.sin θ) ∧ x^2 + y^2 = 9 :=
by
  sorry

end polar_r_eq_3_is_circle_l568_56823


namespace largest_n_unique_k_l568_56871

theorem largest_n_unique_k :
  ∃ (n : ℕ), (∀ (k1 k2 : ℕ), 
    (9 / 17 < n / (n + k1) → n / (n + k1) < 8 / 15 → 9 / 17 < n / (n + k2) → n / (n + k2) < 8 / 15 → k1 = k2) ∧ 
    n = 72) :=
sorry

end largest_n_unique_k_l568_56871


namespace john_receives_more_l568_56817

noncomputable def partnership_difference (investment_john : ℝ) (investment_mike : ℝ) (profit : ℝ) : ℝ :=
  let total_investment := investment_john + investment_mike
  let one_third_profit := profit / 3
  let two_third_profit := 2 * profit / 3
  let john_effort_share := one_third_profit / 2
  let mike_effort_share := one_third_profit / 2
  let ratio_john := investment_john / total_investment
  let ratio_mike := investment_mike / total_investment
  let john_investment_share := ratio_john * two_third_profit
  let mike_investment_share := ratio_mike * two_third_profit
  let john_total := john_effort_share + john_investment_share
  let mike_total := mike_effort_share + mike_investment_share
  john_total - mike_total

theorem john_receives_more (investment_john investment_mike profit : ℝ)
  (h_john : investment_john = 700)
  (h_mike : investment_mike = 300)
  (h_profit : profit = 3000.0000000000005) :
  partnership_difference investment_john investment_mike profit = 800.0000000000001 := 
sorry

end john_receives_more_l568_56817


namespace abs_opposite_sign_eq_sum_l568_56855

theorem abs_opposite_sign_eq_sum (a b : ℤ) (h : (|a + 1| * |b + 2| < 0)) : a + b = -3 :=
sorry

end abs_opposite_sign_eq_sum_l568_56855


namespace greatest_number_zero_l568_56804

-- Define the condition (inequality)
def inequality (x : ℤ) : Prop :=
  3 * x + 2 < 5 - 2 * x

-- Define the property of being the greatest whole number satisfying the inequality
def greatest_whole_number (x : ℤ) : Prop :=
  inequality x ∧ (∀ y : ℤ, inequality y → y ≤ x)

-- The main theorem stating the greatest whole number satisfying the inequality is 0
theorem greatest_number_zero : greatest_whole_number 0 :=
by
  sorry

end greatest_number_zero_l568_56804


namespace find_base_k_l568_56868

-- Define the conversion condition as a polynomial equation.
def base_conversion (k : ℤ) : Prop := k^2 + 3*k + 2 = 42

-- State the theorem to be proven: given the conversion condition, k = 5.
theorem find_base_k (k : ℤ) (h : base_conversion k) : k = 5 :=
by
  sorry

end find_base_k_l568_56868


namespace find_range_of_f_l568_56866

noncomputable def f (x : ℝ) : ℝ := (Real.logb (1/2) x) ^ 2 - 2 * (Real.logb (1/2) x) + 4

theorem find_range_of_f :
  ∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → 7 ≤ f x ∧ f x ≤ 12 :=
by
  sorry

end find_range_of_f_l568_56866


namespace train_car_speed_ratio_l568_56849

theorem train_car_speed_ratio
  (distance_bus : ℕ) (time_bus : ℕ) (distance_car : ℕ) (time_car : ℕ)
  (speed_bus := distance_bus / time_bus)
  (speed_train := speed_bus / (3 / 4))
  (speed_car := distance_car / time_car)
  (ratio := (speed_train : ℚ) / (speed_car : ℚ))
  (h1 : distance_bus = 480)
  (h2 : time_bus = 8)
  (h3 : distance_car = 450)
  (h4 : time_car = 6) :
  ratio = 16 / 15 :=
by
  sorry

end train_car_speed_ratio_l568_56849


namespace initial_fish_count_l568_56819

theorem initial_fish_count (x : ℕ) (h1 : x + 47 = 69) : x = 22 :=
by
  sorry

end initial_fish_count_l568_56819


namespace task_completion_days_l568_56891

theorem task_completion_days (a b c d : ℝ) 
    (h1 : 1/a + 1/b = 1/8)
    (h2 : 1/b + 1/c = 1/6)
    (h3 : 1/c + 1/d = 1/12) :
    1/a + 1/d = 1/24 :=
by
  sorry

end task_completion_days_l568_56891


namespace cars_in_parking_lot_l568_56852

theorem cars_in_parking_lot (initial_cars left_cars entered_cars : ℕ) (h1 : initial_cars = 80)
(h2 : left_cars = 13) (h3 : entered_cars = left_cars + 5) : 
initial_cars - left_cars + entered_cars = 85 :=
by
  rw [h1, h2, h3]
  sorry

end cars_in_parking_lot_l568_56852


namespace john_speed_l568_56847

def johns_speed (race_distance_miles next_fastest_guy_time_min won_by_min : ℕ) : ℕ :=
    let john_time_min := next_fastest_guy_time_min - won_by_min
    let john_time_hr := john_time_min / 60
    race_distance_miles / john_time_hr

theorem john_speed (race_distance_miles next_fastest_guy_time_min won_by_min : ℕ)
    (h1 : race_distance_miles = 5) (h2 : next_fastest_guy_time_min = 23) (h3 : won_by_min = 3) : 
    johns_speed race_distance_miles next_fastest_guy_time_min won_by_min = 15 := 
by
    sorry

end john_speed_l568_56847


namespace existence_not_implied_by_validity_l568_56898

-- Let us formalize the theorem and then show that its validity does not imply the existence of such a function.

-- Definitions for condition (A) and the theorem statement
axiom condition_A (f : ℝ → ℝ) : Prop
axiom theorem_239 : ∀ f, condition_A f → ∃ T, ∀ x, f (x + T) = f x

-- Translation of the problem statement into Lean
theorem existence_not_implied_by_validity :
  (∀ f, condition_A f → ∃ T, ∀ x, f (x + T) = f x) → 
  ¬ (∃ f, condition_A f) :=
sorry

end existence_not_implied_by_validity_l568_56898


namespace cos_E_floor_1000_l568_56896

theorem cos_E_floor_1000 {EF GH FG EH : ℝ} {E G : ℝ} (h1 : EF = 200) (h2 : GH = 200) (h3 : FG + EH = 380) (h4 : E = G) (h5 : EH ≠ FG) :
  ∃ (cE : ℝ), cE = 11/16 ∧ ⌊ 1000 * cE ⌋ = 687 :=
by sorry

end cos_E_floor_1000_l568_56896


namespace rectangle_length_reduction_l568_56862

theorem rectangle_length_reduction (L W : ℝ) (h : L > 0 ∧ W > 0) :
  let new_length := L * (1 - 10 / 100)
  let new_width := W * (10 / 9)
  (new_length * new_width = L * W) → 
  x = 10 := by sorry

end rectangle_length_reduction_l568_56862


namespace probability_of_five_dice_all_same_l568_56861

theorem probability_of_five_dice_all_same : 
  (6 / (6 ^ 5) = 1 / 1296) :=
by
  sorry

end probability_of_five_dice_all_same_l568_56861


namespace dodecahedron_diagonals_l568_56807

-- Define a structure representing a dodecahedron with its properties
structure Dodecahedron where
  faces : Nat
  vertices : Nat
  faces_meeting_at_each_vertex : Nat

-- Concretely define a dodecahedron based on the given problem properties
def dodecahedron_example : Dodecahedron :=
  { faces := 12,
    vertices := 20,
    faces_meeting_at_each_vertex := 3 }

-- Lean statement to prove the number of interior diagonals in a dodecahedron
theorem dodecahedron_diagonals (d : Dodecahedron) (h : d = dodecahedron_example) : 
  (d.vertices * (d.vertices - d.faces_meeting_at_each_vertex) / 2) = 160 := by
  rw [h]
  -- Even though we skip the proof, Lean should recognize the transformation
  sorry

end dodecahedron_diagonals_l568_56807


namespace circle_radius_l568_56875

-- Define the general equation of the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x + 4 * y = 0

-- Prove the radius of the circle given by the equation is √5
theorem circle_radius :
  (∀ x y : ℝ, circle_eq x y) →
  (∃ r : ℝ, r = Real.sqrt 5) :=
by
  sorry

end circle_radius_l568_56875


namespace intersecting_chords_second_length_l568_56839

theorem intersecting_chords_second_length (a b : ℕ) (k : ℕ) 
  (h_a : a = 12) (h_b : b = 18) (h_ratio : k ^ 2 = (a * b) / 24) 
  (x y : ℕ) (h_x : x = 3 * k) (h_y : y = 8 * k) :
  x + y = 33 :=
by
  sorry

end intersecting_chords_second_length_l568_56839


namespace Delta_15_xDelta_eq_neg_15_l568_56877

-- Definitions of the operations based on conditions
def xDelta (x : ℝ) : ℝ := 9 - x
def Delta (x : ℝ) : ℝ := x - 9

-- Statement that we need to prove
theorem Delta_15_xDelta_eq_neg_15 : Delta (xDelta 15) = -15 :=
by
  -- The proof will go here
  sorry

end Delta_15_xDelta_eq_neg_15_l568_56877


namespace number_of_valid_sets_l568_56835

open Set

variable {α : Type} (a b : α)

def is_valid_set (M : Set α) : Prop := M ∪ {a} = {a, b}

theorem number_of_valid_sets (a b : α) : (∃! M : Set α, is_valid_set a b M) := 
sorry

end number_of_valid_sets_l568_56835


namespace circle_area_l568_56856

/--
Given the polar equation of a circle r = -4 * cos θ + 8 * sin θ,
prove that the area of the circle is 20π.
-/
theorem circle_area (θ : ℝ) (r : ℝ) (cos : ℝ → ℝ) (sin : ℝ → ℝ) 
  (h_eq : ∀ θ : ℝ, r = -4 * cos θ + 8 * sin θ) : 
  ∃ A : ℝ, A = 20 * Real.pi :=
by
  sorry

end circle_area_l568_56856


namespace percentage_decrease_last_year_l568_56854

-- Define the percentage decrease last year
variable (x : ℝ)

-- Define the condition that expresses the stock price this year
def final_price_change (x : ℝ) : Prop :=
  (1 - x / 100) * 1.10 = 1 + 4.499999999999993 / 100

-- Theorem stating the percentage decrease
theorem percentage_decrease_last_year : final_price_change 5 := by
  sorry

end percentage_decrease_last_year_l568_56854


namespace speed_with_stream_l568_56873

-- Definitions for the conditions in part a
def Vm : ℕ := 8  -- Speed of the man in still water (in km/h)
def Vs : ℕ := Vm - 4  -- Speed of the stream (in km/h), derived from man's speed against the stream

-- The statement to prove the man's speed with the stream
theorem speed_with_stream : Vm + Vs = 12 := by sorry

end speed_with_stream_l568_56873


namespace find_k_l568_56843

theorem find_k : ∃ k : ℕ, 32 / k = 4 ∧ k = 8 := 
sorry

end find_k_l568_56843


namespace intersection_of_domains_l568_56822

def M (x : ℝ) : Prop := x < 1
def N (x : ℝ) : Prop := x > -1
def P (x : ℝ) : Prop := -1 < x ∧ x < 1

theorem intersection_of_domains : {x : ℝ | M x} ∩ {x : ℝ | N x} = {x : ℝ | P x} :=
by
  sorry

end intersection_of_domains_l568_56822


namespace snow_white_seven_piles_l568_56874

def split_pile_action (piles : List ℕ) : Prop :=
  ∃ pile1 pile2, pile1 > 0 ∧ pile2 > 0 ∧ pile1 + pile2 + 1 ∈ piles

theorem snow_white_seven_piles :
  ∃ piles : List ℕ, piles.length = 7 ∧ ∀ pile ∈ piles, pile = 3 :=
sorry

end snow_white_seven_piles_l568_56874


namespace lines_parallel_iff_a_eq_1_l568_56851

theorem lines_parallel_iff_a_eq_1 (x y a : ℝ) :
    (a = 1 ↔ ∃ k : ℝ, ∀ x y : ℝ, a*x + y - 1 = k*(x + a*y + 1)) :=
sorry

end lines_parallel_iff_a_eq_1_l568_56851


namespace A_sub_B_value_l568_56837

def A : ℕ := 1000 * 1 + 100 * 16 + 10 * 28
def B : ℕ := 355 + 245 * 3

theorem A_sub_B_value : A - B = 1790 := by
  sorry

end A_sub_B_value_l568_56837


namespace determine_truth_tellers_min_questions_to_determine_truth_tellers_l568_56830

variables (n k : ℕ)
variables (h_n_pos : 0 < n) (h_k_pos : 0 < k) (h_k_le_n : k ≤ n)

theorem determine_truth_tellers (h : k % 2 = 0) : 
  ∃ m : ℕ, m = n :=
  sorry

theorem min_questions_to_determine_truth_tellers :
  ∃ m : ℕ, m = n :=
  sorry

end determine_truth_tellers_min_questions_to_determine_truth_tellers_l568_56830


namespace find_a₃_l568_56816

variable (a₁ a₂ a₃ a₄ a₅ : ℝ)
variable (S₅ : ℝ) (a_seq : ℕ → ℝ)

-- Define the conditions for arithmetic sequence and given sum
def is_arithmetic_sequence (a_seq : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a_seq (n+1) - a_seq n = a_seq 1 - a_seq 0

axiom sum_first_five_terms (S₅ : ℝ) (hS : S₅ = 20) : 
  S₅ = (5 * (a₁ + a₅)) / 2

-- Main theorem we need to prove
theorem find_a₃ (hS₅ : S₅ = 20) (h_seq : is_arithmetic_sequence a_seq) :
  (∃ (a₃ : ℝ), a₃ = 4) :=
sorry

end find_a₃_l568_56816


namespace cattle_selling_price_per_pound_correct_l568_56886

def purchase_price : ℝ := 40000
def cattle_count : ℕ := 100
def feed_cost_percentage : ℝ := 0.20
def weight_per_head : ℕ := 1000
def profit : ℝ := 112000

noncomputable def total_feed_cost : ℝ := purchase_price * feed_cost_percentage
noncomputable def total_cost : ℝ := purchase_price + total_feed_cost
noncomputable def total_revenue : ℝ := total_cost + profit
def total_weight : ℕ := cattle_count * weight_per_head
noncomputable def selling_price_per_pound : ℝ := total_revenue / total_weight

theorem cattle_selling_price_per_pound_correct :
  selling_price_per_pound = 1.60 := by
  sorry

end cattle_selling_price_per_pound_correct_l568_56886


namespace sum_of_largest_two_l568_56846

-- Define the three numbers
def a := 10
def b := 11
def c := 12

-- Define the sum of the largest and the next largest numbers
def sum_of_largest_two_numbers (x y z : ℕ) : ℕ :=
  if x >= y ∧ y >= z then x + y
  else if x >= z ∧ z >= y then x + z
  else if y >= x ∧ x >= z then y + x
  else if y >= z ∧ z >= x then y + z
  else if z >= x ∧ x >= y then z + x
  else z + y

-- State the theorem to prove
theorem sum_of_largest_two (x y z : ℕ) : sum_of_largest_two_numbers x y z = 23 :=
by
  sorry

end sum_of_largest_two_l568_56846


namespace find_roots_l568_56889

noncomputable def P (x : ℝ) : ℝ := x^4 - 3 * x^3 + 3 * x^2 - x - 6

theorem find_roots : {x : ℝ | P x = 0} = {-1, 1, 2} :=
by
  sorry

end find_roots_l568_56889


namespace part1_part2_l568_56841

variable (α : Real)
-- Condition
axiom tan_neg_alpha : Real.tan (-α) = -2

-- Question 1
theorem part1 : ((Real.sin α + Real.cos α) / (Real.sin α - Real.cos α)) = 3 := 
by
  sorry

-- Question 2
theorem part2 : Real.sin (2 * α) = 4 / 5 := 
by
  sorry

end part1_part2_l568_56841


namespace number_of_cups_needed_to_fill_container_l568_56834

theorem number_of_cups_needed_to_fill_container (container_capacity cup_capacity : ℕ) (h1 : container_capacity = 640) (h2 : cup_capacity = 120) : 
  (container_capacity + cup_capacity - 1) / cup_capacity = 6 :=
by
  sorry

end number_of_cups_needed_to_fill_container_l568_56834


namespace cube_construction_possible_l568_56879

theorem cube_construction_possible (n : ℕ) : (∃ k : ℕ, n = 12 * k) ↔ ∃ V : ℕ, (n ^ 3) = 12 * V := by
sorry

end cube_construction_possible_l568_56879


namespace domain_of_f_3x_minus_1_domain_of_f_l568_56825

-- Problem (1): Domain of f(3x - 1)
theorem domain_of_f_3x_minus_1 (f : ℝ → ℝ) :
  (∀ x, -2 ≤ f x ∧ f x ≤ 1) →
  (∀ x, -1 / 3 ≤ x ∧ x ≤ 2 / 3) :=
by
  intro h
  sorry

-- Problem (2): Domain of f(x)
theorem domain_of_f (f : ℝ → ℝ) :
  (∀ x, -1 ≤ 2*x + 5 ∧ 2*x + 5 ≤ 4) →
  (∀ y, 3 ≤ y ∧ y ≤ 13) :=
by
  intro h
  sorry

end domain_of_f_3x_minus_1_domain_of_f_l568_56825


namespace value_of_x_plus_y_l568_56885

theorem value_of_x_plus_y (x y : ℤ) (hx : x = -3) (hy : |y| = 5) : x + y = 2 ∨ x + y = -8 := by
  sorry

end value_of_x_plus_y_l568_56885


namespace fraction_filled_l568_56848

-- Definitions for the given conditions
variables (x C : ℝ) (h₁ : 20 * x / 3 = 25 * C / 5) 

-- The goal is to show that x / C = 3 / 4
theorem fraction_filled (h₁ : 20 * x / 3 = 25 * C / 5) : x / C = 3 / 4 :=
by sorry

end fraction_filled_l568_56848


namespace average_of_11_results_l568_56803

theorem average_of_11_results 
  (S1: ℝ) (S2: ℝ) (fifth_result: ℝ) -- Define the variables
  (h1: S1 / 5 = 49)                -- sum of the first 5 results
  (h2: S2 / 7 = 52)                -- sum of the last 7 results
  (h3: fifth_result = 147)         -- the fifth result 
  : (S1 + S2 - fifth_result) / 11 = 42 := -- statement of the problem
by
  sorry

end average_of_11_results_l568_56803


namespace remainder_p_x_minus_2_l568_56826

def p (x : ℝ) := x^5 + 2 * x^2 + 3

theorem remainder_p_x_minus_2 : p 2 = 43 := 
by
  sorry

end remainder_p_x_minus_2_l568_56826


namespace total_oranges_over_four_days_l568_56878

def jeremy_oranges_monday := 100
def jeremy_oranges_tuesday (B: ℕ) := 3 * jeremy_oranges_monday
def jeremy_oranges_wednesday (B: ℕ) (C: ℕ) := 2 * (jeremy_oranges_monday + B)
def jeremy_oranges_thursday := 70
def brother_oranges_tuesday := 3 * jeremy_oranges_monday - jeremy_oranges_monday -- This is B from Tuesday
def cousin_oranges_wednesday (B: ℕ) (C: ℕ) := 2 * (jeremy_oranges_monday + B) - (jeremy_oranges_monday + B)

theorem total_oranges_over_four_days (B: ℕ) (C: ℕ)
        (B_equals_tuesday: B = brother_oranges_tuesday)
        (J_plus_B_equals_300 : jeremy_oranges_tuesday B = 300)
        (J_plus_B_plus_C_equals_600 : jeremy_oranges_wednesday B C = 600)
        (J_thursday_is_70 : jeremy_oranges_thursday = 70)
        (B_thursday_is_B : B = brother_oranges_tuesday):
    100 + 300 + 600 + 270 = 1270 := by
        sorry

end total_oranges_over_four_days_l568_56878


namespace ratio_of_efficiencies_l568_56880

-- Definitions of efficiencies
def efficiency (time : ℕ) : ℚ := 1 / time

-- Conditions:
def E_C : ℚ := efficiency 20
def E_D : ℚ := efficiency 30
def E_A : ℚ := efficiency 18
def E_B : ℚ := 1 / 36 -- Placeholder for efficiency of B to complete the statement

-- The proof goal
theorem ratio_of_efficiencies (h1 : E_A + E_B = E_C + E_D) : E_A / E_B = 2 :=
by
  -- Placeholder to structure the format, the proof will be constructed here
  sorry

end ratio_of_efficiencies_l568_56880


namespace seeds_in_each_flower_bed_l568_56883

theorem seeds_in_each_flower_bed (total_seeds : ℕ) (flower_beds : ℕ) (h1 : total_seeds = 54) (h2 : flower_beds = 9) : total_seeds / flower_beds = 6 :=
by
  sorry

end seeds_in_each_flower_bed_l568_56883


namespace prime_division_or_divisibility_l568_56814

open Nat

theorem prime_division_or_divisibility (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) (hodd : Odd p) (hd : p ∣ q^r + 1) :
    (2 * r ∣ p - 1) ∨ (p ∣ q^2 - 1) := 
sorry

end prime_division_or_divisibility_l568_56814


namespace proof_l568_56818

noncomputable def problem : Prop :=
  let a := 1
  let b := 2
  let angleC := 60 * Real.pi / 180 -- convert degrees to radians
  let cosC := Real.cos angleC
  let sinC := Real.sin angleC
  let c_squared := a^2 + b^2 - 2 * a * b * cosC
  let c := Real.sqrt c_squared
  let area := 0.5 * a * b * sinC
  c = Real.sqrt 3 ∧ area = Real.sqrt 3 / 2

theorem proof : problem :=
by
  sorry

end proof_l568_56818


namespace speed_in_km_per_hr_l568_56897

noncomputable def side : ℝ := 40
noncomputable def time : ℝ := 64

-- Theorem statement
theorem speed_in_km_per_hr (side : ℝ) (time : ℝ) (h₁ : side = 40) (h₂ : time = 64) : 
  (4 * side * 3600) / (time * 1000) = 9 := by
  rw [h₁, h₂]
  sorry

end speed_in_km_per_hr_l568_56897


namespace no_natural_n_such_that_6n2_plus_5n_is_power_of_2_l568_56821

theorem no_natural_n_such_that_6n2_plus_5n_is_power_of_2 :
  ¬ ∃ n : ℕ, ∃ k : ℕ, 6 * n^2 + 5 * n = 2^k :=
by
  sorry

end no_natural_n_such_that_6n2_plus_5n_is_power_of_2_l568_56821


namespace distribute_marbles_correct_l568_56800

def distribute_marbles (total_marbles : Nat) (num_boys : Nat) : Nat :=
  total_marbles / num_boys

theorem distribute_marbles_correct :
  distribute_marbles 20 2 = 10 := 
by 
  sorry

end distribute_marbles_correct_l568_56800


namespace students_neither_cs_nor_robotics_l568_56842

theorem students_neither_cs_nor_robotics
  (total_students : ℕ)
  (cs_students : ℕ)
  (robotics_students : ℕ)
  (both_cs_and_robotics : ℕ)
  (H1 : total_students = 150)
  (H2 : cs_students = 90)
  (H3 : robotics_students = 70)
  (H4 : both_cs_and_robotics = 20) :
  (total_students - (cs_students + robotics_students - both_cs_and_robotics)) = 10 :=
by
  sorry

end students_neither_cs_nor_robotics_l568_56842


namespace complex_root_problem_l568_56853

theorem complex_root_problem (z : ℂ) :
  z^2 - 3*z = 10 - 6*Complex.I ↔
  z = 5.5 - 0.75 * Complex.I ∨
  z = -2.5 + 0.75 * Complex.I ∨
  z = 3.5 - 1.5 * Complex.I ∨
  z = -0.5 + 1.5 * Complex.I :=
sorry

end complex_root_problem_l568_56853


namespace solution_set_of_inequality_l568_56858

def f (x : ℝ) : ℝ := sorry
def f_prime (x : ℝ) : ℝ := sorry

theorem solution_set_of_inequality :
  (∀ x > 0, x^2 * f_prime x + 1 > 0) → 
  f 1 = 5 →
  { x : ℝ | 0 < x ∧ x < 1 } = { x : ℝ | 0 < x ∧ f x < 1 / x + 4 } :=
by 
  intros h1 h2 
  sorry

end solution_set_of_inequality_l568_56858


namespace roses_per_flat_l568_56801

-- Conditions
def flats_petunias := 4
def petunias_per_flat := 8
def flats_roses := 3
def venus_flytraps := 2
def fertilizer_per_petunia := 8
def fertilizer_per_rose := 3
def fertilizer_per_venus_flytrap := 2
def total_fertilizer_needed := 314

-- Derived definitions
def total_petunias := flats_petunias * petunias_per_flat
def fertilizer_for_petunias := total_petunias * fertilizer_per_petunia
def fertilizer_for_venus_flytraps := venus_flytraps * fertilizer_per_venus_flytrap
def total_fertilizer_needed_roses := total_fertilizer_needed - (fertilizer_for_petunias + fertilizer_for_venus_flytraps)

-- Proof statement
theorem roses_per_flat :
  ∃ R : ℕ, flats_roses * R * fertilizer_per_rose = total_fertilizer_needed_roses ∧ R = 6 :=
by
  -- Proof goes here
  sorry

end roses_per_flat_l568_56801


namespace find_value_l568_56824

variable (a b c : Int)

-- Conditions from the problem
axiom abs_a_eq_two : |a| = 2
axiom b_eq_neg_seven : b = -7
axiom neg_c_eq_neg_five : -c = -5

-- Proof problem
theorem find_value : a^2 + (-b) + (-c) = 6 := by
  sorry

end find_value_l568_56824


namespace difference_of_numbers_l568_56811

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 12390) (h2 : b = 2 * a + 18) : b - a = 4142 :=
by {
  sorry
}

end difference_of_numbers_l568_56811


namespace tan_alpha_of_cos_alpha_l568_56870

theorem tan_alpha_of_cos_alpha (α : ℝ) (hα : 0 < α ∧ α < Real.pi) (h_cos : Real.cos α = -3/5) :
  Real.tan α = -4/3 :=
sorry

end tan_alpha_of_cos_alpha_l568_56870


namespace pink_tulips_l568_56827

theorem pink_tulips (total_tulips : ℕ)
    (blue_ratio : ℚ) (red_ratio : ℚ)
    (h_total : total_tulips = 56)
    (h_blue_ratio : blue_ratio = 3/8)
    (h_red_ratio : red_ratio = 3/7) :
    ∃ pink_tulips : ℕ, pink_tulips = total_tulips - ((blue_ratio * total_tulips) + (red_ratio * total_tulips)) ∧ pink_tulips = 11 := by
  sorry

end pink_tulips_l568_56827


namespace amy_required_hours_per_week_l568_56812

variable (summer_hours_per_week : ℕ) (summer_weeks : ℕ) (summer_pay : ℕ) 
variable (pay_raise_percent : ℕ) (school_year_weeks : ℕ) (required_school_year_pay : ℕ)

def summer_hours_total := summer_hours_per_week * summer_weeks
def summer_hourly_pay := summer_pay / summer_hours_total
def new_hourly_pay := summer_hourly_pay + (summer_hourly_pay / 10)  -- 10% pay raise
def total_needed_hours := required_school_year_pay / new_hourly_pay
def required_hours_per_week := total_needed_hours / school_year_weeks

theorem amy_required_hours_per_week :
  summer_hours_per_week = 40 →
  summer_weeks = 12 →
  summer_pay = 4800 →
  pay_raise_percent = 10 →
  school_year_weeks = 36 →
  required_school_year_pay = 7200 →
  required_hours_per_week = 18 := sorry

end amy_required_hours_per_week_l568_56812


namespace eugene_total_cost_l568_56802

variable (TshirtCost PantCost ShoeCost : ℕ)
variable (NumTshirts NumPants NumShoes Discount : ℕ)

theorem eugene_total_cost
  (hTshirtCost : TshirtCost = 20)
  (hPantCost : PantCost = 80)
  (hShoeCost : ShoeCost = 150)
  (hNumTshirts : NumTshirts = 4)
  (hNumPants : NumPants = 3)
  (hNumShoes : NumShoes = 2)
  (hDiscount : Discount = 10) :
  TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes - (TshirtCost * NumTshirts + PantCost * NumPants + ShoeCost * NumShoes) * Discount / 100 = 558 := by
  sorry

end eugene_total_cost_l568_56802


namespace expected_number_of_different_faces_l568_56857

noncomputable def expected_faces : ℝ :=
  let probability_face_1_not_appearing := (5 / 6)^6
  let E_zeta_1 := 1 - probability_face_1_not_appearing
  6 * E_zeta_1

theorem expected_number_of_different_faces :
  expected_faces = (6^6 - 5^6) / 6^5 := by 
  sorry

end expected_number_of_different_faces_l568_56857


namespace correct_population_statement_l568_56860

def correct_statement :=
  "The mathematics scores of all candidates in the city's high school entrance examination last year constitute the population."

def sample_size : ℕ := 500

def is_correct (statement : String) : Prop :=
  statement = correct_statement

theorem correct_population_statement (scores : Fin 500 → ℝ) :
  is_correct "The mathematics scores of all candidates in the city's high school entrance examination last year constitute the population." :=
by
  sorry

end correct_population_statement_l568_56860


namespace coeff_x3_in_expansion_of_x_plus_1_50_l568_56844

theorem coeff_x3_in_expansion_of_x_plus_1_50 :
  (Finset.range 51).sum (λ k => Nat.choose 50 k * (1 : ℕ) ^ (50 - k) * k ^ 3) = 19600 := by
  sorry

end coeff_x3_in_expansion_of_x_plus_1_50_l568_56844


namespace nonempty_solution_set_range_l568_56845

theorem nonempty_solution_set_range (a : ℝ) :
  (∃ x : ℝ, |x - 3| + |x - 4| < a) ↔ a > 1 := sorry

end nonempty_solution_set_range_l568_56845


namespace find_inscribed_circle_area_l568_56831

noncomputable def inscribed_circle_area (length : ℝ) (breadth : ℝ) : ℝ :=
  let perimeter_rectangle := 2 * (length + breadth)
  let side_square := perimeter_rectangle / 4
  let radius_circle := side_square / 2
  Real.pi * radius_circle^2

theorem find_inscribed_circle_area :
  inscribed_circle_area 36 28 = 804.25 := by
  sorry

end find_inscribed_circle_area_l568_56831


namespace mike_earnings_l568_56876

theorem mike_earnings :
  let total_games := 16
  let non_working_games := 8
  let price_per_game := 7
  let working_games := total_games - non_working_games
  let earnings := working_games * price_per_game
  earnings = 56 := 
by
  sorry

end mike_earnings_l568_56876


namespace solution_set_of_abs_x_minus_1_lt_1_l568_56894

theorem solution_set_of_abs_x_minus_1_lt_1 : {x : ℝ | |x - 1| < 1} = {x : ℝ | 0 < x ∧ x < 2} :=
by
  sorry

end solution_set_of_abs_x_minus_1_lt_1_l568_56894


namespace pigeonhole_6_points_3x4_l568_56838

theorem pigeonhole_6_points_3x4 :
  ∀ (points : Fin 6 → (ℝ × ℝ)), 
  (∀ i, 0 ≤ (points i).fst ∧ (points i).fst ≤ 4 ∧ 0 ≤ (points i).snd ∧ (points i).snd ≤ 3) →
  ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 5 :=
by
  sorry

end pigeonhole_6_points_3x4_l568_56838


namespace supplies_total_cost_l568_56809

-- Definitions based on conditions in a)
def cost_of_bow : ℕ := 5
def cost_of_vinegar : ℕ := 2
def cost_of_baking_soda : ℕ := 1
def students_count : ℕ := 23

-- The main theorem to prove
theorem supplies_total_cost :
  cost_of_bow * students_count + cost_of_vinegar * students_count + cost_of_baking_soda * students_count = 184 :=
by
  sorry

end supplies_total_cost_l568_56809


namespace log_base_9_of_x_cubed_is_3_l568_56805

theorem log_base_9_of_x_cubed_is_3 
  (x : Real) 
  (hx : x = 9.000000000000002) : 
  Real.logb 9 (x^3) = 3 := 
by 
  sorry

end log_base_9_of_x_cubed_is_3_l568_56805


namespace min_games_required_l568_56895

-- Given condition: max_games ≤ 15
def max_games := 15

-- Theorem statement to prove: minimum number of games that must be played is 8
theorem min_games_required (n : ℕ) (h : n ≤ max_games) : n = 8 :=
sorry

end min_games_required_l568_56895


namespace reading_time_difference_l568_56882

theorem reading_time_difference :
  let xanthia_reading_speed := 100 -- pages per hour
  let molly_reading_speed := 50 -- pages per hour
  let book_pages := 225
  let xanthia_time := book_pages / xanthia_reading_speed
  let molly_time := book_pages / molly_reading_speed
  let difference_in_hours := molly_time - xanthia_time
  let difference_in_minutes := difference_in_hours * 60
  difference_in_minutes = 135 := by
  sorry

end reading_time_difference_l568_56882


namespace v_2015_eq_2_l568_56850

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 5
  | 2 => 3
  | 3 => 4
  | 4 => 1
  | 5 => 2
  | _ => 0  -- assuming g(x) = 0 for other values, though not used here

def v : ℕ → ℕ
| 0     => 3
| (n+1) => g (v n)

theorem v_2015_eq_2 : v 2015 = 2 :=
by
  sorry

end v_2015_eq_2_l568_56850


namespace max_mn_value_l568_56888

noncomputable def vector_max_sum (OA OB : ℝ) (m n : ℝ) : Prop :=
  (OA * OA = 4 ∧ OB * OB = 4 ∧ OA * OB = 2) →
  ((m * OA + n * OB) * (m * OA + n * OB) = 4) →
  (m + n ≤ 2 * Real.sqrt 3 / 3)

-- Here's the statement for the maximum value problem
theorem max_mn_value {m n : ℝ} (h1 : m > 0) (h2 : n > 0) :
  vector_max_sum 2 2 m n :=
sorry

end max_mn_value_l568_56888


namespace find_constant_l568_56864

-- Definitions based on the conditions provided
variable (f : ℕ → ℕ)
variable (c : ℕ)

-- Given conditions
def f_1_eq_0 : f 1 = 0 := sorry
def functional_equation (m n : ℕ) : f (m + n) = f m + f n + c * (m * n - 1) := sorry
def f_17_eq_4832 : f 17 = 4832 := sorry

-- The mathematically equivalent proof problem
theorem find_constant : c = 4 := 
sorry

end find_constant_l568_56864


namespace bullet_trains_crossing_time_l568_56899

theorem bullet_trains_crossing_time
  (length : ℝ)
  (time1 time2 : ℝ)
  (speed1 speed2 : ℝ)
  (relative_speed : ℝ)
  (total_distance : ℝ)
  (cross_time : ℝ)
  (h_length : length = 120)
  (h_time1 : time1 = 10)
  (h_time2 : time2 = 20)
  (h_speed1 : speed1 = length / time1)
  (h_speed2 : speed2 = length / time2)
  (h_relative_speed : relative_speed = speed1 + speed2)
  (h_total_distance : total_distance = length + length)
  (h_cross_time : cross_time = total_distance / relative_speed) :
  cross_time = 240 / 18 := 
by
  sorry

end bullet_trains_crossing_time_l568_56899


namespace subscription_difference_is_4000_l568_56884

-- Given definitions
def total_subscription (A B C : ℕ) : Prop :=
  A + B + C = 50000

def subscription_B (x : ℕ) : ℕ :=
  x + 5000

def subscription_A (x y : ℕ) : ℕ :=
  x + 5000 + y

def profit_ratio (profit_C total_profit x : ℕ) : Prop :=
  (profit_C : ℚ) / total_profit = (x : ℚ) / 50000

-- Prove that A subscribed Rs. 4,000 more than B
theorem subscription_difference_is_4000 (x y : ℕ)
  (h1 : total_subscription (subscription_A x y) (subscription_B x) x)
  (h2 : profit_ratio 8400 35000 x) :
  y = 4000 :=
sorry

end subscription_difference_is_4000_l568_56884


namespace weight_of_oil_per_ml_l568_56806

variable (w : ℝ)  -- Weight of the oil per ml
variable (total_volume : ℝ := 150)  -- Bowl volume
variable (oil_fraction : ℝ := 2/3)  -- Fraction of oil
variable (vinegar_fraction : ℝ := 1/3)  -- Fraction of vinegar
variable (vinegar_density : ℝ := 4)  -- Vinegar density in g/ml
variable (total_weight : ℝ := 700)  -- Total weight in grams

theorem weight_of_oil_per_ml :
  (total_volume * oil_fraction * w) + (total_volume * vinegar_fraction * vinegar_density) = total_weight →
  w = 5 := by
  sorry

end weight_of_oil_per_ml_l568_56806


namespace maximum_guaranteed_money_l568_56867

theorem maximum_guaranteed_money (board_width board_height tromino_width tromino_height guaranteed_rubles : ℕ) 
  (h_board_width : board_width = 21) 
  (h_board_height : board_height = 20)
  (h_tromino_width : tromino_width = 3) 
  (h_tromino_height : tromino_height = 1)
  (h_guaranteed_rubles : guaranteed_rubles = 14) :
  true := by
  sorry

end maximum_guaranteed_money_l568_56867


namespace min_reciprocal_sum_l568_56810

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = 1) :
  (1/x) + (1/y) = 3 + 2 * Real.sqrt 2 :=
sorry

end min_reciprocal_sum_l568_56810


namespace tan_double_angle_l568_56892

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f_derivative_def (x : ℝ) : ℝ := 3 * f x

theorem tan_double_angle (x : ℝ) (h : f_derivative_def x = Real.cos x - Real.sin x) : 
  Real.tan (2 * x) = -4 / 3 :=
by
  sorry

end tan_double_angle_l568_56892


namespace discriminant_eq_M_l568_56833

theorem discriminant_eq_M (a b c x0 : ℝ) (h1: a ≠ 0) (h2: a * x0^2 + b * x0 + c = 0) :
  (b^2 - 4 * a * c) = (2 * a * x0 + b)^2 :=
by
  sorry

end discriminant_eq_M_l568_56833


namespace tangent_neg_five_pi_six_eq_one_over_sqrt_three_l568_56808

noncomputable def tangent_neg_five_pi_six : Real :=
  Real.tan (-5 * Real.pi / 6)

theorem tangent_neg_five_pi_six_eq_one_over_sqrt_three :
  tangent_neg_five_pi_six = 1 / Real.sqrt 3 := by
  sorry

end tangent_neg_five_pi_six_eq_one_over_sqrt_three_l568_56808


namespace average_class_size_l568_56872

theorem average_class_size 
  (three_year_olds : ℕ := 13)
  (four_year_olds : ℕ := 20)
  (five_year_olds : ℕ := 15)
  (six_year_olds : ℕ := 22) : 
  ((three_year_olds + four_year_olds + five_year_olds + six_year_olds) / 2) = 35 := 
by
  sorry

end average_class_size_l568_56872


namespace tan_alpha_sol_expr_sol_l568_56820

noncomputable def tan_half_alpha (α : ℝ) : ℝ := 2

noncomputable def tan_alpha_from_half (α : ℝ) : ℝ := 
  let tan_half := tan_half_alpha α
  2 * tan_half / (1 - tan_half * tan_half)

theorem tan_alpha_sol (α : ℝ) (h : tan_half_alpha α = 2) : tan_alpha_from_half α = -4 / 3 := by
  sorry

noncomputable def expr_eval (α : ℝ) : ℝ :=
  let tan_α := tan_alpha_from_half α
  let sin_α := tan_α / Real.sqrt (1 + tan_α * tan_α)
  let cos_α := 1 / Real.sqrt (1 + tan_α * tan_α)
  (6 * sin_α + cos_α) / (3 * sin_α - 2 * cos_α)

theorem expr_sol (α : ℝ) (h : tan_half_alpha α = 2) : expr_eval α = 7 / 6 := by
  sorry

end tan_alpha_sol_expr_sol_l568_56820


namespace not_divisible_67_l568_56836

theorem not_divisible_67
  (x y : ℕ)
  (hx : ¬ (67 ∣ x))
  (hy : ¬ (67 ∣ y))
  (h : (7 * x + 32 * y) % 67 = 0)
  : (10 * x + 17 * y + 1) % 67 ≠ 0 := sorry

end not_divisible_67_l568_56836


namespace Olivia_pays_4_dollars_l568_56832

-- Definitions based on the conditions
def quarters_chips : ℕ := 4
def quarters_soda : ℕ := 12
def conversion_rate : ℕ := 4

-- Prove that the total dollars Olivia pays is 4
theorem Olivia_pays_4_dollars (h1 : quarters_chips = 4) (h2 : quarters_soda = 12) (h3 : conversion_rate = 4) : 
  (quarters_chips + quarters_soda) / conversion_rate = 4 :=
by
  -- skipping the proof
  sorry

end Olivia_pays_4_dollars_l568_56832


namespace infinite_chain_resistance_l568_56865

variables (R_0 R_X : ℝ)
def infinite_chain_resistance_condition (R_0 : ℝ) (R_X : ℝ) : Prop :=
  R_X = R_0 + (R_0 * R_X) / (R_0 + R_X)

theorem infinite_chain_resistance (R_0 : ℝ) (h : R_0 = 50) :
  ∃ R_X, infinite_chain_resistance_condition R_0 R_X ∧ R_X = (R_0 * (1 + Real.sqrt 5)) / 2 :=
  sorry

end infinite_chain_resistance_l568_56865


namespace double_root_polynomial_l568_56828

theorem double_root_polynomial (b4 b3 b2 b1 : ℤ) (s : ℤ) :
  (Polynomial.eval s (Polynomial.C 1 * Polynomial.X^5 + Polynomial.C b4 * Polynomial.X^4 + Polynomial.C b3 * Polynomial.X^3 + Polynomial.C b2 * Polynomial.X^2 + Polynomial.C b1 * Polynomial.X + Polynomial.C 24) = 0)
  ∧ (Polynomial.eval s (Polynomial.derivative (Polynomial.C 1 * Polynomial.X^5 + Polynomial.C b4 * Polynomial.X^4 + Polynomial.C b3 * Polynomial.X^3 + Polynomial.C b2 * Polynomial.X^2 + Polynomial.C b1 * Polynomial.X + Polynomial.C 24)) = 0)
  → s = 1 ∨ s = -1 ∨ s = 2 ∨ s = -2 :=
by
  sorry

end double_root_polynomial_l568_56828


namespace farmer_land_l568_56813

theorem farmer_land (A : ℝ) (A_nonneg : A ≥ 0) (cleared_land : ℝ) 
  (soybeans wheat potatoes vegetables corn : ℝ) 
  (h_cleared : cleared_land = 0.95 * A) 
  (h_soybeans : soybeans = 0.35 * cleared_land) 
  (h_wheat : wheat = 0.40 * cleared_land) 
  (h_potatoes : potatoes = 0.15 * cleared_land) 
  (h_vegetables : vegetables = 0.08 * cleared_land) 
  (h_corn : corn = 630) 
  (cleared_sum : soybeans + wheat + potatoes + vegetables + corn = cleared_land) :
  A = 33158 := 
by 
  sorry

end farmer_land_l568_56813


namespace inequality_order_l568_56890

theorem inequality_order (a b : ℝ) (h1 : b > a) (h2 : a > 0) (h3 : a + b = 1) : 
  b > (a^4 - b^4) / (a - b) ∧ (a^4 - b^4) / (a - b) > (a + b) / 2 ∧ (a + b) / 2 > 2 * a * b :=
by 
  sorry

end inequality_order_l568_56890


namespace find_b_value_l568_56859

theorem find_b_value (f : ℝ → ℝ) (f_inv : ℝ → ℝ) (b : ℝ) :
  (∀ x, f x = 1 / (3 * x + b)) →
  (∀ x, f_inv x = (2 - 3 * x) / (3 * x)) →
  b = -3 :=
by
  intros h1 h2
  sorry

end find_b_value_l568_56859


namespace inequality_solution_equality_condition_l568_56840

theorem inequality_solution (a b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ 0) (h3 : b < -1 ∨ b > 0) :
  (1 + a)^2 / (1 + b) ≤ 1 + a^2 / b :=
sorry

theorem equality_condition (a b : ℝ) :
  (1 + a)^2 / (1 + b) = 1 + a^2 / b ↔ a = b :=
sorry

end inequality_solution_equality_condition_l568_56840


namespace max_tied_teams_for_most_wins_l568_56887

-- Definitions based on conditions
def num_teams : ℕ := 7
def total_games_played : ℕ := num_teams * (num_teams - 1) / 2

-- Proposition stating the problem and the expected answer
theorem max_tied_teams_for_most_wins : 
  (∀ (t : ℕ), t ≤ num_teams → ∃ w : ℕ, t * w = total_games_played / num_teams) → 
  t = 7 :=
by
  sorry

end max_tied_teams_for_most_wins_l568_56887


namespace cookies_ratio_l568_56893

theorem cookies_ratio (total_cookies sells_mr_stone brock_buys left_cookies katy_buys : ℕ)
  (h1 : total_cookies = 5 * 12)
  (h2 : sells_mr_stone = 2 * 12)
  (h3 : brock_buys = 7)
  (h4 : left_cookies = 15)
  (h5 : total_cookies - sells_mr_stone - brock_buys - left_cookies = katy_buys) :
  katy_buys / brock_buys = 2 :=
by sorry

end cookies_ratio_l568_56893
