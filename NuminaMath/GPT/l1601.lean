import Mathlib

namespace NUMINAMATH_GPT_complex_number_powers_l1601_160199

theorem complex_number_powers (z : ℂ) (h : z^2 + z + 1 = 0) : 
  z^97 + z^98 + z^99 + z^100 + z^101 = -1 :=
sorry

end NUMINAMATH_GPT_complex_number_powers_l1601_160199


namespace NUMINAMATH_GPT_gcd_360_504_l1601_160111

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end NUMINAMATH_GPT_gcd_360_504_l1601_160111


namespace NUMINAMATH_GPT_selling_price_percentage_l1601_160164

  variable (L : ℝ)  -- List price
  variable (C : ℝ)  -- Cost price after discount
  variable (M : ℝ)  -- Marked price
  variable (S : ℝ)  -- Selling price after discount

  -- Conditions
  def cost_price_condition (L : ℝ) : ℝ := 0.7 * L
  def profit_condition (C S : ℝ) : Prop := 0.75 * S = C
  def marked_price_condition (S M : ℝ) : Prop := 0.85 * M = S

  theorem selling_price_percentage (L : ℝ) (h1 : C = cost_price_condition L)
    (h2 : profit_condition C S) (h3 : marked_price_condition S M) :
    S = 0.9333 * L :=
  by
    -- This is where the proof would go
    sorry
  
end NUMINAMATH_GPT_selling_price_percentage_l1601_160164


namespace NUMINAMATH_GPT_other_root_of_quadratic_l1601_160153

theorem other_root_of_quadratic (a b : ℝ) (h : (1:ℝ) = 1) (h_root : (1:ℝ) ^ 2 + a * (1:ℝ) + 2 = 0): b = 2 :=
by
  sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l1601_160153


namespace NUMINAMATH_GPT_rectangular_field_perimeter_l1601_160115

theorem rectangular_field_perimeter
  (a b : ℝ)
  (diag_eq : a^2 + b^2 = 1156)
  (area_eq : a * b = 240)
  (side_relation : a = 2 * b) :
  2 * (a + b) = 91.2 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_field_perimeter_l1601_160115


namespace NUMINAMATH_GPT_find_x_l1601_160187

def digit_sum (n : ℕ) : ℕ := 
  n.digits 10 |> List.sum

def k := (10^45 - 999999999999999999999999999999999999999999994 : ℕ)

theorem find_x :
  digit_sum k = 397 := 
sorry

end NUMINAMATH_GPT_find_x_l1601_160187


namespace NUMINAMATH_GPT_total_water_carried_l1601_160110

/-- Define the capacities of the four tanks in each truck -/
def tank1_capacity : ℝ := 200
def tank2_capacity : ℝ := 250
def tank3_capacity : ℝ := 300
def tank4_capacity : ℝ := 350

/-- The total capacity of one truck -/
def total_truck_capacity : ℝ := tank1_capacity + tank2_capacity + tank3_capacity + tank4_capacity

/-- Define the fill percentages for each truck -/
def fill_percentage (truck_number : ℕ) : ℝ :=
if truck_number = 1 then 1
else if truck_number = 2 then 0.75
else if truck_number = 3 then 0.5
else if truck_number = 4 then 0.25
else 0

/-- Define the amounts of water each truck carries -/
def water_carried_by_truck (truck_number : ℕ) : ℝ :=
(fill_percentage truck_number) * total_truck_capacity

/-- Prove that the total amount of water the farmer can carry in his trucks is 2750 liters -/
theorem total_water_carried : 
  water_carried_by_truck 1 + water_carried_by_truck 2 + water_carried_by_truck 3 +
  water_carried_by_truck 4 + water_carried_by_truck 5 = 2750 :=
by sorry

end NUMINAMATH_GPT_total_water_carried_l1601_160110


namespace NUMINAMATH_GPT_find_students_that_got_As_l1601_160151

variables (Emily Frank Grace Harry : Prop)

theorem find_students_that_got_As
  (cond1 : Emily → Frank)
  (cond2 : Frank → Grace)
  (cond3 : Grace → Harry)
  (cond4 : Harry → ¬ Emily)
  (three_A_students : ¬ (Emily ∧ Frank ∧ Grace ∧ Harry) ∧
                      (Emily ∧ Frank ∧ Grace ∧ ¬ Harry ∨
                       Emily ∧ Frank ∧ ¬ Grace ∧ Harry ∨
                       Emily ∧ ¬ Frank ∧ Grace ∧ Harry ∨
                       ¬ Emily ∧ Frank ∧ Grace ∧ Harry)) :
  (¬ Emily ∧ Frank ∧ Grace ∧ Harry) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_students_that_got_As_l1601_160151


namespace NUMINAMATH_GPT_calories_350_grams_mint_lemonade_l1601_160160

-- Definitions for the weights of ingredients in grams
def lemon_juice_weight := 150
def sugar_weight := 200
def water_weight := 300
def mint_weight := 50
def total_weight := lemon_juice_weight + sugar_weight + water_weight + mint_weight

-- Definitions for the caloric content per specified weight
def lemon_juice_calories_per_100g := 30
def sugar_calories_per_100g := 400
def mint_calories_per_10g := 7
def water_calories := 0

-- Calculate total calories from each ingredient
def lemon_juice_calories := (lemon_juice_calories_per_100g * lemon_juice_weight) / 100
def sugar_calories := (sugar_calories_per_100g * sugar_weight) / 100
def mint_calories := (mint_calories_per_10g * mint_weight) / 10

-- Calculate total calories in the lemonade
def total_calories := lemon_juice_calories + sugar_calories + mint_calories + water_calories

noncomputable def calories_in_350_grams : ℕ := (total_calories * 350) / total_weight

-- Theorem stating the number of calories in 350 grams of Marco’s lemonade
theorem calories_350_grams_mint_lemonade : calories_in_350_grams = 440 := 
by
  sorry

end NUMINAMATH_GPT_calories_350_grams_mint_lemonade_l1601_160160


namespace NUMINAMATH_GPT_D_is_painting_l1601_160174

def A_activity (act : String) : Prop := 
  act ≠ "walking" ∧ act ≠ "playing basketball"

def B_activity (act : String) : Prop :=
  act ≠ "dancing" ∧ act ≠ "running"

def C_activity_implies_A_activity (C_act A_act : String) : Prop :=
  C_act = "walking" → A_act = "dancing"

def D_activity (act : String) : Prop :=
  act ≠ "playing basketball" ∧ act ≠ "running"

def C_activity (act : String) : Prop :=
  act ≠ "dancing" ∧ act ≠ "playing basketball"

theorem D_is_painting :
  (∃ a b c d : String,
    A_activity a ∧
    B_activity b ∧
    C_activity_implies_A_activity c a ∧
    D_activity d ∧
    C_activity c) →
  ∃ d : String, d = "painting" :=
by
  intros h
  sorry

end NUMINAMATH_GPT_D_is_painting_l1601_160174


namespace NUMINAMATH_GPT_positive_number_percentage_of_itself_is_9_l1601_160184

theorem positive_number_percentage_of_itself_is_9 (x : ℝ) (hx_pos : 0 < x) (h_condition : 0.01 * x^2 = 9) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_positive_number_percentage_of_itself_is_9_l1601_160184


namespace NUMINAMATH_GPT_Victor_Total_Money_l1601_160185

-- Definitions for the conditions
def originalAmount : Nat := 10
def allowance : Nat := 8

-- The proof problem statement
theorem Victor_Total_Money : originalAmount + allowance = 18 := by
  sorry

end NUMINAMATH_GPT_Victor_Total_Money_l1601_160185


namespace NUMINAMATH_GPT_find_second_divisor_l1601_160198

theorem find_second_divisor:
  ∃ x: ℝ, (8900 / 6) / x = 370.8333333333333 ∧ x = 4 :=
sorry

end NUMINAMATH_GPT_find_second_divisor_l1601_160198


namespace NUMINAMATH_GPT_min_trucks_needed_l1601_160188

theorem min_trucks_needed (n : ℕ) (w : ℕ) (t : ℕ) (total_weight : ℕ) (max_box_weight : ℕ) : 
    (total_weight = 10) → 
    (max_box_weight = 1) → 
    (t = 3) →
    (n * max_box_weight = total_weight) →
    (n ≥ 10) →
    ∀ min_trucks : ℕ, (min_trucks * t ≥ total_weight) → 
    min_trucks = 5 :=
by
  intro total_weight_eq max_box_weight_eq truck_capacity box_total_weight_eq n_lower_bound min_trucks min_trucks_condition
  sorry

end NUMINAMATH_GPT_min_trucks_needed_l1601_160188


namespace NUMINAMATH_GPT_measure_of_angle_A_proof_range_of_values_of_b_plus_c_over_a_proof_l1601_160155

noncomputable def measure_of_angle_a (a b c : ℝ) (S : ℝ) (h_c : c = 2) (h_S : b * Real.cos (A / 2) = S) : Prop :=
  A = Real.pi / 3

theorem measure_of_angle_A_proof (a b c : ℝ) (S : ℝ) (h_c : c = 2) (h_S : b * Real.cos (A / 2) = S) : measure_of_angle_a a b c S h_c h_S :=
sorry

noncomputable def range_of_values_of_b_plus_c_over_a (a b c : ℝ) (A : ℝ) (h_A : A = Real.pi / 3) (h_c : c = 2) : Set ℝ :=
  {x : ℝ | 1 < x ∧ x ≤ 2}

theorem range_of_values_of_b_plus_c_over_a_proof (a b c : ℝ) (A : ℝ) (h_A : A = Real.pi / 3) (h_c : c = 2) : 
  ∃ x, x ∈ range_of_values_of_b_plus_c_over_a a b c A h_A h_c :=
sorry

end NUMINAMATH_GPT_measure_of_angle_A_proof_range_of_values_of_b_plus_c_over_a_proof_l1601_160155


namespace NUMINAMATH_GPT_sara_quarters_l1601_160166

theorem sara_quarters (initial_quarters : ℕ) (additional_quarters : ℕ) (total_quarters : ℕ) 
    (h1 : initial_quarters = 21) 
    (h2 : additional_quarters = 49) 
    (h3 : total_quarters = initial_quarters + additional_quarters) : 
    total_quarters = 70 :=
sorry

end NUMINAMATH_GPT_sara_quarters_l1601_160166


namespace NUMINAMATH_GPT_find_k_l1601_160149

theorem find_k (k : ℝ) (h1 : k > 1) 
(h2 : ∑' n : ℕ, (7 * (n + 1) - 3) / k^(n + 1) = 2) : 
  k = 2 + 3 * Real.sqrt 2 / 2 := 
sorry

end NUMINAMATH_GPT_find_k_l1601_160149


namespace NUMINAMATH_GPT_count_squares_and_cubes_l1601_160132

theorem count_squares_and_cubes (bound : ℕ) (hk : bound = 1000) : 
  Nat.card {n : ℕ | n < bound ∧ (∃ k : ℕ, n = k ^ 6)} = 3 := 
by
  sorry

end NUMINAMATH_GPT_count_squares_and_cubes_l1601_160132


namespace NUMINAMATH_GPT_remainder_div_7_l1601_160119

theorem remainder_div_7 (k : ℕ) (h1 : k % 5 = 2) (h2 : k % 6 = 5) (h3 : k < 39) : k % 7 = 3 :=
sorry

end NUMINAMATH_GPT_remainder_div_7_l1601_160119


namespace NUMINAMATH_GPT_find_m_if_extraneous_root_l1601_160168

theorem find_m_if_extraneous_root :
  (∃ x : ℝ, x = 2 ∧ (∀ z : ℝ, z ≠ 2 → (m / (z-2) - 2*z / (2-z) = 1)) ∧ m = -4) :=
sorry

end NUMINAMATH_GPT_find_m_if_extraneous_root_l1601_160168


namespace NUMINAMATH_GPT_count_valid_triangles_l1601_160193

def triangle_area (a b c : ℕ) : ℕ :=
  let s := (a + b + c) / 2
  s * (s - a) * (s - b) * (s - c)

noncomputable def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b + c < 20 ∧ a ≤ b ∧ b ≤ c ∧ a + b > c ∧ a < b ∧ b < c ∧ a^2 + b^2 ≠ c^2

theorem count_valid_triangles : { n : ℕ // n = 24 } :=
  sorry

end NUMINAMATH_GPT_count_valid_triangles_l1601_160193


namespace NUMINAMATH_GPT_least_product_of_distinct_primes_greater_than_50_l1601_160140

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def distinct_primes_greater_than_50 (p q : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ p ≠ q ∧ p > 50 ∧ q > 50

theorem least_product_of_distinct_primes_greater_than_50 :
  ∃ (p q : ℕ), distinct_primes_greater_than_50 p q ∧ p * q = 3127 :=
by
  sorry

end NUMINAMATH_GPT_least_product_of_distinct_primes_greater_than_50_l1601_160140


namespace NUMINAMATH_GPT_time_for_embankments_l1601_160147

theorem time_for_embankments (rate : ℚ) (t1 t2 : ℕ) (w1 w2 : ℕ)
    (h1 : w1 = 75) (h2 : w2 = 60) (h3 : t1 = 4)
    (h4 : rate = 1 / (w1 * t1 : ℚ)) 
    (h5 : t2 = 1 / (w2 * rate)) : 
    t1 + t2 = 9 :=
sorry

end NUMINAMATH_GPT_time_for_embankments_l1601_160147


namespace NUMINAMATH_GPT_sought_circle_equation_l1601_160107

def circle_passing_through_point (D E F : ℝ) : Prop :=
  ∀ (x y : ℝ), (x = 0) → (y = 2) → x^2 + y^2 + D * x + E * y + F = 0

def chord_lies_on_line (D E F : ℝ) : Prop :=
  (D + 1) / 5 = (E - 2) / 2 ∧ (D + 1) / 5 = (F + 3)

theorem sought_circle_equation :
  ∃ (D E F : ℝ), 
  circle_passing_through_point D E F ∧ 
  chord_lies_on_line D E F ∧
  (D = -6) ∧ (E = 0) ∧ (F = -4) ∧ 
  ∀ x y : ℝ, x^2 + y^2 + D * x + E * y + F = 0 ↔ x^2 + y^2 - 6 * x - 4 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sought_circle_equation_l1601_160107


namespace NUMINAMATH_GPT_marco_strawberries_weight_l1601_160194

theorem marco_strawberries_weight 
  (m : ℕ) 
  (total_weight : ℕ := 40) 
  (dad_weight : ℕ := 32) 
  (h : total_weight = m + dad_weight) : 
  m = 8 := 
sorry

end NUMINAMATH_GPT_marco_strawberries_weight_l1601_160194


namespace NUMINAMATH_GPT_a_7_is_127_l1601_160159

-- Define the sequence {a_n}
def a : ℕ → ℕ
| 0       => 0  -- Define a_0 which is not used but useful for indexing
| 1       => 1
| (n + 2) => 2 * (a (n + 1)) + 1

-- Prove that a_7 = 127
theorem a_7_is_127 : a 7 = 127 := 
sorry

end NUMINAMATH_GPT_a_7_is_127_l1601_160159


namespace NUMINAMATH_GPT_area_of_parallelogram_l1601_160105

def parallelogram_base : ℝ := 26
def parallelogram_height : ℝ := 14

theorem area_of_parallelogram : parallelogram_base * parallelogram_height = 364 := by
  sorry

end NUMINAMATH_GPT_area_of_parallelogram_l1601_160105


namespace NUMINAMATH_GPT_solve_system_of_equations_l1601_160109

theorem solve_system_of_equations 
  (x y : ℝ) 
  (h1 : x / 3 - (y + 1) / 2 = 1) 
  (h2 : 4 * x - (2 * y - 5) = 11) : 
  x = 0 ∧ y = -3 :=
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l1601_160109


namespace NUMINAMATH_GPT_sum_of_edge_lengths_of_truncated_octahedron_prism_l1601_160125

-- Define the vertices, edge length, and the assumption of the prism being a truncated octahedron
def prism_vertices : ℕ := 24
def edge_length : ℕ := 5
def truncated_octahedron_edges : ℕ := 36

-- The Lean statement to prove the sum of edge lengths
theorem sum_of_edge_lengths_of_truncated_octahedron_prism :
  prism_vertices = 24 ∧ edge_length = 5 ∧ truncated_octahedron_edges = 36 →
  truncated_octahedron_edges * edge_length = 180 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_edge_lengths_of_truncated_octahedron_prism_l1601_160125


namespace NUMINAMATH_GPT_mean_equality_l1601_160157

-- Define average calculation function
def average (a b c : ℕ) : ℕ :=
  (a + b + c) / 3

def average_two (a b : ℕ) : ℕ :=
  (a + b) / 2

theorem mean_equality (x : ℕ) 
  (h : average 8 16 24 = average_two 10 x) : 
  x = 22 :=
by {
  -- The actual proof is here
  sorry
}

end NUMINAMATH_GPT_mean_equality_l1601_160157


namespace NUMINAMATH_GPT_log_10_850_consecutive_integers_l1601_160145

theorem log_10_850_consecutive_integers : 
  (2:ℝ) < Real.log 850 / Real.log 10 ∧ Real.log 850 / Real.log 10 < (3:ℝ) →
  ∃ (a b : ℕ), (a = 2) ∧ (b = 3) ∧ (2 < Real.log 850 / Real.log 10) ∧ (Real.log 850 / Real.log 10 < 3) ∧ (a + b = 5) :=
by
  sorry

end NUMINAMATH_GPT_log_10_850_consecutive_integers_l1601_160145


namespace NUMINAMATH_GPT_length_of_bridge_l1601_160122

theorem length_of_bridge (t : ℝ) (s : ℝ) (d : ℝ) : 
  (t = 24 / 60) ∧ (s = 10) ∧ (d = s * t) → d = 4 := by
  sorry

end NUMINAMATH_GPT_length_of_bridge_l1601_160122


namespace NUMINAMATH_GPT_great_wall_scientific_notation_l1601_160118

theorem great_wall_scientific_notation :
  6700000 = 6.7 * 10^6 :=
sorry

end NUMINAMATH_GPT_great_wall_scientific_notation_l1601_160118


namespace NUMINAMATH_GPT_correct_propositions_l1601_160197

-- Definitions of parallel and perpendicular
def parallel (x y : Type) : Prop := sorry
def perpendicular (x y : Type) : Prop := sorry

-- Main theorem
theorem correct_propositions (m n α β γ : Type) :
  ( (parallel m α ∧ parallel n β ∧ parallel α β → parallel m n) ∧
    (parallel α γ ∧ parallel β γ → parallel α β) ∧
    (perpendicular m α ∧ perpendicular n β ∧ parallel α β → parallel m n) ∧
    (perpendicular α γ ∧ perpendicular β γ → parallel α β) ) →
  ( (parallel α γ ∧ parallel β γ → parallel α β) ∧
    (perpendicular m α ∧ perpendicular n β ∧ parallel α β → parallel m n) ) :=
  sorry

end NUMINAMATH_GPT_correct_propositions_l1601_160197


namespace NUMINAMATH_GPT_math_problem_l1601_160123

theorem math_problem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 1 / y) :
  (x - 1 / x) * (y + 1 / y) = x^2 - y^2 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l1601_160123


namespace NUMINAMATH_GPT_intersection_of_sets_l1601_160161

theorem intersection_of_sets :
  let M := { x : ℝ | -3 < x ∧ x ≤ 5 }
  let N := { x : ℝ | -5 < x ∧ x < 5 }
  M ∩ N = { x : ℝ | -3 < x ∧ x < 5 } := 
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l1601_160161


namespace NUMINAMATH_GPT_hyperbola_asymptote_l1601_160186

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, (x^2 - y^2 / a^2) = 1 → (y = 2*x ∨ y = -2*x)) → a = 2 :=
by
  intro h_asymptote
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_l1601_160186


namespace NUMINAMATH_GPT_quadratic_roots_relation_l1601_160133

theorem quadratic_roots_relation (a b c d : ℝ) (h : ∀ x : ℝ, (c * x^2 + d * x + a = 0) → 
  (a * (2007 * x)^2 + b * (2007 * x) + c = 0)) : b^2 = d^2 := 
sorry

end NUMINAMATH_GPT_quadratic_roots_relation_l1601_160133


namespace NUMINAMATH_GPT_min_total_trees_l1601_160121

theorem min_total_trees (L X : ℕ) (h1: 13 * L < 100 * X) (h2: 100 * X < 14 * L) : L ≥ 15 :=
  sorry

end NUMINAMATH_GPT_min_total_trees_l1601_160121


namespace NUMINAMATH_GPT_root_polynomial_sum_l1601_160144

theorem root_polynomial_sum {b c : ℝ} (hb : b^2 - b - 1 = 0) (hc : c^2 - c - 1 = 0) : 
  (1 / (1 - b)) + (1 / (1 - c)) = -1 := 
sorry

end NUMINAMATH_GPT_root_polynomial_sum_l1601_160144


namespace NUMINAMATH_GPT_number_of_houses_on_block_l1601_160163

theorem number_of_houses_on_block 
  (total_mail : ℕ) 
  (white_mailboxes : ℕ) 
  (red_mailboxes : ℕ) 
  (mail_per_house : ℕ) 
  (total_white_mail : ℕ) 
  (total_red_mail : ℕ) 
  (remaining_mail : ℕ)
  (additional_houses : ℕ)
  (total_houses : ℕ) :
  total_mail = 48 ∧ 
  white_mailboxes = 2 ∧ 
  red_mailboxes = 3 ∧ 
  mail_per_house = 6 ∧ 
  total_white_mail = white_mailboxes * mail_per_house ∧
  total_red_mail = red_mailboxes * mail_per_house ∧
  remaining_mail = total_mail - (total_white_mail + total_red_mail) ∧
  additional_houses = remaining_mail / mail_per_house ∧
  total_houses = white_mailboxes + red_mailboxes + additional_houses →
  total_houses = 8 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_houses_on_block_l1601_160163


namespace NUMINAMATH_GPT_prism_volume_l1601_160143

theorem prism_volume 
    (x y z : ℝ) 
    (h_xy : x * y = 18) 
    (h_yz : y * z = 12) 
    (h_xz : x * z = 8) 
    (h_longest_shortest : max x (max y z) = 2 * min x (min y z)) : 
    x * y * z = 16 := 
  sorry

end NUMINAMATH_GPT_prism_volume_l1601_160143


namespace NUMINAMATH_GPT_arithmetic_sequence_a1_l1601_160141

/-- In an arithmetic sequence {a_n],
given a_3 = -2, a_n = 3 / 2, and S_n = -15 / 2,
prove that the value of a_1 is -3 or -19 / 6.
-/
theorem arithmetic_sequence_a1 (a_n S_n : ℕ → ℚ)
  (h1 : a_n 3 = -2)
  (h2 : ∃ n : ℕ, a_n n = 3 / 2)
  (h3 : ∃ n : ℕ, S_n n = -15 / 2) :
  ∃ x : ℚ, x = -3 ∨ x = -19 / 6 :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a1_l1601_160141


namespace NUMINAMATH_GPT_ratio_fifteenth_term_l1601_160165

-- Definitions of S_n and T_n based on the given conditions
def S_n (a d n : ℕ) : ℕ := n * (2 * a + (n - 1) * d) / 2
def T_n (b e n : ℕ) : ℕ := n * (2 * b + (n - 1) * e) / 2

-- Statement of the problem
theorem ratio_fifteenth_term 
  (a b d e : ℕ) 
  (h : ∀ n, (S_n a d n : ℚ) / (T_n b e n : ℚ) = (9 * n + 5) / (6 * n + 31)) : 
  (a + 14 * d : ℚ) / (b + 14 * e : ℚ) = (92 : ℚ) / 71 :=
by sorry

end NUMINAMATH_GPT_ratio_fifteenth_term_l1601_160165


namespace NUMINAMATH_GPT_intersection_of_sets_l1601_160173

noncomputable def U : Set ℝ := Set.univ

noncomputable def M : Set ℝ := {x | x < -1 ∨ x > 1}

noncomputable def N : Set ℝ := {x | 0 < x ∧ x < 2}

noncomputable def complement_U_M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

noncomputable def intersection_N_complement_U_M : Set ℝ := {x | 0 < x ∧ x ≤ 1}

theorem intersection_of_sets :
  N ∩ complement_U_M = intersection_N_complement_U_M := 
sorry

end NUMINAMATH_GPT_intersection_of_sets_l1601_160173


namespace NUMINAMATH_GPT_pigs_remaining_l1601_160112

def initial_pigs : ℕ := 364
def pigs_joined : ℕ := 145
def pigs_moved : ℕ := 78

theorem pigs_remaining : initial_pigs + pigs_joined - pigs_moved = 431 := by
  sorry

end NUMINAMATH_GPT_pigs_remaining_l1601_160112


namespace NUMINAMATH_GPT_proof_problem_l1601_160191

noncomputable def question (a b c d m : ℚ) : ℚ :=
  2 * a + 2 * b + (a + b - 3 * (c * d)) - m

def condition1 (m : ℚ) : Prop :=
  abs (m + 1) = 4

def condition2 (a b : ℚ) : Prop :=
  a = -b

def condition3 (c d : ℚ) : Prop :=
  c * d = 1

theorem proof_problem (a b c d m : ℚ) :
  condition1 m → condition2 a b → condition3 c d →
  (question a b c d m = 2 ∨ question a b c d m = -6) :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l1601_160191


namespace NUMINAMATH_GPT_sum_s_h_e_base_three_l1601_160171

def distinct_non_zero_digits (S H E : ℕ) : Prop :=
  S ≠ 0 ∧ H ≠ 0 ∧ E ≠ 0 ∧ S < 3 ∧ H < 3 ∧ E < 3 ∧ S ≠ H ∧ H ≠ E ∧ S ≠ E

def base_three_addition (S H E : ℕ) :=
  (S + H * 3 + E * 9) + (H + E * 3) == (H * 3 + S * 9 + S*27)

theorem sum_s_h_e_base_three (S H E : ℕ) (h1 : distinct_non_zero_digits S H E) (h2 : base_three_addition S H E) :
  (S + H + E = 5) := by sorry

end NUMINAMATH_GPT_sum_s_h_e_base_three_l1601_160171


namespace NUMINAMATH_GPT_number_of_triangles_l1601_160120

theorem number_of_triangles (x y : ℕ) (P Q : ℕ × ℕ) (O : ℕ × ℕ := (0,0)) (area : ℕ) :
  (P ≠ Q) ∧ (P.1 * 31 + P.2 = 2023) ∧ (Q.1 * 31 + Q.2 = 2023) ∧ 
  (P.1 ≠ Q.1 → P.1 - Q.1 = n ∧ 2023 * n % 6 = 0) → area = 165 :=
sorry

end NUMINAMATH_GPT_number_of_triangles_l1601_160120


namespace NUMINAMATH_GPT_number_of_lines_through_point_intersect_hyperbola_once_l1601_160169

noncomputable def hyperbola (x y : ℝ) : Prop :=
  x^2 / 4 - y^2 = 1

noncomputable def point_P : ℝ × ℝ :=
  (-4, 1)

noncomputable def line_through (P : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop :=
  l P

noncomputable def one_point_intersection (l : ℝ × ℝ → Prop) (H : ℝ → ℝ → Prop) : Prop :=
  ∃! p : ℝ × ℝ, l p ∧ H p.1 p.2

theorem number_of_lines_through_point_intersect_hyperbola_once :
  (∃ (l₁ l₂ : ℝ × ℝ → Prop),
    line_through point_P l₁ ∧
    line_through point_P l₂ ∧
    one_point_intersection l₁ hyperbola ∧
    one_point_intersection l₂ hyperbola ∧
    l₁ ≠ l₂) ∧ ¬ (∃ (l₃ : ℝ × ℝ → Prop),
    line_through point_P l₃ ∧
    one_point_intersection l₃ hyperbola ∧
    ∃! (other_line : ℝ × ℝ → Prop),
    line_through point_P other_line ∧
    one_point_intersection other_line hyperbola ∧
    l₃ ≠ other_line) :=
sorry

end NUMINAMATH_GPT_number_of_lines_through_point_intersect_hyperbola_once_l1601_160169


namespace NUMINAMATH_GPT_trig_identity_l1601_160113

theorem trig_identity (α : ℝ) (h1 : (-Real.pi / 2) < α ∧ α < 0)
  (h2 : Real.sin α + Real.cos α = 1 / 5) :
  1 / (Real.cos α ^ 2 - Real.sin α ^ 2) = 25 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_trig_identity_l1601_160113


namespace NUMINAMATH_GPT_find_square_side_length_l1601_160101

noncomputable def square_side_length (a : ℝ) : Prop :=
  let angle_deg := 30
  let a_sqr_minus_1 := Real.sqrt (a ^ 2 - 1)
  let a_sqr_minus_4 := Real.sqrt (a ^ 2 - 4)
  let dihedral_cos := Real.cos (Real.pi / 6)  -- 30 degrees in radians
  let dihedral_sin := Real.sin (Real.pi / 6)
  let area_1 := 0.5 * a_sqr_minus_1 * a_sqr_minus_4 * dihedral_sin
  let area_2 := 0.5 * Real.sqrt (a ^ 4 - 5 * a ^ 2)
  dihedral_cos = (Real.sqrt 3 / 2) -- Using the provided angle
  ∧ dihedral_sin = 0.5
  ∧ area_1 = area_2
  ∧ a = 2 * Real.sqrt 5

-- The theorem stating that the side length of the square is 2\sqrt{5}
theorem find_square_side_length (a : ℝ) (H : square_side_length a) : a = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_GPT_find_square_side_length_l1601_160101


namespace NUMINAMATH_GPT_tan_150_deg_l1601_160137

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end NUMINAMATH_GPT_tan_150_deg_l1601_160137


namespace NUMINAMATH_GPT_Zenobius_more_descendants_l1601_160138

/-- Total number of descendants in King Pafnutius' lineage --/
def descendants_Pafnutius : Nat :=
  2 + 60 * 2 + 20 * 1

/-- Total number of descendants in King Zenobius' lineage --/
def descendants_Zenobius : Nat :=
  4 + 35 * 3 + 35 * 1

theorem Zenobius_more_descendants : descendants_Zenobius > descendants_Pafnutius := by
  sorry

end NUMINAMATH_GPT_Zenobius_more_descendants_l1601_160138


namespace NUMINAMATH_GPT_converse_false_l1601_160134

variable {a b : ℝ}

theorem converse_false : (¬ (∀ a b : ℝ, (ab = 0 → a = 0))) :=
by
  sorry

end NUMINAMATH_GPT_converse_false_l1601_160134


namespace NUMINAMATH_GPT_cubic_roots_c_div_d_l1601_160104

theorem cubic_roots_c_div_d (a b c d : ℚ) :
  (∀ x, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = -1 ∨ x = 1/2 ∨ x = 4) →
  (c / d = 9 / 4) :=
by
  intros h
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_cubic_roots_c_div_d_l1601_160104


namespace NUMINAMATH_GPT_February_March_Ratio_l1601_160116

theorem February_March_Ratio (J F M : ℕ) (h1 : F = 2 * J) (h2 : M = 8800) (h3 : J + F + M = 12100) : F / M = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_February_March_Ratio_l1601_160116


namespace NUMINAMATH_GPT_cornbread_pieces_count_l1601_160170

def cornbread_pieces (pan_length pan_width piece_length piece_width : ℕ) : ℕ := 
  (pan_length * pan_width) / (piece_length * piece_width)

theorem cornbread_pieces_count :
  cornbread_pieces 24 20 3 3 = 53 :=
by
  -- The definitions and the equivalence transformation tell us that this is true
  sorry

end NUMINAMATH_GPT_cornbread_pieces_count_l1601_160170


namespace NUMINAMATH_GPT_h_at_neg_one_l1601_160128

-- Definitions based on the conditions
def f (x : ℝ) : ℝ := 3 * x + 6
def g (x : ℝ) : ℝ := x ^ 3
def h (x : ℝ) : ℝ := f (g x)

-- The main statement to prove
theorem h_at_neg_one : h (-1) = 3 := by
  sorry

end NUMINAMATH_GPT_h_at_neg_one_l1601_160128


namespace NUMINAMATH_GPT_paul_eats_sandwiches_l1601_160103

theorem paul_eats_sandwiches (S : ℕ) (h : (S + 2 * S + 4 * S) * 2 = 28) : S = 2 :=
by
  sorry

end NUMINAMATH_GPT_paul_eats_sandwiches_l1601_160103


namespace NUMINAMATH_GPT_problem1_problem2_l1601_160154

-- Define the function f(x) = |x + 2| + |x - 1|
def f (x : ℝ) : ℝ := |x + 2| + |x - 1|

-- 1. Prove the solution set of f(x) > 5 is {x | x < -3 or x > 2}
theorem problem1 : {x : ℝ | f x > 5} = {x : ℝ | x < -3 ∨ x > 2} :=
by
  sorry

-- 2. Prove that if f(x) ≥ a^2 - 2a always holds, then -1 ≤ a ≤ 3
theorem problem2 (a : ℝ) (h : ∀ x : ℝ, f x ≥ a^2 - 2 * a) : -1 ≤ a ∧ a ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1601_160154


namespace NUMINAMATH_GPT_cost_of_each_book_is_six_l1601_160167

-- Define variables for the number of books bought
def books_about_animals := 8
def books_about_outer_space := 6
def books_about_trains := 3

-- Define the total number of books
def total_books := books_about_animals + books_about_outer_space + books_about_trains

-- Define the total amount spent
def total_amount_spent := 102

-- Define the cost per book
def cost_per_book := total_amount_spent / total_books

-- Prove that the cost per book is $6
theorem cost_of_each_book_is_six : cost_per_book = 6 := by
  sorry

end NUMINAMATH_GPT_cost_of_each_book_is_six_l1601_160167


namespace NUMINAMATH_GPT_correctness_of_option_C_l1601_160177

noncomputable def vec_a : ℝ × ℝ := (-1/2, Real.sqrt 3 / 2)
noncomputable def vec_b : ℝ × ℝ := (Real.sqrt 3 / 2, -1/2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def is_orthogonal (u v : ℝ × ℝ) : Prop :=
  dot_product u v = 0

theorem correctness_of_option_C :
  is_orthogonal (vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2) :=
by
  sorry

end NUMINAMATH_GPT_correctness_of_option_C_l1601_160177


namespace NUMINAMATH_GPT_algebraic_expression_value_l1601_160152

theorem algebraic_expression_value (x : ℝ) (h : x^2 + 3 * x - 3 = 0) : x^3 + 2 * x^2 - 6 * x + 3 = 0 := 
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1601_160152


namespace NUMINAMATH_GPT_determine_120_percent_of_y_l1601_160192

def x := 0.80 * 350
def y := 0.60 * x
def result := 1.20 * y

theorem determine_120_percent_of_y : result = 201.6 := by
  sorry

end NUMINAMATH_GPT_determine_120_percent_of_y_l1601_160192


namespace NUMINAMATH_GPT_circle_inscribed_isosceles_trapezoid_l1601_160179

theorem circle_inscribed_isosceles_trapezoid (r a c : ℝ) : 
  (∃ base1 base2 : ℝ,  2 * a = base1 ∧ 2 * c = base2) →
  (∃ O : ℝ, O = r) →
  r^2 = a * c :=
by
  sorry

end NUMINAMATH_GPT_circle_inscribed_isosceles_trapezoid_l1601_160179


namespace NUMINAMATH_GPT_initial_population_l1601_160136

theorem initial_population (P : ℝ) (h : P * (1.24 : ℝ)^2 = 18451.2) : P = 12000 :=
by
  sorry

end NUMINAMATH_GPT_initial_population_l1601_160136


namespace NUMINAMATH_GPT_find_f_of_2_l1601_160158

variable (f : ℝ → ℝ)

-- Given condition: f is the inverse function of the exponential function 2^x
def inv_function : Prop := ∀ x, f (2^x) = x ∧ 2^(f x) = x

theorem find_f_of_2 (h : inv_function f) : f 2 = 1 :=
by sorry

end NUMINAMATH_GPT_find_f_of_2_l1601_160158


namespace NUMINAMATH_GPT_color_swap_rectangle_l1601_160172

theorem color_swap_rectangle 
  (n : ℕ) 
  (square_size : ℕ := 2*n - 1) 
  (colors : Finset ℕ := Finset.range n) 
  (vertex_colors : Fin (square_size + 1) × Fin (square_size + 1) → ℕ) 
  (h_vertex_colors : ∀ v, vertex_colors v ∈ colors) :
  ∃ row, ∃ (v₁ v₂ : Fin (square_size + 1) × Fin (square_size + 1)),
    (v₁.1 = row ∧ v₂.1 = row ∧ v₁ ≠ v₂ ∧
    (∃ r₀ r₁ r₂, r₀ ≠ r₁ ∧ r₁ ≠ r₂ ∧ r₂ ≠ r₀ ∧
    vertex_colors v₁ = vertex_colors (r₀, v₁.2) ∧
    vertex_colors v₂ = vertex_colors (r₀, v₂.2) ∧
    vertex_colors (r₁, v₁.2) = vertex_colors (r₂, v₂.2))) := 
sorry

end NUMINAMATH_GPT_color_swap_rectangle_l1601_160172


namespace NUMINAMATH_GPT_total_trees_in_park_l1601_160124

theorem total_trees_in_park (oak_planted_total maple_planted_total birch_planted_total : ℕ)
  (initial_oak initial_maple initial_birch : ℕ)
  (oak_removed_day2 maple_removed_day2 birch_removed_day2 : ℕ)
  (D1_oak_plant : ℕ) (D2_oak_plant : ℕ) (D1_maple_plant : ℕ) (D2_maple_plant : ℕ)
  (D1_birch_plant : ℕ) (D2_birch_plant : ℕ):
  initial_oak = 25 → initial_maple = 40 → initial_birch = 20 →
  oak_planted_total = 73 → maple_planted_total = 52 → birch_planted_total = 35 →
  D1_oak_plant = 29 → D2_oak_plant = 26 →
  D1_maple_plant = 26 → D2_maple_plant = 13 →
  D1_birch_plant = 10 → D2_birch_plant = 16 →
  oak_removed_day2 = 15 → maple_removed_day2 = 10 → birch_removed_day2 = 5 →
  (initial_oak + oak_planted_total - oak_removed_day2) +
  (initial_maple + maple_planted_total - maple_removed_day2) +
  (initial_birch + birch_planted_total - birch_removed_day2) = 215 :=
by
  intros h_initial_oak h_initial_maple h_initial_birch
         h_oak_planted_total h_maple_planted_total h_birch_planted_total
         h_D1_oak h_D2_oak h_D1_maple h_D2_maple h_D1_birch h_D2_birch
         h_oak_removed h_maple_removed h_birch_removed
  sorry

end NUMINAMATH_GPT_total_trees_in_park_l1601_160124


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_l1601_160127
-- Import the Mathlib library for mathematical tools and structures

-- Define the condition for the ellipse and the arithmetic sequence
variables {a b c : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : 2 * b = a + c) (h4 : b^2 = a^2 - c^2)

-- State the theorem to prove
theorem eccentricity_of_ellipse : ∃ e : ℝ, e = 3 / 5 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_l1601_160127


namespace NUMINAMATH_GPT_derivative_at_pi_div_2_l1601_160175

noncomputable def f (x : ℝ) : ℝ := x * Real.sin (2 * x)

theorem derivative_at_pi_div_2 : deriv f (Real.pi / 2) = -Real.pi := by
  sorry

end NUMINAMATH_GPT_derivative_at_pi_div_2_l1601_160175


namespace NUMINAMATH_GPT_sin_30_plus_cos_60_l1601_160139

-- Define the trigonometric evaluations as conditions
def sin_30_degree := 1 / 2
def cos_60_degree := 1 / 2

-- Lean statement for proving the sum of these values
theorem sin_30_plus_cos_60 : sin_30_degree + cos_60_degree = 1 := by
  sorry

end NUMINAMATH_GPT_sin_30_plus_cos_60_l1601_160139


namespace NUMINAMATH_GPT_triangle_inequality_l1601_160106

variable {a b c S n : ℝ}

theorem triangle_inequality (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
(habc : a + b > c) (habc' : a + c > b) (habc'' : b + c > a)
(hS : 2 * S = a + b + c) (hn : n ≥ 1) :
  (a^n / (b + c)) + (b^n / (c + a)) + (c^n / (a + b)) ≥ ((2 / 3)^(n - 2)) * S^(n - 1) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1601_160106


namespace NUMINAMATH_GPT_number_of_men_l1601_160182

theorem number_of_men (M W C : ℕ) 
  (h1 : M + W + C = 10000)
  (h2 : C = 2500)
  (h3 : C = 5 * W) : 
  M = 7000 := 
by
  sorry

end NUMINAMATH_GPT_number_of_men_l1601_160182


namespace NUMINAMATH_GPT_mixed_operation_with_rationals_l1601_160148

theorem mixed_operation_with_rationals :
  (- (2 / 21)) / (1 / 6 - 3 / 14 + 2 / 3 - 9 / 7) = 1 / 7 := 
by 
  sorry

end NUMINAMATH_GPT_mixed_operation_with_rationals_l1601_160148


namespace NUMINAMATH_GPT_problem_solution_l1601_160131

theorem problem_solution
  (P Q R S : ℕ)
  (h1 : 2 * Q = P + R)
  (h2 : R * R = Q * S)
  (h3 : R = 4 * Q / 3) :
  P + Q + R + S = 171 :=
by sorry

end NUMINAMATH_GPT_problem_solution_l1601_160131


namespace NUMINAMATH_GPT_area_of_paper_l1601_160146

theorem area_of_paper (L W : ℕ) (h1 : 2 * L + W = 34) (h2 : L + 2 * W = 38) : 
  L * W = 140 := 
by sorry

end NUMINAMATH_GPT_area_of_paper_l1601_160146


namespace NUMINAMATH_GPT_no_real_solution_arctan_eqn_l1601_160135

theorem no_real_solution_arctan_eqn :
  ¬∃ x : ℝ, 0 < x ∧ (Real.arctan (1 / x ^ 2) + Real.arctan (1 / x ^ 4) = (Real.pi / 4)) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_arctan_eqn_l1601_160135


namespace NUMINAMATH_GPT_custom_operation_correct_l1601_160162

noncomputable def custom_operation (a b c : ℕ) : ℝ :=
  (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)

theorem custom_operation_correct : custom_operation 6 15 5 = 2 := by
  sorry

end NUMINAMATH_GPT_custom_operation_correct_l1601_160162


namespace NUMINAMATH_GPT_deceased_member_income_l1601_160114

theorem deceased_member_income (a b c d : ℝ)
    (h1 : a = 735) 
    (h2 : b = 650)
    (h3 : c = 4 * 735)
    (h4 : d = 3 * 650) :
    c - d = 990 := by
  sorry

end NUMINAMATH_GPT_deceased_member_income_l1601_160114


namespace NUMINAMATH_GPT_odometer_reading_at_lunch_l1601_160102

axiom odometer_start : ℝ
axiom miles_traveled : ℝ
axiom odometer_at_lunch : ℝ
axiom starting_reading : odometer_start = 212.3
axiom travel_distance : miles_traveled = 159.7
axiom at_lunch_reading : odometer_at_lunch = odometer_start + miles_traveled

theorem odometer_reading_at_lunch :
  odometer_at_lunch = 372.0 :=
  by
  sorry

end NUMINAMATH_GPT_odometer_reading_at_lunch_l1601_160102


namespace NUMINAMATH_GPT_parabola_shifted_left_and_down_l1601_160117

-- Define the initial parabola
def initial_parabola (x : ℝ) : ℝ :=
  3 * (x - 4) ^ 2 + 3

-- Define the transformation (shift 4 units to the left and 4 units down)
def transformed_parabola (x : ℝ) : ℝ :=
  initial_parabola (x + 4) - 4

-- Prove that after transformation the given parabola becomes y = 3x^2 - 1
theorem parabola_shifted_left_and_down :
  ∀ x : ℝ, transformed_parabola x = 3 * x ^ 2 - 1 := 
by 
  sorry

end NUMINAMATH_GPT_parabola_shifted_left_and_down_l1601_160117


namespace NUMINAMATH_GPT_toothpick_removal_l1601_160129

noncomputable def removalStrategy : ℕ :=
  let numToothpicks := 60
  let numUpward1Triangles := 22
  let numDownward1Triangles := 14
  let numUpward2Triangles := 4

  -- minimum toothpicks to remove to achieve the goal
  15

theorem toothpick_removal :
  let numToothpicks := 60
  let numUpward1Triangles := 22
  let numDownward1Triangles := 14
  let numUpward2Triangles := 4
  removalStrategy = 15 := by
  sorry

end NUMINAMATH_GPT_toothpick_removal_l1601_160129


namespace NUMINAMATH_GPT_no_integers_satisfying_polynomials_l1601_160100

theorem no_integers_satisfying_polynomials 
: ¬ ∃ (a b c d : ℤ), a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧ a * 62^3 + b * 62^2 + c * 62 + d = 2 := 
by
  sorry

end NUMINAMATH_GPT_no_integers_satisfying_polynomials_l1601_160100


namespace NUMINAMATH_GPT_students_only_in_math_l1601_160126

-- Define the sets and their cardinalities according to the problem conditions
def total_students : ℕ := 120
def math_students : ℕ := 85
def foreign_language_students : ℕ := 65
def sport_students : ℕ := 50
def all_three_classes : ℕ := 10

-- Define the Lean theorem to prove the number of students taking only a math class
theorem students_only_in_math (total : ℕ) (M F S : ℕ) (MFS : ℕ)
  (H_total : total = 120)
  (H_M : M = 85)
  (H_F : F = 65)
  (H_S : S = 50)
  (H_MFS : MFS = 10) :
  (M - (MFS + MFS - MFS) = 35) :=
sorry

end NUMINAMATH_GPT_students_only_in_math_l1601_160126


namespace NUMINAMATH_GPT_inequality_chain_l1601_160178

open Real

theorem inequality_chain (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > a * b ∧ a * b > b^2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_chain_l1601_160178


namespace NUMINAMATH_GPT_find_pairs_solution_l1601_160183

theorem find_pairs_solution (x y : ℝ) :
  (x^3 + x^2 * y + x * y^2 + y^3 = 8 * (x^2 + x * y + y^2 + 1)) ↔ 
  (x, y) = (8, -2) ∨ (x, y) = (-2, 8) ∨ 
  (x, y) = (4 + Real.sqrt 15, 4 - Real.sqrt 15) ∨ 
  (x, y) = (4 - Real.sqrt 15, 4 + Real.sqrt 15) :=
by 
  sorry

end NUMINAMATH_GPT_find_pairs_solution_l1601_160183


namespace NUMINAMATH_GPT_average_monthly_income_l1601_160150

theorem average_monthly_income (P Q R : ℝ) (h1 : (P + Q) / 2 = 5050)
  (h2 : (Q + R) / 2 = 6250) (h3 : P = 4000) : (P + R) / 2 = 5200 := by
  sorry

end NUMINAMATH_GPT_average_monthly_income_l1601_160150


namespace NUMINAMATH_GPT_triangle_fraction_squared_l1601_160196

theorem triangle_fraction_squared (a b c : ℝ) (h1 : b > a) 
  (h2 : a / b = (1 / 2) * (b / c)) (h3 : a + b + c = 12) 
  (h4 : c = Real.sqrt (a^2 + b^2)) : 
  (a / b)^2 = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_triangle_fraction_squared_l1601_160196


namespace NUMINAMATH_GPT_number_of_workers_who_read_all_three_books_l1601_160181

theorem number_of_workers_who_read_all_three_books
  (W S K A SK SA KA SKA N : ℝ)
  (hW : W = 75)
  (hS : S = 1 / 2 * W)
  (hK : K = 1 / 4 * W)
  (hA : A = 1 / 5 * W)
  (hSK : SK = 2 * SKA)
  (hN : N = S - (SK + SA + SKA) - 1)
  (hTotal : S + K + A - (SK + SA + KA - SKA) + N = W) :
  SKA = 6 :=
by
  -- The proof steps are omitted
  sorry

end NUMINAMATH_GPT_number_of_workers_who_read_all_three_books_l1601_160181


namespace NUMINAMATH_GPT_tangent_line_at_P_l1601_160130

/-- Define the center of the circle as the origin and point P --/
def center : ℝ × ℝ := (0, 0)

def P : ℝ × ℝ := (1, 2)

/-- Define the circle with radius squared r², where the radius passes through point P leading to r² = 5 --/
def circle_equation (x y : ℝ) : Prop := x * x + y * y = 5

/-- Define the condition that point P lies on the circle centered at the origin --/
def P_on_circle : Prop := circle_equation P.1 P.2

/-- Define what it means for a line to be the tangent at point P --/
def tangent_line (x y : ℝ) : Prop := x + 2 * y - 5 = 0

theorem tangent_line_at_P : P_on_circle → ∃ x y, tangent_line x y :=
by {
  sorry
}

end NUMINAMATH_GPT_tangent_line_at_P_l1601_160130


namespace NUMINAMATH_GPT_completing_square_solution_l1601_160195

theorem completing_square_solution (x : ℝ) :
  x^2 - 4*x - 3 = 0 ↔ (x - 2)^2 = 7 :=
sorry

end NUMINAMATH_GPT_completing_square_solution_l1601_160195


namespace NUMINAMATH_GPT_recurring_decimal_product_l1601_160176

theorem recurring_decimal_product : (0.3333333333 : ℝ) * (0.4545454545 : ℝ) = (5 / 33 : ℝ) :=
sorry

end NUMINAMATH_GPT_recurring_decimal_product_l1601_160176


namespace NUMINAMATH_GPT_profit_difference_l1601_160189

variable (P : ℕ) -- P is the total profit
variable (r1 r2 : ℚ) -- r1 and r2 are the parts of the ratio for X and Y, respectively

noncomputable def X_share (P : ℕ) (r1 r2 : ℚ) : ℚ :=
  (r1 / (r1 + r2)) * P

noncomputable def Y_share (P : ℕ) (r1 r2 : ℚ) : ℚ :=
  (r2 / (r1 + r2)) * P

theorem profit_difference (P : ℕ) (r1 r2 : ℚ) (hP : P = 800) (hr1 : r1 = 1/2) (hr2 : r2 = 1/3) :
  X_share P r1 r2 - Y_share P r1 r2 = 160 := by
  sorry

end NUMINAMATH_GPT_profit_difference_l1601_160189


namespace NUMINAMATH_GPT_geometric_seq_ratio_l1601_160190

theorem geometric_seq_ratio
  (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : ∀ n, a (n+1) = q * a n)
  (h2 : 0 < q)                    -- ensuring positivity
  (h3 : 3 * a 0 + 2 * q * a 0 = q^2 * a 0)  -- condition from problem
  : ∀ n, (a (n+3) + a (n+2)) / (a (n+1) + a n) = 9 :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_ratio_l1601_160190


namespace NUMINAMATH_GPT_find_pairs_l1601_160108

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_pairs (a b : ℕ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) : 
  (digit_sum (a^(b+1)) = a^b) ↔ 
  ((a = 1) ∨ (a = 3 ∧ b = 2) ∨ (a = 9 ∧ b = 1)) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_l1601_160108


namespace NUMINAMATH_GPT_Juan_run_time_l1601_160180

theorem Juan_run_time
  (d : ℕ) (s : ℕ) (t : ℕ)
  (H1: d = 80)
  (H2: s = 10)
  (H3: t = d / s) :
  t = 8 := 
sorry

end NUMINAMATH_GPT_Juan_run_time_l1601_160180


namespace NUMINAMATH_GPT_danny_watermelon_slices_l1601_160156

theorem danny_watermelon_slices : 
  ∀ (x : ℕ), 3 * x + 15 = 45 -> x = 10 := by
  intros x h
  sorry

end NUMINAMATH_GPT_danny_watermelon_slices_l1601_160156


namespace NUMINAMATH_GPT_divides_five_iff_l1601_160142

theorem divides_five_iff (a : ℤ) : (5 ∣ a^2) ↔ (5 ∣ a) := sorry

end NUMINAMATH_GPT_divides_five_iff_l1601_160142
