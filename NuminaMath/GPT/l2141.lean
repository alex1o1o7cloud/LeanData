import Mathlib

namespace NUMINAMATH_GPT_vasya_petya_distance_l2141_214105

theorem vasya_petya_distance :
  ∀ (D : ℝ), 
    (3 : ℝ) ≠ 0 → (6 : ℝ) ≠ 0 →
    ((D / 3) + (D / 6) = 2.5) →
    ((D / 6) + (D / 3) = 3.5) →
    D = 12 := 
by
  intros D h3 h6 h1 h2
  sorry

end NUMINAMATH_GPT_vasya_petya_distance_l2141_214105


namespace NUMINAMATH_GPT_distinct_solutions_square_difference_l2141_214147

theorem distinct_solutions_square_difference 
  (Φ φ : ℝ) (h1 : Φ^2 = Φ + 2) (h2 : φ^2 = φ + 2) (h_distinct : Φ ≠ φ) :
  (Φ - φ)^2 = 9 :=
  sorry

end NUMINAMATH_GPT_distinct_solutions_square_difference_l2141_214147


namespace NUMINAMATH_GPT_point_M_coordinates_l2141_214186

theorem point_M_coordinates :
  ∃ M : ℝ × ℝ × ℝ, 
    M.1 = 0 ∧ M.2.1 = 0 ∧  
    (dist (1, 0, 2) (M.1, M.2.1, M.2.2) = dist (1, -3, 1) (M.1, M.2.1, M.2.2)) ∧ 
    M = (0, 0, -3) :=
by
  sorry

end NUMINAMATH_GPT_point_M_coordinates_l2141_214186


namespace NUMINAMATH_GPT_p3_mp_odd_iff_m_even_l2141_214108

theorem p3_mp_odd_iff_m_even (p m : ℕ) (hp : p % 2 = 1) : (p^3 + m * p) % 2 = 1 ↔ m % 2 = 0 := sorry

end NUMINAMATH_GPT_p3_mp_odd_iff_m_even_l2141_214108


namespace NUMINAMATH_GPT_number_equation_l2141_214126

variable (x : ℝ)

theorem number_equation :
  5 * x - 2 * x = 10 :=
sorry

end NUMINAMATH_GPT_number_equation_l2141_214126


namespace NUMINAMATH_GPT_sphere_volume_diameter_l2141_214140

theorem sphere_volume_diameter {D : ℝ} : 
  (D^3/2 + (1/21) * (D^3/2)) = (π * D^3 / 6) ↔ π = 22 / 7 := 
sorry

end NUMINAMATH_GPT_sphere_volume_diameter_l2141_214140


namespace NUMINAMATH_GPT_relationship_of_y_values_l2141_214119

theorem relationship_of_y_values (k : ℝ) (y₁ y₂ y₃ : ℝ) :
  (y₁ = (k^2 + 3) / (-3)) ∧ (y₂ = (k^2 + 3) / (-1)) ∧ (y₃ = (k^2 + 3) / 2) →
  y₂ < y₁ ∧ y₁ < y₃ :=
by
  intro h
  have h₁ : y₁ = (k^2 + 3) / (-3) := h.1
  have h₂ : y₂ = (k^2 + 3) / (-1) := h.2.1
  have h₃ : y₃ = (k^2 + 3) / 2 := h.2.2
  sorry

end NUMINAMATH_GPT_relationship_of_y_values_l2141_214119


namespace NUMINAMATH_GPT_most_reasonable_sampling_method_l2141_214188

-- Definitions for the conditions
def significant_difference_by_stage : Prop := 
  -- There is a significant difference in vision condition at different educational stages
  sorry

def no_significant_difference_by_gender : Prop :=
  -- There is no significant difference in vision condition between male and female students
  sorry

-- Theorem statement
theorem most_reasonable_sampling_method 
  (h1 : significant_difference_by_stage) 
  (h2 : no_significant_difference_by_gender) : 
  -- The most reasonable sampling method is stratified sampling by educational stage
  sorry :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_most_reasonable_sampling_method_l2141_214188


namespace NUMINAMATH_GPT_find_number_l2141_214115

theorem find_number (x n : ℤ) 
  (h1 : 0 < x) (h2 : x < 7) 
  (h3 : x < 15) 
  (h4 : -1 < x) (h5 : x < 5) 
  (h6 : x < 3) (h7 : 0 < x) 
  (h8 : x + n < 4) 
  (hx : x = 1): 
  n < 3 := 
sorry

end NUMINAMATH_GPT_find_number_l2141_214115


namespace NUMINAMATH_GPT_father_children_problem_l2141_214163

theorem father_children_problem {F C n : ℕ} 
  (hF_C : F = C) 
  (sum_ages_after_15_years : C + 15 * n = 2 * (F + 15)) 
  (father_age : F = 75) : 
  n = 7 :=
by
  sorry

end NUMINAMATH_GPT_father_children_problem_l2141_214163


namespace NUMINAMATH_GPT_initial_red_balloons_l2141_214190

variable (initial_red : ℕ)
variable (given_away : ℕ := 24)
variable (left_with : ℕ := 7)

theorem initial_red_balloons : initial_red = given_away + left_with :=
by sorry

end NUMINAMATH_GPT_initial_red_balloons_l2141_214190


namespace NUMINAMATH_GPT_debby_drink_days_l2141_214113

theorem debby_drink_days :
  ∀ (total_bottles : ℕ) (bottles_per_day : ℕ) (remaining_bottles : ℕ),
  total_bottles = 301 →
  bottles_per_day = 144 →
  remaining_bottles = 157 →
  (total_bottles - remaining_bottles) / bottles_per_day = 1 :=
by
  intros total_bottles bottles_per_day remaining_bottles ht he hb
  sorry

end NUMINAMATH_GPT_debby_drink_days_l2141_214113


namespace NUMINAMATH_GPT_triangle_pyramid_angle_l2141_214101

theorem triangle_pyramid_angle (φ : ℝ) (vertex_angle : ∀ (A B C : ℝ), (A + B + C = φ)) :
  ∃ θ : ℝ, θ = φ :=
by
  sorry

end NUMINAMATH_GPT_triangle_pyramid_angle_l2141_214101


namespace NUMINAMATH_GPT_chef_made_10_cakes_l2141_214173

-- Definitions based on the conditions
def total_eggs : ℕ := 60
def eggs_in_fridge : ℕ := 10
def eggs_per_cake : ℕ := 5

-- Calculated values based on the definitions
def eggs_for_cakes : ℕ := total_eggs - eggs_in_fridge
def number_of_cakes : ℕ := eggs_for_cakes / eggs_per_cake

-- Theorem to prove
theorem chef_made_10_cakes : number_of_cakes = 10 := by
  sorry

end NUMINAMATH_GPT_chef_made_10_cakes_l2141_214173


namespace NUMINAMATH_GPT_schedule_problem_l2141_214159

def num_schedule_ways : Nat :=
  -- total ways to pick 3 out of 6 periods and arrange 3 courses
  let total_ways := Nat.choose 6 3 * Nat.factorial 3
  -- at least two consecutive courses (using Principle of Inclusion and Exclusion)
  let two_consecutive := 5 * 6 * 4
  let three_consecutive := 4 * 6
  let invalid_ways := two_consecutive + three_consecutive
  total_ways - invalid_ways

theorem schedule_problem (h : num_schedule_ways = 24) : num_schedule_ways = 24 := by {
  exact h
}

end NUMINAMATH_GPT_schedule_problem_l2141_214159


namespace NUMINAMATH_GPT_ramu_selling_price_l2141_214177

theorem ramu_selling_price (P R : ℝ) (profit_percent : ℝ) 
  (P_def : P = 42000)
  (R_def : R = 13000)
  (profit_percent_def : profit_percent = 17.272727272727273) :
  let total_cost := P + R
  let selling_price := total_cost * (1 + (profit_percent / 100))
  selling_price = 64500 := 
by
  sorry

end NUMINAMATH_GPT_ramu_selling_price_l2141_214177


namespace NUMINAMATH_GPT_boat_speed_still_water_l2141_214155

-- Define the conditions
def speed_of_stream : ℝ := 4
def distance_downstream : ℕ := 68
def time_downstream : ℕ := 4

-- State the theorem
theorem boat_speed_still_water : 
  ∃V_b : ℝ, distance_downstream = (V_b + speed_of_stream) * time_downstream ∧ V_b = 13 :=
by 
  sorry

end NUMINAMATH_GPT_boat_speed_still_water_l2141_214155


namespace NUMINAMATH_GPT_n_and_m_integers_and_n2_plus_m3_odd_then_n_plus_m_odd_l2141_214154

theorem n_and_m_integers_and_n2_plus_m3_odd_then_n_plus_m_odd :
  ∀ (n m : ℤ), (n^2 + m^3) % 2 ≠ 0 → (n + m) % 2 = 1 :=
by sorry

end NUMINAMATH_GPT_n_and_m_integers_and_n2_plus_m3_odd_then_n_plus_m_odd_l2141_214154


namespace NUMINAMATH_GPT_find_a_l2141_214131

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x < 0 then -(Real.log (-x) / Real.log 2) + a else 0

theorem find_a (a : ℝ) :
  (f a (-2) + f a (-4) = 1) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2141_214131


namespace NUMINAMATH_GPT_find_first_number_l2141_214181

theorem find_first_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = (x + 60 + 35) / 3 + 5 → 
  x = 10 := 
by 
  sorry

end NUMINAMATH_GPT_find_first_number_l2141_214181


namespace NUMINAMATH_GPT_polynomial_possible_integer_roots_l2141_214152

theorem polynomial_possible_integer_roots (b1 b2 : ℤ) :
  ∀ x : ℤ, (x ∣ 18) ↔ (x^3 + b2 * x^2 + b1 * x + 18 = 0) → 
  x = -18 ∨ x = -9 ∨ x = -6 ∨ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 6 ∨ x = 9 ∨ x = 18 :=
by {
  sorry
}


end NUMINAMATH_GPT_polynomial_possible_integer_roots_l2141_214152


namespace NUMINAMATH_GPT_find_constant_l2141_214150

noncomputable def f (x : ℝ) : ℝ := x + 4

theorem find_constant : ∃ c : ℝ, (∀ x : ℝ, x = 0.4 → (3 * f (x - c)) / f 0 + 4 = f (2 * x + 1)) ∧ c = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_constant_l2141_214150


namespace NUMINAMATH_GPT_toy_store_fraction_l2141_214167

theorem toy_store_fraction
  (allowance : ℝ) (arcade_fraction : ℝ) (candy_store_amount : ℝ)
  (h1 : allowance = 1.50)
  (h2 : arcade_fraction = 3 / 5)
  (h3 : candy_store_amount = 0.40) :
  (0.60 - candy_store_amount) / (allowance - arcade_fraction * allowance) = 1 / 3 :=
by
  -- We're skipping the actual proof steps
  sorry

end NUMINAMATH_GPT_toy_store_fraction_l2141_214167


namespace NUMINAMATH_GPT_find_exponent_l2141_214103

theorem find_exponent 
  (h1 : (1 : ℝ) / 9 = 3 ^ (-2 : ℝ))
  (h2 : (3 ^ (20 : ℝ) : ℝ) / 9 = 3 ^ x) : 
  x = 18 :=
by sorry

end NUMINAMATH_GPT_find_exponent_l2141_214103


namespace NUMINAMATH_GPT_initial_chocolate_bars_l2141_214132

theorem initial_chocolate_bars (B : ℕ) 
  (H1 : Thomas_and_friends_take = B / 4)
  (H2 : One_friend_returns_5 = Thomas_and_friends_take - 5)
  (H3 : Piper_takes = Thomas_and_friends_take - 5 - 5)
  (H4 : Remaining_bars = B - Thomas_and_friends_take - Piper_takes)
  (H5 : Remaining_bars = 110) :
  B = 190 := 
sorry

end NUMINAMATH_GPT_initial_chocolate_bars_l2141_214132


namespace NUMINAMATH_GPT_inequality_always_true_l2141_214118

theorem inequality_always_true (x : ℝ) : (4 * x) / (x ^ 2 + 4) ≤ 1 := by
  sorry

end NUMINAMATH_GPT_inequality_always_true_l2141_214118


namespace NUMINAMATH_GPT_find_y_l2141_214165

-- Definitions of vectors and parallel relationship
def vector_a : ℝ × ℝ := (4, 2)
def vector_b (y : ℝ) : ℝ × ℝ := (6, y)
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

-- The theorem we want to prove
theorem find_y (y : ℝ) (h : parallel vector_a (vector_b y)) : y = 3 :=
sorry

end NUMINAMATH_GPT_find_y_l2141_214165


namespace NUMINAMATH_GPT_gcd_45_75_eq_15_l2141_214185

theorem gcd_45_75_eq_15 : Nat.gcd 45 75 = 15 := by
  sorry

end NUMINAMATH_GPT_gcd_45_75_eq_15_l2141_214185


namespace NUMINAMATH_GPT_percentage_increase_l2141_214194

theorem percentage_increase (P Q R : ℝ) (x y : ℝ) 
  (h1 : P > 0) (h2 : Q > 0) (h3 : R > 0)
  (h4 : P = (1 + x / 100) * Q)
  (h5 : Q = (1 + y / 100) * R)
  (h6 : P = 2.4 * R) :
  x + y = 140 :=
sorry

end NUMINAMATH_GPT_percentage_increase_l2141_214194


namespace NUMINAMATH_GPT_grocery_store_more_expensive_per_can_l2141_214161

theorem grocery_store_more_expensive_per_can :
  ∀ (bulk_case_price : ℝ) (bulk_cans_per_case : ℕ)
    (grocery_case_price : ℝ) (grocery_cans_per_case : ℕ),
  bulk_case_price = 12.00 →
  bulk_cans_per_case = 48 →
  grocery_case_price = 6.00 →
  grocery_cans_per_case = 12 →
  (grocery_case_price / grocery_cans_per_case - bulk_case_price / bulk_cans_per_case) * 100 = 25 :=
by
  intros _ _ _ _ h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_grocery_store_more_expensive_per_can_l2141_214161


namespace NUMINAMATH_GPT_problem_l2141_214130

theorem problem (a b : ℝ) (h : a > b) : a / 3 > b / 3 :=
sorry

end NUMINAMATH_GPT_problem_l2141_214130


namespace NUMINAMATH_GPT_common_chord_equation_l2141_214162

def circle1 (x y : ℝ) := x^2 + y^2 + 2*x + 2*y - 8 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 2*x + 10*y - 24 = 0

theorem common_chord_equation :
  ∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle2 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 B.1 B.2 ∧
                     ∀ (x y : ℝ), (x - 2*y + 4 = 0) ↔ ((x, y) = A ∨ (x, y) = B) :=
by
  sorry

end NUMINAMATH_GPT_common_chord_equation_l2141_214162


namespace NUMINAMATH_GPT_binary_sum_in_base_10_l2141_214100

theorem binary_sum_in_base_10 :
  (255 : ℕ) + (63 : ℕ) = 318 :=
sorry

end NUMINAMATH_GPT_binary_sum_in_base_10_l2141_214100


namespace NUMINAMATH_GPT_repeating_decimals_sum_l2141_214135

-- Define the repeating decimals as rational numbers
def dec_0_3 : ℚ := 1 / 3
def dec_0_02 : ℚ := 2 / 99
def dec_0_0004 : ℚ := 4 / 9999

-- State the theorem that we need to prove
theorem repeating_decimals_sum :
  dec_0_3 + dec_0_02 + dec_0_0004 = 10581 / 29889 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimals_sum_l2141_214135


namespace NUMINAMATH_GPT_new_ratio_books_clothes_l2141_214125

theorem new_ratio_books_clothes :
  ∀ (B C E : ℝ), (B = 22.5) → (C = 18) → (E = 9) → (C_new = C - 9) → C_new = 9 → B / C_new = 2.5 :=
by
  intros B C E HB HC HE HCnew Hnew
  sorry

end NUMINAMATH_GPT_new_ratio_books_clothes_l2141_214125


namespace NUMINAMATH_GPT_largest_of_four_numbers_l2141_214124

theorem largest_of_four_numbers (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  max (max (max (a^2 + b^2) (2 * a * b)) a) (1 / 2) = a^2 + b^2 :=
by
  sorry

end NUMINAMATH_GPT_largest_of_four_numbers_l2141_214124


namespace NUMINAMATH_GPT_intersection_point_l2141_214198

theorem intersection_point (x y : ℝ) (h1 : x - 2 * y = 0) (h2 : x + y - 3 = 0) : x = 2 ∧ y = 1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_l2141_214198


namespace NUMINAMATH_GPT_solve_for_x_l2141_214138

def operation (a b : ℝ) : ℝ := a^2 - 3*a + b

theorem solve_for_x (x : ℝ) : operation x 2 = 6 → (x = -1 ∨ x = 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2141_214138


namespace NUMINAMATH_GPT_gcd_of_three_numbers_l2141_214180

theorem gcd_of_three_numbers (a b c d : ℕ) (ha : a = 72) (hb : b = 120) (hc : c = 168) (hd : d = 24) : 
  Nat.gcd (Nat.gcd a b) c = d :=
by
  rw [ha, hb, hc, hd]
  -- Placeholder for the actual proof
  exact sorry

end NUMINAMATH_GPT_gcd_of_three_numbers_l2141_214180


namespace NUMINAMATH_GPT_least_multiple_of_24_gt_500_l2141_214146

theorem least_multiple_of_24_gt_500 : ∃ x : ℕ, (x % 24 = 0) ∧ (x > 500) ∧ (∀ y : ℕ, (y % 24 = 0) ∧ (y > 500) → y ≥ x) ∧ (x = 504) := by
  sorry

end NUMINAMATH_GPT_least_multiple_of_24_gt_500_l2141_214146


namespace NUMINAMATH_GPT_proof_f_f_2008_eq_2008_l2141_214160

-- Define the function f
axiom f : ℝ → ℝ

-- The conditions given in the problem
axiom odd_f : ∀ x, f (-x) = -f x
axiom periodic_f : ∀ x, f (x + 6) = f x
axiom f_at_4 : f 4 = -2008

-- The goal to prove
theorem proof_f_f_2008_eq_2008 : f (f 2008) = 2008 :=
by
  sorry

end NUMINAMATH_GPT_proof_f_f_2008_eq_2008_l2141_214160


namespace NUMINAMATH_GPT_mutually_exclusive_events_l2141_214111

-- Definitions based on the given conditions
def sample_inspection (n : ℕ) := n = 10
def event_A (defective_products : ℕ) := defective_products ≥ 2
def event_B (defective_products : ℕ) := defective_products ≤ 1

-- The proof statement
theorem mutually_exclusive_events (n : ℕ) (defective_products : ℕ) 
  (h1 : sample_inspection n) (h2 : event_A defective_products) : 
  event_B defective_products = false :=
by
  sorry

end NUMINAMATH_GPT_mutually_exclusive_events_l2141_214111


namespace NUMINAMATH_GPT_cloth_sold_l2141_214156

theorem cloth_sold (C S P: ℝ) (N : ℕ) 
  (h1 : S = 3 * C)
  (h2 : P = 10 * S)
  (h3 : (200 : ℝ) = (P / (N * C)) * 100) : N = 15 := 
sorry

end NUMINAMATH_GPT_cloth_sold_l2141_214156


namespace NUMINAMATH_GPT_find_f_of_1_over_2016_l2141_214137

noncomputable def f (x : ℝ) : ℝ := sorry

lemma f_property_0 : f 0 = 0 := sorry
lemma f_property_1 (x : ℝ) : f x + f (1 - x) = 1 := sorry
lemma f_property_2 (x : ℝ) : f (x / 3) = (1 / 2) * f x := sorry
lemma f_property_3 {x₁ x₂ : ℝ} (h₀ : 0 ≤ x₁) (h₁ : x₁ < x₂) (h₂ : x₂ ≤ 1): f x₁ ≤ f x₂ := sorry

theorem find_f_of_1_over_2016 : f (1 / 2016) = 1 / 128 := sorry

end NUMINAMATH_GPT_find_f_of_1_over_2016_l2141_214137


namespace NUMINAMATH_GPT_determine_points_on_line_l2141_214120

def pointA : ℝ × ℝ := (2, 5)
def pointB : ℝ × ℝ := (1, 2.2)
def line_eq (x y : ℝ) : ℝ := 3 * x - 5 * y + 8

theorem determine_points_on_line :
  (line_eq pointA.1 pointA.2 ≠ 0) ∧ (line_eq pointB.1 pointB.2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_determine_points_on_line_l2141_214120


namespace NUMINAMATH_GPT_exists_consecutive_integers_sum_cube_l2141_214191

theorem exists_consecutive_integers_sum_cube :
  ∃ (n : ℤ), ∃ (k : ℤ), 1981 * (n + 990) = k^3 :=
by
  sorry

end NUMINAMATH_GPT_exists_consecutive_integers_sum_cube_l2141_214191


namespace NUMINAMATH_GPT_sequence_general_term_l2141_214158

-- Define the sequence using a recurrence relation for clarity in formal proof
def a (n : ℕ) : ℕ :=
  if h : n > 0 then 2^n + 1 else 3

theorem sequence_general_term :
  ∀ n : ℕ, n > 0 → a n = 2^n + 1 := 
by 
  sorry

end NUMINAMATH_GPT_sequence_general_term_l2141_214158


namespace NUMINAMATH_GPT_value_to_subtract_l2141_214121

variable (x y : ℝ)

theorem value_to_subtract (h1 : (x - 5) / 7 = 7) (h2 : (x - y) / 8 = 6) : y = 6 := by
  sorry

end NUMINAMATH_GPT_value_to_subtract_l2141_214121


namespace NUMINAMATH_GPT_contradiction_method_l2141_214106

variable (a b : ℝ)

theorem contradiction_method (h1 : a > b) (h2 : 3 * a ≤ 3 * b) : false :=
by sorry

end NUMINAMATH_GPT_contradiction_method_l2141_214106


namespace NUMINAMATH_GPT_milkman_A_rent_share_l2141_214178

theorem milkman_A_rent_share : 
  let A_cows := 24
  let A_months := 3
  let B_cows := 10
  let B_months := 5
  let C_cows := 35
  let C_months := 4
  let D_cows := 21
  let D_months := 3
  let total_rent := 3250
  let A_cow_months := A_cows * A_months
  let B_cow_months := B_cows * B_months
  let C_cow_months := C_cows * C_months
  let D_cow_months := D_cows * D_months
  let total_cow_months := A_cow_months + B_cow_months + C_cow_months + D_cow_months
  let fraction_A := A_cow_months / total_cow_months
  let A_rent_share := total_rent * fraction_A
  A_rent_share = 720 := 
by
  sorry

end NUMINAMATH_GPT_milkman_A_rent_share_l2141_214178


namespace NUMINAMATH_GPT_number_of_perfect_square_factors_450_l2141_214197

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def prime_factorization_450 := (2, 1) :: (3, 2) :: (5, 2) :: []

def perfect_square_factors (n : ℕ) : ℕ :=
  if n = 450 then 4 else 0

theorem number_of_perfect_square_factors_450 : perfect_square_factors 450 = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_perfect_square_factors_450_l2141_214197


namespace NUMINAMATH_GPT_sufficient_condition_inequality_l2141_214128

theorem sufficient_condition_inequality (k : ℝ) :
  (k = 0 ∨ (-3 < k ∧ k < 0)) → ∀ x : ℝ, 2 * k * x^2 + k * x - 3 / 8 < 0 :=
sorry

end NUMINAMATH_GPT_sufficient_condition_inequality_l2141_214128


namespace NUMINAMATH_GPT_integer_solutions_eq_0_or_2_l2141_214102

theorem integer_solutions_eq_0_or_2 (a : ℤ) (x : ℤ) : 
  (a * x^2 + 6 = 0) → (a = -6 ∧ (x = 1 ∨ x = -1)) ∨ (¬ (a = -6) ∧ (x ≠ 1) ∧ (x ≠ -1)) :=
by 
sorry

end NUMINAMATH_GPT_integer_solutions_eq_0_or_2_l2141_214102


namespace NUMINAMATH_GPT_range_of_a_l2141_214141

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l2141_214141


namespace NUMINAMATH_GPT_initial_stickers_correct_l2141_214136

-- Definitions based on the conditions
def initial_stickers (X : ℕ) : ℕ := X
def after_buying (X : ℕ) : ℕ := X + 26
def after_birthday (X : ℕ) : ℕ := after_buying X + 20
def after_giving (X : ℕ) : ℕ := after_birthday X - 6
def after_decorating (X : ℕ) : ℕ := after_giving X - 58

-- Theorem stating the problem and the expected answer
theorem initial_stickers_correct (X : ℕ) (h : after_decorating X = 2) : initial_stickers X = 26 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_stickers_correct_l2141_214136


namespace NUMINAMATH_GPT_arithmetic_sequence_100th_term_l2141_214171

theorem arithmetic_sequence_100th_term (a b : ℤ)
  (h1 : 2 * a - a = a) -- definition of common difference d where d = a
  (h2 : b - 2 * a = a) -- b = 3a
  (h3 : a - 6 - b = -2 * a - 6) -- consistency of fourth term
  (h4 : 6 * a = -6) -- equation to solve for a
  : (a + 99 * (2 * a - a)) = -100 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_100th_term_l2141_214171


namespace NUMINAMATH_GPT_removed_cubes_total_l2141_214148

-- Define the large cube composed of 125 smaller cubes (5x5x5 cube)
def large_cube := 5 * 5 * 5

-- Number of smaller cubes removed from each face to opposite face
def removed_faces := (5 * 5 + 5 * 5 + 5 * 3)

-- Overlapping cubes deducted
def overlapping_cubes := (3 + 1)

-- Final number of removed smaller cubes
def removed_total := removed_faces - overlapping_cubes

-- Lean theorem statement
theorem removed_cubes_total : removed_total = 49 :=
by
  -- Definitions provided above imply the theorem
  sorry

end NUMINAMATH_GPT_removed_cubes_total_l2141_214148


namespace NUMINAMATH_GPT_carol_to_cathy_ratio_l2141_214139

-- Define the number of cars owned by Cathy, Lindsey, Carol, and Susan
def cathy_cars : ℕ := 5
def lindsey_cars : ℕ := cathy_cars + 4
def carol_cars : ℕ := cathy_cars
def susan_cars : ℕ := carol_cars - 2

-- Define the total number of cars in the problem statement
def total_cars : ℕ := 32

-- Theorem to prove the ratio of Carol's cars to Cathy's cars is 1:1
theorem carol_to_cathy_ratio : carol_cars = cathy_cars := by
  sorry

end NUMINAMATH_GPT_carol_to_cathy_ratio_l2141_214139


namespace NUMINAMATH_GPT_average_velocity_first_second_instantaneous_velocity_end_first_second_velocity_reaches_14_after_2_seconds_l2141_214183

open Real

noncomputable def f (x : ℝ) := (2/3) * x ^ 3 + x ^ 2 + 2 * x

-- (1) Prove that the average velocity of the particle during the first second is 3 m/s
theorem average_velocity_first_second : (f 1 - f 0) / (1 - 0) = 3 := by
  sorry

-- (2) Prove that the instantaneous velocity at the end of the first second is 6 m/s
theorem instantaneous_velocity_end_first_second : deriv f 1 = 6 := by
  sorry

-- (3) Prove that the velocity of the particle reaches 14 m/s after 2 seconds
theorem velocity_reaches_14_after_2_seconds :
  ∃ x : ℝ, deriv f x = 14 ∧ x = 2 := by
  sorry

end NUMINAMATH_GPT_average_velocity_first_second_instantaneous_velocity_end_first_second_velocity_reaches_14_after_2_seconds_l2141_214183


namespace NUMINAMATH_GPT_goldfish_equal_months_l2141_214144

theorem goldfish_equal_months :
  ∃ (n : ℕ), 
    let B_n := 3 * 3^n 
    let G_n := 125 * 5^n 
    B_n = G_n ∧ n = 5 :=
by
  sorry

end NUMINAMATH_GPT_goldfish_equal_months_l2141_214144


namespace NUMINAMATH_GPT_factor_x4_plus_16_l2141_214112

theorem factor_x4_plus_16 (x : ℝ) : x^4 + 16 = (x^2 + 2*x + 2) * (x^2 - 2*x + 2) := by
  sorry

end NUMINAMATH_GPT_factor_x4_plus_16_l2141_214112


namespace NUMINAMATH_GPT_arrangement_count_BANANA_l2141_214142

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end NUMINAMATH_GPT_arrangement_count_BANANA_l2141_214142


namespace NUMINAMATH_GPT_smallest_n_l2141_214189

-- Definitions for arithmetic sequences with given conditions
def arithmetic_sequence_a (n : ℕ) (x : ℕ) : ℕ := 1 + (n-1) * x
def arithmetic_sequence_b (n : ℕ) (y : ℕ) : ℕ := 1 + (n-1) * y

-- Main theorem statement
theorem smallest_n (x y n : ℕ) (hxy : x < y) (ha1 : arithmetic_sequence_a 1 x = 1) (hb1 : arithmetic_sequence_b 1 y = 1) 
  (h_sum : arithmetic_sequence_a n x + arithmetic_sequence_b n y = 2556) : n = 3 :=
sorry

end NUMINAMATH_GPT_smallest_n_l2141_214189


namespace NUMINAMATH_GPT_verify_segment_lengths_l2141_214122

noncomputable def segment_lengths_proof : Prop :=
  let a := 2
  let b := 3
  let alpha := Real.arccos (5 / 16)
  let segment1 := 4 / 3
  let segment2 := 2 / 3
  let segment3 := 2
  let segment4 := 1
  ∀ (s1 s2 s3 s4 : ℝ), 
    (s1 = segment1 ∧ s2 = segment2 ∧ s3 = segment3 ∧ s4 = segment4) ↔
    -- Parallelogram sides and angle constraints
    (s1 + s2 = a ∧ s3 + s4 = b ∧ 
     -- Mutually perpendicular lines divide into equal areas
     (s1 * s3 * Real.sin alpha / 2 = s2 * s4 * Real.sin alpha / 2) )

-- Placeholder for proof
theorem verify_segment_lengths : segment_lengths_proof :=
  sorry

end NUMINAMATH_GPT_verify_segment_lengths_l2141_214122


namespace NUMINAMATH_GPT_find_value_of_z_l2141_214195

theorem find_value_of_z (z : ℂ) (h1 : ∀ a : ℝ, z = a * I) (h2 : ((z + 2) / (1 - I)).im = 0) : z = -2 * I :=
sorry

end NUMINAMATH_GPT_find_value_of_z_l2141_214195


namespace NUMINAMATH_GPT_sqrt_range_real_l2141_214104

theorem sqrt_range_real (x : ℝ) (h : 1 - 3 * x ≥ 0) : x ≤ 1 / 3 :=
sorry

end NUMINAMATH_GPT_sqrt_range_real_l2141_214104


namespace NUMINAMATH_GPT_primary_schools_to_be_selected_l2141_214164

noncomputable def total_schools : ℕ := 150 + 75 + 25
noncomputable def proportion_primary : ℚ := 150 / total_schools
noncomputable def selected_primary : ℚ := proportion_primary * 30

theorem primary_schools_to_be_selected : selected_primary = 18 :=
by sorry

end NUMINAMATH_GPT_primary_schools_to_be_selected_l2141_214164


namespace NUMINAMATH_GPT_sum_of_first_2m_terms_l2141_214134

variable (m : ℕ)
variable (S : ℕ → ℤ)

-- Conditions
axiom Sm : S m = 100
axiom S3m : S (3 * m) = -150

-- Theorem statement
theorem sum_of_first_2m_terms : S (2 * m) = 50 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_2m_terms_l2141_214134


namespace NUMINAMATH_GPT_quartic_poly_roots_l2141_214133

noncomputable def roots_polynomial : List ℝ := [
  (1 + Real.sqrt 5) / 2,
  (1 - Real.sqrt 5) / 2,
  (3 + Real.sqrt 13) / 6,
  (3 - Real.sqrt 13) / 6
]

theorem quartic_poly_roots :
  ∀ x : ℝ, x ∈ roots_polynomial ↔ 3*x^4 - 4*x^3 - 5*x^2 - 4*x + 3 = 0 :=
by sorry

end NUMINAMATH_GPT_quartic_poly_roots_l2141_214133


namespace NUMINAMATH_GPT_handshake_count_l2141_214151

theorem handshake_count
  (total_people : ℕ := 40)
  (groupA_size : ℕ := 30)
  (groupB_size : ℕ := 10)
  (groupB_knowsA_5 : ℕ := 3)
  (groupB_knowsA_0 : ℕ := 7)
  (handshakes_between_A_and_B5 : ℕ := groupB_knowsA_5 * (groupA_size - 5))
  (handshakes_between_A_and_B0 : ℕ := groupB_knowsA_0 * groupA_size)
  (handshakes_within_B : ℕ := groupB_size * (groupB_size - 1) / 2) :
  handshakes_between_A_and_B5 + handshakes_between_A_and_B0 + handshakes_within_B = 330 :=
sorry

end NUMINAMATH_GPT_handshake_count_l2141_214151


namespace NUMINAMATH_GPT_find_f_8_l2141_214123

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom periodicity : ∀ x : ℝ, f (x + 6) = f x
axiom function_on_interval : ∀ x : ℝ, -3 < x ∧ x < 0 → f x = 2 * x - 5

theorem find_f_8 : f 8 = -9 :=
by
  sorry

end NUMINAMATH_GPT_find_f_8_l2141_214123


namespace NUMINAMATH_GPT_f_no_zeros_in_interval_f_zeros_in_interval_l2141_214169

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x - Real.log x

theorem f_no_zeros_in_interval (x : ℝ) (hx1 : x > 1 / Real.exp 1) (hx2 : x < 1) :
  f x ≠ 0 := sorry

theorem f_zeros_in_interval (h1 : 1 < e) (x_exists : ∃ x, 1 < x ∧ x < Real.exp 1 ∧ f x = 0) :
  true := sorry

end NUMINAMATH_GPT_f_no_zeros_in_interval_f_zeros_in_interval_l2141_214169


namespace NUMINAMATH_GPT_sandy_marks_per_correct_sum_l2141_214170

theorem sandy_marks_per_correct_sum 
  (total_sums : ℕ)
  (total_marks : ℤ)
  (correct_sums : ℕ)
  (marks_per_incorrect_sum : ℤ)
  (marks_obtained : ℤ) 
  (marks_per_correct_sum : ℕ) :
  total_sums = 30 →
  total_marks = 45 →
  correct_sums = 21 →
  marks_per_incorrect_sum = 2 →
  marks_obtained = total_marks →
  marks_obtained = marks_per_correct_sum * correct_sums - marks_per_incorrect_sum * (total_sums - correct_sums) → 
  marks_per_correct_sum = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_sandy_marks_per_correct_sum_l2141_214170


namespace NUMINAMATH_GPT_number_of_friends_is_five_l2141_214157

def total_cards : ℕ := 455
def cards_per_friend : ℕ := 91

theorem number_of_friends_is_five (n : ℕ) (h : total_cards = n * cards_per_friend) : n = 5 := 
sorry

end NUMINAMATH_GPT_number_of_friends_is_five_l2141_214157


namespace NUMINAMATH_GPT_ratio_of_boys_to_girls_l2141_214176

theorem ratio_of_boys_to_girls (G B : ℕ) (hg : G = 30) (hb : B = G + 18) : B / G = 8 / 5 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_boys_to_girls_l2141_214176


namespace NUMINAMATH_GPT_sum_of_special_multiples_l2141_214153

def smallest_two_digit_multiple_of_5 : ℕ := 10
def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_special_multiples :
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_special_multiples_l2141_214153


namespace NUMINAMATH_GPT_bottles_left_l2141_214168

theorem bottles_left (total_bottles : ℕ) (bottles_per_day : ℕ) (days : ℕ)
  (h_total : total_bottles = 264)
  (h_bottles_per_day : bottles_per_day = 15)
  (h_days : days = 11) :
  total_bottles - bottles_per_day * days = 99 :=
by
  sorry

end NUMINAMATH_GPT_bottles_left_l2141_214168


namespace NUMINAMATH_GPT_extra_pieces_correct_l2141_214145

def pieces_per_package : ℕ := 7
def number_of_packages : ℕ := 5
def total_pieces : ℕ := 41

theorem extra_pieces_correct : total_pieces - (number_of_packages * pieces_per_package) = 6 :=
by
  sorry

end NUMINAMATH_GPT_extra_pieces_correct_l2141_214145


namespace NUMINAMATH_GPT_roots_inverse_cubed_l2141_214107

-- Define the conditions and the problem statement
theorem roots_inverse_cubed (p q m r s : ℝ) (h1 : r + s = -q / p) (h2 : r * s = m / p) 
  (h3 : ∀ x : ℝ, p * x^2 + q * x + m = 0 → x = r ∨ x = s) : 
  1 / r^3 + 1 / s^3 = (-q^3 + 3 * q * m) / m^3 := 
sorry

end NUMINAMATH_GPT_roots_inverse_cubed_l2141_214107


namespace NUMINAMATH_GPT_value_of_fraction_l2141_214172

theorem value_of_fraction : (20 + 15) / (30 - 25) = 7 := by
  sorry

end NUMINAMATH_GPT_value_of_fraction_l2141_214172


namespace NUMINAMATH_GPT_angle_relationship_l2141_214196

theorem angle_relationship (u x y z w : ℝ)
    (H1 : ∀ (D E : ℝ), x + y + (360 - u - z) = 360)
    (H2 : ∀ (D E : ℝ), z + w + (360 - w - x) = 360) :
    x = (u + 2*z - y - w) / 2 := by
  sorry

end NUMINAMATH_GPT_angle_relationship_l2141_214196


namespace NUMINAMATH_GPT_binomial_probability_X_eq_3_l2141_214116

theorem binomial_probability_X_eq_3 :
  let n := 6
  let p := 1 / 2
  let k := 3
  let binom := Nat.choose n k
  (binom * p ^ k * (1 - p) ^ (n - k)) = 5 / 16 := by 
  sorry

end NUMINAMATH_GPT_binomial_probability_X_eq_3_l2141_214116


namespace NUMINAMATH_GPT_problem_1_problem_2_l2141_214182

-- Define sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2 * x - 8 = 0}

-- First problem statement
theorem problem_1 (a : ℝ) : (A a ∩ B = A a ∪ B) → a = 5 :=
by
  -- proof omitted
  sorry

-- Second problem statement
theorem problem_2 (a : ℝ) : (∅ ⊆ A a ∩ B) ∧ (A a ∩ C = ∅) → a = -2 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2141_214182


namespace NUMINAMATH_GPT_goodColoringsOfPoints_l2141_214179

noncomputable def countGoodColorings (k m : ℕ) : ℕ :=
  (k * (k - 1) + 2) * 2 ^ m

theorem goodColoringsOfPoints :
  countGoodColorings 2011 2011 = (2011 * 2010 + 2) * 2 ^ 2011 :=
  by
    sorry

end NUMINAMATH_GPT_goodColoringsOfPoints_l2141_214179


namespace NUMINAMATH_GPT_solve_equation_l2141_214175

theorem solve_equation (x : ℚ) :
  (x + 10) / (x - 4) = (x - 3) / (x + 6) ↔ x = -48 / 23 :=
by sorry

end NUMINAMATH_GPT_solve_equation_l2141_214175


namespace NUMINAMATH_GPT_difference_of_extremes_l2141_214184

def digits : List ℕ := [2, 0, 1, 3]

def largest_integer : ℕ := 3210
def smallest_integer_greater_than_1000 : ℕ := 1023
def expected_difference : ℕ := 2187

theorem difference_of_extremes :
  largest_integer - smallest_integer_greater_than_1000 = expected_difference := by
  sorry

end NUMINAMATH_GPT_difference_of_extremes_l2141_214184


namespace NUMINAMATH_GPT_box_and_apples_weight_l2141_214193

theorem box_and_apples_weight
  (total_weight : ℝ)
  (weight_after_half : ℝ)
  (h1 : total_weight = 62.8)
  (h2 : weight_after_half = 31.8) :
  ∃ (box_weight apple_weight : ℝ), box_weight = 0.8 ∧ apple_weight = 62 :=
by
  sorry

end NUMINAMATH_GPT_box_and_apples_weight_l2141_214193


namespace NUMINAMATH_GPT_number_of_measures_of_C_l2141_214187

theorem number_of_measures_of_C (C D : ℕ) (h1 : C + D = 180) (h2 : ∃ k : ℕ, k ≥ 1 ∧ C = k * D) : 
  ∃ n : ℕ, n = 17 :=
by
  sorry

end NUMINAMATH_GPT_number_of_measures_of_C_l2141_214187


namespace NUMINAMATH_GPT_find_sum_3xyz_l2141_214143

variables (x y z : ℚ)

def equation1 : Prop := y + z = 18 - 4 * x
def equation2 : Prop := x + z = 16 - 4 * y
def equation3 : Prop := x + y = 9 - 4 * z

theorem find_sum_3xyz (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) : 
  3 * x + 3 * y + 3 * z = 43 / 2 := 
sorry

end NUMINAMATH_GPT_find_sum_3xyz_l2141_214143


namespace NUMINAMATH_GPT_find_m_for_opposite_solutions_l2141_214110

theorem find_m_for_opposite_solutions (x y m : ℝ) 
  (h1 : x = -y)
  (h2 : 3 * x + 5 * y = 2)
  (h3 : 2 * x + 7 * y = m - 18) : 
  m = 23 :=
sorry

end NUMINAMATH_GPT_find_m_for_opposite_solutions_l2141_214110


namespace NUMINAMATH_GPT_triangle_perpendicular_division_l2141_214109

variable (a b c : ℝ)
variable (b_gt_c : b > c)
variable (triangle : True)

theorem triangle_perpendicular_division (a b c : ℝ) (b_gt_c : b > c) :
  let CK := (1 / 2) * Real.sqrt (a^2 + b^2 - c^2)
  CK = (1 / 2) * Real.sqrt (a^2 + b^2 - c^2) :=
by
  sorry

end NUMINAMATH_GPT_triangle_perpendicular_division_l2141_214109


namespace NUMINAMATH_GPT_correct_tile_for_b_l2141_214166

structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

def TileI : Tile := {top := 5, right := 3, bottom := 1, left := 6}
def TileII : Tile := {top := 2, right := 6, bottom := 3, left := 5}
def TileIII : Tile := {top := 6, right := 1, bottom := 4, left := 2}
def TileIV : Tile := {top := 4, right := 5, bottom := 2, left := 1}

def RectangleBTile := TileIII

theorem correct_tile_for_b : RectangleBTile = TileIII :=
  sorry

end NUMINAMATH_GPT_correct_tile_for_b_l2141_214166


namespace NUMINAMATH_GPT_find_total_price_l2141_214117

-- Define the cost parameters
variables (sugar_price salt_price : ℝ)

-- Define the given conditions
def condition_1 : Prop := 2 * sugar_price + 5 * salt_price = 5.50
def condition_2 : Prop := sugar_price = 1.50

-- Theorem to be proven
theorem find_total_price (h1 : condition_1 sugar_price salt_price) (h2 : condition_2 sugar_price) : 
  3 * sugar_price + 1 * salt_price = 5.00 :=
by
  sorry

end NUMINAMATH_GPT_find_total_price_l2141_214117


namespace NUMINAMATH_GPT_rectangle_sides_l2141_214127

theorem rectangle_sides :
  ∀ (x : ℝ), 
    (3 * x = 8) ∧ (8 / 3 * 3 = 8) →
    ((2 * (3 * x + x) = 3 * x^2) ∧ (2 * (3 * (8 / 3) + (8 / 3)) = 3 * (8 / 3)^2) →
    x = 8 / 3
      ∧ 3 * x = 8) := 
by
  sorry

end NUMINAMATH_GPT_rectangle_sides_l2141_214127


namespace NUMINAMATH_GPT_michael_monica_age_ratio_l2141_214149

theorem michael_monica_age_ratio
  (x y : ℕ)
  (Patrick Michael Monica : ℕ)
  (h1 : Patrick = 3 * x)
  (h2 : Michael = 5 * x)
  (h3 : Monica = y)
  (h4 : y - Patrick = 64)
  (h5 : Patrick + Michael + Monica = 196) :
  Michael * 5 = Monica * 3 :=
by
  sorry

end NUMINAMATH_GPT_michael_monica_age_ratio_l2141_214149


namespace NUMINAMATH_GPT_Bob_wins_game_l2141_214174

theorem Bob_wins_game :
  ∀ (initial_set : Set ℕ),
    47 ∈ initial_set →
    2016 ∈ initial_set →
    (∀ (a b : ℕ), a ∈ initial_set → b ∈ initial_set → a > b → (a - b) ∉ initial_set → (a - b) ∈ initial_set) →
    (∀ (S : Set ℕ), S ⊆ initial_set → ∃ (n : ℕ), ∀ m ∈ S, m > n) → false :=
by
  sorry

end NUMINAMATH_GPT_Bob_wins_game_l2141_214174


namespace NUMINAMATH_GPT_part1_part2_l2141_214192

-- Definitions
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := -a^2 + a * b + a + 3

-- First proof: When a = -1 and b = 10, prove 4A - (3A - 2B) = -45
theorem part1 : 4 * A (-1) 10 - (3 * A (-1) 10 - 2 * B (-1) 10) = -45 := by
  sorry

-- Second proof: If a and b are reciprocal, prove 4A - (3A - 2B) = 10
theorem part2 (a b : ℝ) (hab : a * b = 1) : 4 * A a b - (3 * A a b - 2 * B a b) = 10 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l2141_214192


namespace NUMINAMATH_GPT_range_of_m_l2141_214199

namespace ProofProblem

-- Define propositions P and Q in Lean
def P (m : ℝ) : Prop := 2 * m > 1
def Q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

-- Assumptions
variables (m : ℝ)
axiom hP_or_Q : P m ∨ Q m
axiom hP_and_Q_false : ¬(P m ∧ Q m)

-- We need to prove the range of m
theorem range_of_m : m ∈ (Set.Icc (-2 : ℝ) (1 / 2 : ℝ) ∪ Set.Ioi (2 : ℝ)) :=
sorry

end ProofProblem

end NUMINAMATH_GPT_range_of_m_l2141_214199


namespace NUMINAMATH_GPT_math_problem_statements_l2141_214129

theorem math_problem_statements :
  (∀ a : ℝ, (a = -a) → (a = 0)) ∧
  (∀ b : ℝ, (1 / b = b) ↔ (b = 1 ∨ b = -1)) ∧
  (∀ c : ℝ, (c < -1) → (1 / c > c)) ∧
  (∀ d : ℝ, (d > 1) → (1 / d < d)) ∧
  (∃ n : ℕ, n > 0 ∧ ∀ m : ℕ, m > 0 → n ≤ m) :=
by {
  sorry
}

end NUMINAMATH_GPT_math_problem_statements_l2141_214129


namespace NUMINAMATH_GPT_max_points_right_triangle_l2141_214114

theorem max_points_right_triangle (n : ℕ) :
  (∀ (pts : Fin n → ℝ × ℝ), ∀ (i j k : Fin n), i ≠ j → j ≠ k → i ≠ k →
    let p1 := pts i
    let p2 := pts j
    let p3 := pts k
    let a := (p2.1 - p1.1)^2 + (p2.2 - p1.2)^2
    let b := (p3.1 - p2.1)^2 + (p3.2 - p2.2)^2
    let c := (p3.1 - p1.1)^2 + (p3.2 - p1.2)^2
    a + b = c ∨ b + c = a ∨ c + a = b) →
  n ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_points_right_triangle_l2141_214114
