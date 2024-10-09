import Mathlib

namespace coffee_per_cup_for_weak_l2426_242674

-- Defining the conditions
def weak_coffee_cups : ℕ := 12
def strong_coffee_cups : ℕ := 12
def total_coffee_tbsp : ℕ := 36
def weak_increase_factor : ℕ := 1
def strong_increase_factor : ℕ := 2

-- The theorem stating the problem
theorem coffee_per_cup_for_weak :
  ∃ W : ℝ, (weak_coffee_cups * W + strong_coffee_cups * (strong_increase_factor * W) = total_coffee_tbsp) ∧ (W = 1) :=
  sorry

end coffee_per_cup_for_weak_l2426_242674


namespace hyperbola_asymptote_l2426_242663

theorem hyperbola_asymptote (y x : ℝ) :
  (y^2 / 9 - x^2 / 16 = 1) → (y = x * 3 / 4 ∨ y = -x * 3 / 4) :=
sorry

end hyperbola_asymptote_l2426_242663


namespace product_of_radii_l2426_242678

theorem product_of_radii (x y r₁ r₂ : ℝ) (hx : 0 < x) (hy : 0 < y)
  (hr₁ : (x - r₁)^2 + (y - r₁)^2 = r₁^2)
  (hr₂ : (x - r₂)^2 + (y - r₂)^2 = r₂^2)
  (hroots : r₁ + r₂ = 2 * (x + y)) : r₁ * r₂ = x^2 + y^2 := by
  sorry

end product_of_radii_l2426_242678


namespace equation_holds_true_l2426_242654

theorem equation_holds_true (a b : ℝ) (h₁ : a ≠ 0) (h₂ : 2 * b - a ≠ 0) :
  ((a + 2 * b) / a = b / (2 * b - a)) ↔ 
  (a = -b * (1 + Real.sqrt 17) / 2 ∨ a = -b * (1 - Real.sqrt 17) / 2) := 
sorry

end equation_holds_true_l2426_242654


namespace charlie_original_price_l2426_242623

theorem charlie_original_price (acorns_Alice acorns_Bob acorns_Charlie ν_Alice ν_Bob discount price_Charlie_before_discount price_Charlie_after_discount total_paid_by_AliceBob total_acorns_AliceBob average_price_per_acorn price_per_acorn_Alice price_per_acorn_Bob total_paid_Alice total_paid_Bob: ℝ) :
  acorns_Alice = 3600 →
  acorns_Bob = 2400 →
  acorns_Charlie = 4500 →
  ν_Bob = 6000 →
  ν_Alice = 9 * ν_Bob →
  price_per_acorn_Bob = ν_Bob / acorns_Bob →
  price_per_acorn_Alice = ν_Alice / acorns_Alice →
  total_paid_Alice = acorns_Alice * price_per_acorn_Alice →
  total_paid_Bob = ν_Bob →
  total_paid_by_AliceBob = total_paid_Alice + total_paid_Bob →
  total_acorns_AliceBob = acorns_Alice + acorns_Bob →
  average_price_per_acorn = total_paid_by_AliceBob / total_acorns_AliceBob →
  discount = 10 / 100 →
  price_Charlie_after_discount = average_price_per_acorn * (1 - discount) →
  price_Charlie_before_discount = average_price_per_acorn →
  price_Charlie_before_discount = 14.50 →
  price_per_acorn_Alice = 22.50 →
  price_Charlie_before_discount * acorns_Charlie = 4500 * 14.50 :=
by sorry

end charlie_original_price_l2426_242623


namespace my_inequality_l2426_242668

open Real

variable {a b c : ℝ}

theorem my_inequality 
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : 0 < c)
  (h4 : a * b + b * c + c * a = 1) :
  sqrt (a ^ 3 + a) + sqrt (b ^ 3 + b) + sqrt (c ^ 3 + c) ≥ 2 * sqrt (a + b + c) := 
  sorry

end my_inequality_l2426_242668


namespace unique_solution_of_system_l2426_242699

theorem unique_solution_of_system (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h1 : x * (x + y + z) = 26) (h2 : y * (x + y + z) = 27) (h3 : z * (x + y + z) = 28) :
  x = 26 / 9 ∧ y = 3 ∧ z = 28 / 9 :=
by
  sorry

end unique_solution_of_system_l2426_242699


namespace park_area_l2426_242697

theorem park_area (w : ℝ) (h1 : 2 * (w + 3 * w) = 72) : w * (3 * w) = 243 :=
by
  sorry

end park_area_l2426_242697


namespace correct_expression_l2426_242622

variable (a b : ℝ)

theorem correct_expression : (∃ x, x = 3 * a + b^2) ∧ 
    (x = (3 * a + b)^2 ∨ x = 3 * (a + b)^2 ∨ x = 3 * a + b^2 ∨ x = (a + 3 * b)^2) → 
    x = 3 * a + b^2 := by sorry

end correct_expression_l2426_242622


namespace has_real_root_neg_one_l2426_242638

theorem has_real_root_neg_one : 
  (-1)^2 - (-1) - 2 = 0 :=
by 
  sorry

end has_real_root_neg_one_l2426_242638


namespace sqrt_of_sum_of_powers_l2426_242671

theorem sqrt_of_sum_of_powers : Real.sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 := by
  sorry

end sqrt_of_sum_of_powers_l2426_242671


namespace machine_rate_ratio_l2426_242614

theorem machine_rate_ratio (A B : ℕ) (h1 : ∃ A : ℕ, 8 * A = 8 * A)
  (h2 : ∃ W : ℕ, W = 8 * A)
  (h3 : ∃ W1 : ℕ, W1 = 6 * A)
  (h4 : ∃ W2 : ℕ, W2 = 2 * A)
  (h5 : ∃ B : ℕ, 8 * B = 2 * A) :
  (B:ℚ) / (A:ℚ) = 1 / 4 :=
by sorry

end machine_rate_ratio_l2426_242614


namespace original_time_to_cover_distance_l2426_242691

theorem original_time_to_cover_distance (S : ℝ) (T : ℝ) (D : ℝ) :
  (0.8 * S) * (T + 10 / 60) = S * T → T = 2 / 3 :=
  by sorry

end original_time_to_cover_distance_l2426_242691


namespace work_completion_days_l2426_242682

theorem work_completion_days (D : ℕ) (W : ℕ) :
  (D : ℕ) = 6 :=
by 
  -- define constants and given conditions
  let original_men := 10
  let additional_men := 10
  let early_days := 3

  -- define the premise
  -- work done with original men in original days
  have work_done_original : W = (original_men * D) := sorry
  -- work done with additional men in reduced days
  have work_done_with_additional : W = ((original_men + additional_men) * (D - early_days)) := sorry

  -- prove the equality from the condition
  have eq : original_men * D = (original_men + additional_men) * (D - early_days) := sorry

  -- simplify to solve for D
  have solution : D = 6 := sorry

  exact solution

end work_completion_days_l2426_242682


namespace proof_m_plus_n_l2426_242637

variable (m n : ℚ) -- Defining m and n as rational numbers (ℚ)
-- Conditions from the problem:
axiom condition1 : 2 * m + 5 * n + 8 = 1
axiom condition2 : m - n - 3 = 1

-- Proof statement (theorem) that needs to be established:
theorem proof_m_plus_n : m + n = -2/7 :=
by
-- Since the proof is not required, we use "sorry" to placeholder the proof.
sorry

end proof_m_plus_n_l2426_242637


namespace find_g_one_l2426_242677

variable {α : Type} [AddGroup α]

def is_odd (f : α → α) : Prop :=
∀ x, f (-x) = - f x

def is_even (g : α → α) : Prop :=
∀ x, g (-x) = g x

theorem find_g_one
  (f g : ℝ → ℝ)
  (h_odd : is_odd f)
  (h_even : is_even g)
  (h1 : f (-1) + g 1 = 2)
  (h2 : f 1 + g (-1) = 4) :
  g 1 = 3 := by
  sorry

end find_g_one_l2426_242677


namespace mean_equality_l2426_242695

theorem mean_equality (y : ℝ) (h : (6 + 9 + 18) / 3 = (12 + y) / 2) : y = 10 :=
by sorry

end mean_equality_l2426_242695


namespace fixed_line_of_midpoint_l2426_242653

theorem fixed_line_of_midpoint
  (A B : ℝ × ℝ)
  (H : ∀ (P : ℝ × ℝ), (P = A ∨ P = B) → (P.1^2 / 3 - P.2^2 / 6 = 1))
  (slope_l : (B.2 - A.2) / (B.1 - A.1) = 2)
  (midpoint_lies : (A.1 + B.1) / 2 = (A.2 + B.2) / 2) :
  ∀ (M : ℝ × ℝ), (M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2) → M.1 - M.2 = 0 :=
by
  sorry

end fixed_line_of_midpoint_l2426_242653


namespace profit_percentage_no_initial_discount_l2426_242602

theorem profit_percentage_no_initial_discount
  (CP : ℝ := 100)
  (bulk_discount : ℝ := 0.02)
  (sales_tax : ℝ := 0.065)
  (no_discount_price : ℝ := CP - CP * bulk_discount)
  (selling_price : ℝ := no_discount_price + no_discount_price * sales_tax)
  (profit : ℝ := selling_price - CP) :
  (profit / CP) * 100 = 4.37 :=
by
  -- proof here
  sorry

end profit_percentage_no_initial_discount_l2426_242602


namespace percentage_difference_l2426_242693

theorem percentage_difference :
  let a := 0.80 * 40
  let b := (4 / 5) * 15
  a - b = 20 := by
sorry

end percentage_difference_l2426_242693


namespace total_cost_eq_1400_l2426_242694

theorem total_cost_eq_1400 (stove_cost : ℝ) (wall_repair_fraction : ℝ) (wall_repair_cost : ℝ) (total_cost : ℝ)
  (h₁ : stove_cost = 1200)
  (h₂ : wall_repair_fraction = 1/6)
  (h₃ : wall_repair_cost = stove_cost * wall_repair_fraction)
  (h₄ : total_cost = stove_cost + wall_repair_cost) :
  total_cost = 1400 :=
sorry

end total_cost_eq_1400_l2426_242694


namespace find_linear_function_l2426_242629

theorem find_linear_function (f : ℝ → ℝ) (hf_inc : ∀ x y, x < y → f x < f y)
  (hf_lin : ∃ a b, a > 0 ∧ ∀ x, f x = a * x + b)
  (h_comp : ∀ x, f (f x) = 4 * x + 3) :
  ∀ x, f x = 2 * x + 1 :=
by
  sorry

end find_linear_function_l2426_242629


namespace time_saved_is_six_minutes_l2426_242639

-- Conditions
def distance_monday : ℝ := 3
def distance_wednesday : ℝ := 4
def distance_friday : ℝ := 5

def speed_monday : ℝ := 6
def speed_wednesday : ℝ := 4
def speed_friday : ℝ := 5

def speed_constant : ℝ := 5

-- Question (proof statement)
theorem time_saved_is_six_minutes : 
  (distance_monday / speed_monday + distance_wednesday / speed_wednesday + distance_friday / speed_friday) - (distance_monday + distance_wednesday + distance_friday) / speed_constant = 0.1 :=
by
  sorry

end time_saved_is_six_minutes_l2426_242639


namespace number_of_balls_sold_l2426_242600

-- Let n be the number of balls sold
variable (n : ℕ)

-- The given conditions
def selling_price : ℕ := 720
def cost_price_per_ball : ℕ := 60
def loss := 5 * cost_price_per_ball

-- Prove that if the selling price of 'n' balls is Rs. 720 and 
-- the loss is equal to the cost price of 5 balls, then the 
-- number of balls sold (n) is 17.
theorem number_of_balls_sold (h1 : selling_price = 720) 
                             (h2 : cost_price_per_ball = 60) 
                             (h3 : loss = 5 * cost_price_per_ball) 
                             (hsale : n * cost_price_per_ball - selling_price = loss) : 
  n = 17 := 
by
  sorry

end number_of_balls_sold_l2426_242600


namespace inequality_solution_l2426_242698

theorem inequality_solution (x : ℝ) :
  ( (x^2 + 3*x + 3) > 0 ) → ( ((x^2 + 3*x + 3)^(5*x^3 - 3*x^2)) ≤ ((x^2 + 3*x + 3)^(3*x^3 + 5*x)) )
  ↔ ( x ∈ (Set.Iic (-2) ∪ ({-1} : Set ℝ) ∪ Set.Icc 0 (5/2)) ) :=
by
  sorry

end inequality_solution_l2426_242698


namespace select_numbers_with_sum_713_l2426_242606

noncomputable def is_suitable_sum (numbers : List ℤ) : Prop :=
  ∃ subset : List ℤ, subset ⊆ numbers ∧ (subset.sum % 10000 = 713)

theorem select_numbers_with_sum_713 :
  ∀ numbers : List ℤ, 
  numbers.length = 1000 → 
  (∀ n ∈ numbers, n % 2 = 1 ∧ n % 5 ≠ 0) →
  is_suitable_sum numbers :=
sorry

end select_numbers_with_sum_713_l2426_242606


namespace prob_odd_sum_l2426_242667

-- Given conditions on the spinners
def spinner_P := [1, 2, 3]
def spinner_Q := [2, 4, 6]
def spinner_R := [1, 3, 5]

-- Probability of spinner P landing on an even number is 1/3
def prob_even_P : ℚ := 1 / 3

-- Probability of odd sum from spinners P, Q, and R
theorem prob_odd_sum : 
  (prob_even_P = 1 / 3) → 
  ∃ p : ℚ, p = 1 / 3 :=
by
  sorry

end prob_odd_sum_l2426_242667


namespace price_of_orange_l2426_242608

-- Define relevant conditions
def price_apple : ℝ := 1.50
def morning_apples : ℕ := 40
def morning_oranges : ℕ := 30
def afternoon_apples : ℕ := 50
def afternoon_oranges : ℕ := 40
def total_sales : ℝ := 205

-- Define the proof problem
theorem price_of_orange (O : ℝ) 
  (h : (morning_apples * price_apple + morning_oranges * O) + 
       (afternoon_apples * price_apple + afternoon_oranges * O) = total_sales) : 
  O = 1 :=
by
  sorry

end price_of_orange_l2426_242608


namespace f_minimum_at_l2426_242670

noncomputable def f (x : ℝ) : ℝ := x * 2^x

theorem f_minimum_at : ∀ x : ℝ, x = -Real.log 2 → (∀ y : ℝ, f y ≥ f x) :=
by
  sorry

end f_minimum_at_l2426_242670


namespace intersection_is_empty_l2426_242630

def A : Set ℝ := { α | ∃ k : ℤ, α = (5 * k * Real.pi) / 3 }
def B : Set ℝ := { β | ∃ k : ℤ, β = (3 * k * Real.pi) / 2 }

theorem intersection_is_empty : A ∩ B = ∅ :=
by
  sorry

end intersection_is_empty_l2426_242630


namespace largest_number_eq_l2426_242661

theorem largest_number_eq (x y z : ℚ) (h1 : x + y + z = 82) (h2 : z - y = 10) (h3 : y - x = 4) :
  z = 106 / 3 :=
sorry

end largest_number_eq_l2426_242661


namespace volume_of_prism_l2426_242626

variable (x y z : ℝ)
variable (h1 : x * y = 15)
variable (h2 : y * z = 10)
variable (h3 : x * z = 6)

theorem volume_of_prism : x * y * z = 30 :=
by
  sorry

end volume_of_prism_l2426_242626


namespace interior_lattice_points_of_triangle_l2426_242635

-- Define the vertices of the triangle
def A : (ℤ × ℤ) := (0, 99)
def B : (ℤ × ℤ) := (5, 100)
def C : (ℤ × ℤ) := (2003, 500)

-- The problem is to find the number of interior lattice points
-- according to Pick's Theorem (excluding boundary points).

theorem interior_lattice_points_of_triangle :
  let I : ℤ := 0 -- number of interior lattice points
  I = 0 :=
by
  sorry

end interior_lattice_points_of_triangle_l2426_242635


namespace unique_records_l2426_242618

variable (Samantha_records : Nat)
variable (shared_records : Nat)
variable (Lily_unique_records : Nat)

theorem unique_records (h1 : Samantha_records = 24) (h2 : shared_records = 15) (h3 : Lily_unique_records = 9) :
  let Samantha_unique_records := Samantha_records - shared_records
  Samantha_unique_records + Lily_unique_records = 18 :=
by
  sorry

end unique_records_l2426_242618


namespace rectangle_width_length_ratio_l2426_242627

theorem rectangle_width_length_ratio (w l : ℕ) 
  (h1 : l = 12) 
  (h2 : 2 * w + 2 * l = 36) : 
  w / l = 1 / 2 := 
by 
  sorry

end rectangle_width_length_ratio_l2426_242627


namespace nurses_count_l2426_242616

theorem nurses_count (total personnel_ratio d_ratio n_ratio : ℕ)
  (ratio_eq: personnel_ratio = 280)
  (ratio_condition: d_ratio = 5)
  (person_count: n_ratio = 9) :
  n_ratio * (personnel_ratio / (d_ratio + n_ratio)) = 180 := by
  -- Total personnel = 280
  -- Ratio of doctors to nurses = 5/9
  -- Prove that the number of nurses is 180
  -- sorry is used to skip proof
  sorry

end nurses_count_l2426_242616


namespace expression_equivalence_l2426_242690

theorem expression_equivalence :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 :=
by
  sorry

end expression_equivalence_l2426_242690


namespace parabola_vertex_coordinates_l2426_242636

theorem parabola_vertex_coordinates :
  ∀ x : ℝ, (3 * (x - 7) ^ 2 + 5) = 3 * (x - 7) ^ 2 + 5 := by
  sorry

end parabola_vertex_coordinates_l2426_242636


namespace geometric_sequence_sum_l2426_242662

-- We state the main problem in Lean as a theorem.
theorem geometric_sequence_sum (S : ℕ → ℕ) (S_4_eq : S 4 = 8) (S_8_eq : S 8 = 24) : S 12 = 88 :=
  sorry

end geometric_sequence_sum_l2426_242662


namespace remainder_of_sum_mod_13_l2426_242646

theorem remainder_of_sum_mod_13 :
  ∀ (D : ℕ) (k1 k2 : ℕ),
    D = 13 →
    (242 = k1 * D + 8) →
    (698 = k2 * D + 9) →
    (242 + 698) % D = 4 :=
by
  intros D k1 k2 hD h242 h698
  sorry

end remainder_of_sum_mod_13_l2426_242646


namespace point_N_in_second_quadrant_l2426_242685

theorem point_N_in_second_quadrant (a b : ℝ) (h1 : 1 + a < 0) (h2 : 2 * b - 1 < 0) :
    (a - 1 < 0) ∧ (1 - 2 * b > 0) :=
by
  -- Insert proof here
  sorry

end point_N_in_second_quadrant_l2426_242685


namespace domain_of_log_function_l2426_242640

noncomputable def domain_f (k : ℤ) : Set ℝ :=
  {x : ℝ | (2 * k * Real.pi - Real.pi / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3) ∨
           (2 * k * Real.pi + 2 * Real.pi / 3 < x ∧ x < 2 * k * Real.pi + 4 * Real.pi / 3)}

theorem domain_of_log_function :
  ∀ x : ℝ, (∃ k : ℤ, (2 * k * Real.pi - Real.pi / 3 < x ∧ x < 2 * k * Real.pi + Real.pi / 3) ∨
                      (2 * k * Real.pi + 2 * Real.pi / 3 < x ∧ x < 2 * k * Real.pi + 4 * Real.pi / 3))
  ↔ (3 - 4 * Real.sin x ^ 2 > 0) :=
by {
  sorry
}

end domain_of_log_function_l2426_242640


namespace thermostat_range_l2426_242659

theorem thermostat_range (T : ℝ) : 
  |T - 22| ≤ 6 ↔ 16 ≤ T ∧ T ≤ 28 := 
by sorry

end thermostat_range_l2426_242659


namespace cylinder_volume_expansion_l2426_242603

theorem cylinder_volume_expansion (r h : ℝ) :
  (π * (2 * r)^2 * h) = 4 * (π * r^2 * h) :=
by
  sorry

end cylinder_volume_expansion_l2426_242603


namespace machine_makes_12_shirts_l2426_242645

def shirts_per_minute : ℕ := 2
def minutes_worked : ℕ := 6

def total_shirts_made : ℕ := shirts_per_minute * minutes_worked

theorem machine_makes_12_shirts :
  total_shirts_made = 12 :=
by
  -- proof placeholder
  sorry

end machine_makes_12_shirts_l2426_242645


namespace linear_inequality_m_eq_one_l2426_242675

theorem linear_inequality_m_eq_one
  (m : ℤ)
  (h1 : |m| = 1)
  (h2 : m + 1 ≠ 0) :
  m = 1 :=
sorry

end linear_inequality_m_eq_one_l2426_242675


namespace total_students_l2426_242657

theorem total_students (n x : ℕ) (h1 : 3 * n + 48 = 6 * n) (h2 : 4 * n + x = 2 * n + 2 * x) : n = 16 :=
by
  sorry

end total_students_l2426_242657


namespace correct_operation_l2426_242624

theorem correct_operation (a : ℝ) : a^8 / a^2 = a^6 :=
by
  -- proof will go here, let's use sorry to indicate it's unfinished
  sorry

end correct_operation_l2426_242624


namespace range_of_a_l2426_242692

variable (a : ℝ)

def proposition_p : Prop :=
  ∃ x₀ : ℝ, x₀^2 - a * x₀ + a = 0

def proposition_q : Prop :=
  ∀ x : ℝ, 1 < x → x + 1 / (x - 1) ≥ a

theorem range_of_a (h : ¬proposition_p a ∧ proposition_q a) : 0 < a ∧ a ≤ 3 :=
sorry

end range_of_a_l2426_242692


namespace count_sums_of_fours_and_fives_l2426_242609

theorem count_sums_of_fours_and_fives :
  ∃ n, (∀ x y : ℕ, 4 * x + 5 * y = 1800 ↔ (x = 0 ∨ x ≤ 1800) ∧ (y = 0 ∨ y ≤ 1800)) ∧ n = 201 :=
by
  -- definition and theorem statement is complete. The proof is omitted.
  sorry

end count_sums_of_fours_and_fives_l2426_242609


namespace contractor_engaged_days_l2426_242625

variable (x : ℕ)
variable (days_absent : ℕ) (wage_per_day fine_per_day_Rs total_payment_Rs : ℝ)

theorem contractor_engaged_days :
  days_absent = 10 →
  wage_per_day = 25 →
  fine_per_day_Rs = 7.5 →
  total_payment_Rs = 425 →
  (x * wage_per_day - days_absent * fine_per_day_Rs = total_payment_Rs) →
  x = 20 :=
by
  sorry

end contractor_engaged_days_l2426_242625


namespace geometric_sequence_common_ratio_l2426_242680

/--
  Given a geometric sequence with the first three terms:
  a₁ = 27,
  a₂ = 54,
  a₃ = 108,
  prove that the common ratio is r = 2.
-/
theorem geometric_sequence_common_ratio :
  let a₁ := 27
  let a₂ := 54
  let a₃ := 108
  ∃ r : ℕ, (a₂ = r * a₁) ∧ (a₃ = r * a₂) ∧ r = 2 := by
  sorry

end geometric_sequence_common_ratio_l2426_242680


namespace value_of_abc_l2426_242628

noncomputable def f (x a b c : ℝ) := |(1 - x^2) * (x^2 + a * x + b)| - c

theorem value_of_abc :
  (∀ x : ℝ, f (x + 4) 8 15 9 = f (-x) 8 15 9) ∧
  (∃ x : ℝ, f x 8 15 9 = 0) ∧
  (∃ x : ℝ, f (-(x-4)) 8 15 9 = 0) ∧
  (∀ c : ℝ, c ≠ 0) →
  8 + 15 + 9 = 32 :=
by sorry

end value_of_abc_l2426_242628


namespace solve_congruence_l2426_242664

theorem solve_congruence : ∃ n : ℕ, 0 ≤ n ∧ n < 43 ∧ 11 * n % 43 = 7 :=
by
  sorry

end solve_congruence_l2426_242664


namespace suitcase_combinations_l2426_242621

def count_odd_numbers (n : Nat) : Nat := n / 2

def count_multiples_of_4 (n : Nat) : Nat := n / 4

def count_multiples_of_5 (n : Nat) : Nat := n / 5

theorem suitcase_combinations : count_odd_numbers 40 * count_multiples_of_4 40 * count_multiples_of_5 40 = 1600 :=
by
  sorry

end suitcase_combinations_l2426_242621


namespace percentage_increase_x_y_l2426_242656

theorem percentage_increase_x_y (Z Y X : ℝ) (h1 : Z = 300) (h2 : Y = 1.20 * Z) (h3 : X = 1110 - Y - Z) :
  ((X - Y) / Y) * 100 = 25 :=
by
  sorry

end percentage_increase_x_y_l2426_242656


namespace find_a_b_find_m_l2426_242610

-- Define the parabola and the points it passes through
def parabola (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

-- The conditions based on the given problem
def condition1 (a b : ℝ) : Prop := parabola a b 1 = -2
def condition2 (a b : ℝ) : Prop := parabola a b (-2) = 13

-- Part 1: Proof for a and b
theorem find_a_b : ∃ a b : ℝ, condition1 a b ∧ condition2 a b ∧ a = 1 ∧ b = -4 :=
by sorry

-- Part 2: Given y equation and the specific points
def parabola2 (x : ℝ) : ℝ := x^2 - 4 * x + 1

-- Conditions for the second part
def condition3 : Prop := parabola2 5 = 6
def condition4 (m : ℝ) : Prop := parabola2 m = 12 - 6

-- Theorem statement for the second part
theorem find_m : ∃ m : ℝ, condition3 ∧ condition4 m ∧ m = -1 :=
by sorry

end find_a_b_find_m_l2426_242610


namespace value_of_b_minus_a_l2426_242684

open Real

def condition (a b : ℝ) : Prop := 
  abs a = 3 ∧ abs b = 2 ∧ a + b > 0

theorem value_of_b_minus_a (a b : ℝ) (h : condition a b) :
  b - a = -1 ∨ b - a = -5 :=
  sorry

end value_of_b_minus_a_l2426_242684


namespace number_of_games_played_l2426_242641

-- Define our conditions
def teams : ℕ := 14
def games_per_pair : ℕ := 5

-- Define the function to calculate the number of combinations
def combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Define the expected total games
def total_games : ℕ := 455

-- Statement asserting that given the conditions, the number of games played in the season is total_games
theorem number_of_games_played : (combinations teams 2) * games_per_pair = total_games := 
by 
  sorry

end number_of_games_played_l2426_242641


namespace addition_subtraction_result_l2426_242681

theorem addition_subtraction_result :
  27474 + 3699 + 1985 - 2047 = 31111 :=
by {
  sorry
}

end addition_subtraction_result_l2426_242681


namespace simplify_expression_eq_sqrt3_l2426_242666

theorem simplify_expression_eq_sqrt3
  (a : ℝ)
  (h : a = Real.sqrt 3 + 1) :
  ( (a + 1) / a / (a - (1 + 2 * a^2) / (3 * a)) ) = Real.sqrt 3 := sorry

end simplify_expression_eq_sqrt3_l2426_242666


namespace initial_investment_l2426_242689

theorem initial_investment (P : ℝ) 
  (h1: ∀ (r : ℝ) (n : ℕ), r = 0.20 ∧ n = 3 → P * (1 + r)^n = P * 1.728)
  (h2: ∀ (A : ℝ), A = P * 1.728 → 3 * A = 5.184 * P)
  (h3: ∀ (P_new : ℝ) (r_new : ℝ), P_new = 5.184 * P ∧ r_new = 0.15 → P_new * (1 + r_new) = 5.9616 * P)
  (h4: 5.9616 * P = 59616)
  : P = 10000 :=
sorry

end initial_investment_l2426_242689


namespace y_intercept_of_line_l2426_242687

theorem y_intercept_of_line (x y : ℝ) (h : 4 * x + 7 * y = 28) : y = 4 :=
by
  -- The proof steps will go here.
  sorry

end y_intercept_of_line_l2426_242687


namespace black_to_white_ratio_l2426_242650

/-- 
Given:
- The original square pattern consists of 13 black tiles and 23 white tiles
- Attaching a border of black tiles around the original 6x6 square pattern results in an 8x8 square pattern

To prove:
- The ratio of black tiles to white tiles in the extended 8x8 pattern is 41/23.
-/
theorem black_to_white_ratio (b_orig w_orig b_added b_total w_total : ℕ) 
  (h_black_orig: b_orig = 13)
  (h_white_orig: w_orig = 23)
  (h_size_orig: 6 * 6 = b_orig + w_orig)
  (h_size_ext: 8 * 8 = (b_orig + b_added) + w_orig)
  (h_b_added: b_added = 28)
  (h_b_total: b_total = b_orig + b_added)
  (h_w_total: w_total = w_orig)
  :
  b_total / w_total = 41 / 23 :=
by
  sorry

end black_to_white_ratio_l2426_242650


namespace find_a_minus_b_l2426_242644

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.log x + b * x + 1

theorem find_a_minus_b (a b : ℝ)
  (h1 : deriv (f a b) 1 = -2)
  (h2 : deriv (f a b) (2 / 3) = 0) :
  a - b = 10 :=
sorry

end find_a_minus_b_l2426_242644


namespace min_value_of_ratio_l2426_242676

theorem min_value_of_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y = 1) :
  (4 / x + 1 / y) ≥ 6 + 4 * Real.sqrt 2 :=
sorry

end min_value_of_ratio_l2426_242676


namespace abs_lt_inequality_solution_l2426_242605

theorem abs_lt_inequality_solution (x : ℝ) : |x - 1| < 2 ↔ -1 < x ∧ x < 3 :=
by sorry

end abs_lt_inequality_solution_l2426_242605


namespace eq1_solution_eq2_solution_l2426_242647

theorem eq1_solution (x : ℝ) : (3 * x * (x - 1) = 2 - 2 * x) ↔ (x = 1 ∨ x = -2/3) :=
sorry

theorem eq2_solution (x : ℝ) : (3 * x^2 - 6 * x + 2 = 0) ↔ (x = 1 + (Real.sqrt 3) / 3 ∨ x = 1 - (Real.sqrt 3) / 3) :=
sorry

end eq1_solution_eq2_solution_l2426_242647


namespace cars_with_air_bags_l2426_242633

/--
On a car lot with 65 cars:
- Some have air-bags.
- 30 have power windows.
- 12 have both air-bag and power windows.
- 2 have neither air-bag nor power windows.

Prove that the number of cars with air-bags is 45.
-/
theorem cars_with_air_bags 
    (total_cars : ℕ)
    (cars_with_power_windows : ℕ)
    (cars_with_both : ℕ)
    (cars_with_neither : ℕ)
    (total_cars_eq : total_cars = 65)
    (cars_with_power_windows_eq : cars_with_power_windows = 30)
    (cars_with_both_eq : cars_with_both = 12)
    (cars_with_neither_eq : cars_with_neither = 2) :
    ∃ (A : ℕ), A = 45 :=
by
  sorry

end cars_with_air_bags_l2426_242633


namespace milk_mixture_l2426_242649

theorem milk_mixture (x : ℝ) : 
  (2.4 + 0.1 * x) / (8 + x) = 0.2 → x = 8 :=
by
  sorry

end milk_mixture_l2426_242649


namespace find_integer_l2426_242683

theorem find_integer (n : ℤ) 
  (h1 : 50 ≤ n ∧ n ≤ 150)
  (h2 : n % 7 = 0)
  (h3 : n % 9 = 3)
  (h4 : n % 6 = 3) : 
  n = 63 := by 
  sorry

end find_integer_l2426_242683


namespace min_sum_of_factors_l2426_242643

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 1176) :
  a + b + c ≥ 59 :=
sorry

end min_sum_of_factors_l2426_242643


namespace evaluate_expression_l2426_242632

variable {R : Type} [CommRing R]

theorem evaluate_expression (x y z w : R) :
  (x - (y - 3 * z + w)) - ((x - y + w) - 3 * z) = 6 * z - 2 * w :=
by
  sorry

end evaluate_expression_l2426_242632


namespace average_minutes_proof_l2426_242688

noncomputable def average_minutes_heard (total_minutes : ℕ) (total_attendees : ℕ) (full_listened_fraction : ℚ) (none_listened_fraction : ℚ) (half_remainder_fraction : ℚ) : ℚ := 
  let full_listeners := full_listened_fraction * total_attendees
  let none_listeners := none_listened_fraction * total_attendees
  let remaining_listeners := total_attendees - full_listeners - none_listeners
  let half_listeners := half_remainder_fraction * remaining_listeners
  let quarter_listeners := remaining_listeners - half_listeners
  let total_heard := (full_listeners * total_minutes) + (none_listeners * 0) + (half_listeners * (total_minutes / 2)) + (quarter_listeners * (total_minutes / 4))
  total_heard / total_attendees

theorem average_minutes_proof : 
  average_minutes_heard 120 100 (30/100) (15/100) (40/100) = 59.1 := 
by
  sorry

end average_minutes_proof_l2426_242688


namespace solve_for_y_l2426_242613

theorem solve_for_y 
  (a b c d y : ℚ) 
  (h₀ : a ≠ b) 
  (h₁ : a ≠ 0) 
  (h₂ : c ≠ d) 
  (h₃ : (b + y) / (a + y) = d / c) : 
  y = (a * d - b * c) / (c - d) :=
by
  sorry

end solve_for_y_l2426_242613


namespace each_child_play_time_l2426_242665

-- Define the conditions
def number_of_children : ℕ := 6
def pair_play_time : ℕ := 120
def pairs_playing_at_a_time : ℕ := 2

-- Define main theorem
theorem each_child_play_time : 
  (pairs_playing_at_a_time * pair_play_time) / number_of_children = 40 :=
sorry

end each_child_play_time_l2426_242665


namespace original_price_of_suit_l2426_242655

theorem original_price_of_suit (P : ℝ) (hP : 0.70 * 1.30 * P = 182) : P = 200 :=
by
  sorry

end original_price_of_suit_l2426_242655


namespace product_quantities_l2426_242631

theorem product_quantities (a b x y : ℝ) 
  (h1 : a * x + b * y = 1500)
  (h2 : (a + 1.5) * (x - 10) + (b + 1) * y = 1529)
  (h3 : (a + 1) * (x - 5) + (b + 1) * y = 1563.5)
  (h4 : 205 < 2 * x + y ∧ 2 * x + y < 210) :
  (x + 2 * y = 186) ∧ (73 ≤ x ∧ x ≤ 75) :=
by
  sorry

end product_quantities_l2426_242631


namespace fold_minus2_2_3_coincides_neg3_fold_minus1_3_7_coincides_neg5_fold_distanceA_to_B_coincide_l2426_242686

section FoldingNumberLine

-- Part (1)
def coincides_point_3_if_minus2_2_fold (x : ℝ) : Prop :=
  x = -3

theorem fold_minus2_2_3_coincides_neg3 :
  coincides_point_3_if_minus2_2_fold 3 :=
by
  sorry

-- Part (2) ①
def coincides_point_7_if_minus1_3_fold (x : ℝ) : Prop :=
  x = -5

theorem fold_minus1_3_7_coincides_neg5 :
  coincides_point_7_if_minus1_3_fold 7 :=
by
  sorry

-- Part (2) ②
def B_position_after_folding (m : ℝ) (h : m > 0) (A B : ℝ) : Prop :=
  B = 1 + m / 2

theorem fold_distanceA_to_B_coincide (m : ℝ) (h : m > 0) (A B : ℝ) :
  B_position_after_folding m h A B :=
by
  sorry

end FoldingNumberLine

end fold_minus2_2_3_coincides_neg3_fold_minus1_3_7_coincides_neg5_fold_distanceA_to_B_coincide_l2426_242686


namespace water_flow_speed_l2426_242642

/-- A person rows a boat for 15 li. If he rows at his usual speed,
the time taken to row downstream is 5 hours less than rowing upstream.
If he rows at twice his usual speed, the time taken to row downstream
is only 1 hour less than rowing upstream. 
Prove that the speed of the water flow is 2 li/hour.
-/
theorem water_flow_speed (y x : ℝ)
  (h1 : 15 / (y - x) - 15 / (y + x) = 5)
  (h2 : 15 / (2 * y - x) - 15 / (2 * y + x) = 1) :
  x = 2 := 
sorry

end water_flow_speed_l2426_242642


namespace total_problems_is_correct_l2426_242652

/-- Definition of the number of pages of math homework. -/
def math_pages : ℕ := 2

/-- Definition of the number of pages of reading homework. -/
def reading_pages : ℕ := 4

/-- Definition that each page of homework contains 5 problems. -/
def problems_per_page : ℕ := 5

/-- The proof statement: given the number of pages of math and reading homework,
    and the number of problems per page, prove that the total number of problems is 30. -/
theorem total_problems_is_correct : (math_pages + reading_pages) * problems_per_page = 30 := by
  sorry

end total_problems_is_correct_l2426_242652


namespace distance_to_intersection_of_quarter_circles_eq_zero_l2426_242620

open Real

theorem distance_to_intersection_of_quarter_circles_eq_zero (s : ℝ) :
  let A := (0, 0)
  let B := (s, 0)
  let C := (s, s)
  let D := (0, s)
  let center := (s / 2, s / 2)
  let arc_from_A := {p : ℝ × ℝ | p.1^2 + p.2^2 = s^2}
  let arc_from_C := {p : ℝ × ℝ | (p.1 - s)^2 + (p.2 - s)^2 = s^2}
  (center ∈ arc_from_A ∧ center ∈ arc_from_C) →
  let (ix, iy) := (s / 2, s / 2)
  dist (ix, iy) center = 0 :=
by
  sorry

end distance_to_intersection_of_quarter_circles_eq_zero_l2426_242620


namespace skier_total_time_l2426_242660

variable (t1 t2 t3 : ℝ)

-- Conditions
def condition1 : Prop := t1 + t2 = 40.5
def condition2 : Prop := t2 + t3 = 37.5
def condition3 : Prop := 1 / t2 = 2 / (t1 + t3)

-- Theorem to prove total time is 58.5 minutes
theorem skier_total_time (h1 : condition1 t1 t2) (h2 : condition2 t2 t3) (h3 : condition3 t1 t2 t3) : t1 + t2 + t3 = 58.5 := 
by
  sorry

end skier_total_time_l2426_242660


namespace sequence_unique_integers_l2426_242648

theorem sequence_unique_integers (a : ℕ → ℤ) 
  (H_inf_pos : ∀ N : ℤ, ∃ n : ℕ, n > 0 ∧ a n > N) 
  (H_inf_neg : ∀ N : ℤ, ∃ n : ℕ, n > 0 ∧ a n < N)
  (H_diff_remainders : ∀ n : ℕ, n > 0 → ∀ i j : ℕ, (1 ≤ i ∧ i ≤ n) → (1 ≤ j ∧ j ≤ n) → i ≠ j → (a i % ↑n) ≠ (a j % ↑n)) :
  ∀ m : ℤ, ∃! n : ℕ, a n = m := sorry

end sequence_unique_integers_l2426_242648


namespace angle_between_lines_in_folded_rectangle_l2426_242634

theorem angle_between_lines_in_folded_rectangle
  (a b : ℝ) 
  (h : b > a)
  (dihedral_angle : ℝ)
  (h_dihedral_angle : dihedral_angle = 18) :
  ∃ (angle_AC_MN : ℝ), angle_AC_MN = 90 :=
by
  sorry

end angle_between_lines_in_folded_rectangle_l2426_242634


namespace cards_drawn_to_product_even_l2426_242679

theorem cards_drawn_to_product_even :
  ∃ n, (∀ (cards_drawn : Finset ℕ), 
    (cards_drawn ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}) ∧
    (cards_drawn.card = n) → 
    ¬ (∀ c ∈ cards_drawn, c % 2 = 1)
  ) ∧ n = 8 :=
by
  sorry

end cards_drawn_to_product_even_l2426_242679


namespace range_of_a_l2426_242617

theorem range_of_a (a : ℚ) (h₀ : 0 < a) (h₁ : ∃ n : ℕ, (2 * n - 1 = 2007) ∧ (-a < n ∧ n < a)) :
  1003 < a ∧ a ≤ 1004 :=
sorry

end range_of_a_l2426_242617


namespace problem1_problem2_l2426_242619

variable (m n x y : ℝ)

theorem problem1 : 4 * m * n^3 * (2 * m^2 - (3 / 4) * m * n^2) = 8 * m^3 * n^3 - 3 * m^2 * n^5 := sorry

theorem problem2 : (x - 6 * y^2) * (3 * x^3 + y) = 3 * x^4 + x * y - 18 * x^3 * y^2 - 6 * y^3 := sorry

end problem1_problem2_l2426_242619


namespace simplify_complex_expr_l2426_242607

theorem simplify_complex_expr : 
  ∀ (i : ℂ), i^2 = -1 → ( (2 + 4 * i) / (2 - 4 * i) - (2 - 4 * i) / (2 + 4 * i) )
  = -8/5 + (16/5 : ℂ) * i :=
by
  intro i h_i_squared
  sorry

end simplify_complex_expr_l2426_242607


namespace average_marks_l2426_242669

/-- Shekar scored 76, 65, 82, 67, and 85 marks in Mathematics, Science, Social Studies, English, and Biology respectively.
    We aim to prove that his average marks are 75. -/

def marks : List ℕ := [76, 65, 82, 67, 85]

theorem average_marks : (marks.sum / marks.length) = 75 := by
  sorry

end average_marks_l2426_242669


namespace binary_representation_253_l2426_242673

-- Define the decimal number
def decimal := 253

-- Define the number of zeros (x) and ones (y) in the binary representation of 253
def num_zeros := 1
def num_ones := 7

-- Prove that 2y - x = 13 given these conditions
theorem binary_representation_253 : (2 * num_ones - num_zeros) = 13 :=
by
  sorry

end binary_representation_253_l2426_242673


namespace abs_val_of_minus_two_and_half_l2426_242672

-- Definition of the absolute value function for real numbers
def abs_val (x : ℚ) : ℚ := if x < 0 then -x else x

-- Prove that the absolute value of -2.5 (which is -5/2) is equal to 2.5 (which is 5/2)
theorem abs_val_of_minus_two_and_half : abs_val (-5/2) = 5/2 := by
  sorry

end abs_val_of_minus_two_and_half_l2426_242672


namespace paul_bags_on_saturday_l2426_242611

-- Definitions and Conditions
def total_cans : ℕ := 72
def cans_per_bag : ℕ := 8
def extra_bags : ℕ := 3

-- Statement of the problem
theorem paul_bags_on_saturday (S : ℕ) :
  S * cans_per_bag = total_cans - (extra_bags * cans_per_bag) →
  S = 6 :=
sorry

end paul_bags_on_saturday_l2426_242611


namespace rotate_D_90_clockwise_l2426_242658

-- Define the point D with its coordinates.
structure Point where
  x : Int
  y : Int

-- Define the original point D.
def D : Point := { x := -3, y := -8 }

-- Define the rotation transformation.
def rotate90Clockwise (p : Point) : Point :=
  { x := p.y, y := -p.x }

-- Statement to be proven.
theorem rotate_D_90_clockwise :
  rotate90Clockwise D = { x := -8, y := 3 } :=
sorry

end rotate_D_90_clockwise_l2426_242658


namespace problem_statement_l2426_242615

theorem problem_statement (x : ℝ) (hx : x^2 + 1/(x^2) = 2) : x^4 + 1/(x^4) = 2 := by
  sorry

end problem_statement_l2426_242615


namespace grain_milling_necessary_pounds_l2426_242604

theorem grain_milling_necessary_pounds (x : ℝ) (h : 0.90 * x = 100) : x = 111 + 1 / 9 := 
by
  sorry

end grain_milling_necessary_pounds_l2426_242604


namespace prism_cubes_paint_condition_l2426_242696

theorem prism_cubes_paint_condition
  (m n r : ℕ)
  (h1 : m ≤ n)
  (h2 : n ≤ r)
  (h3 : (m - 2) * (n - 2) * (r - 2)
        - 2 * ((m - 2) * (n - 2) + (m - 2) * (r - 2) + (n - 2) * (r - 2)) 
        + 4 * (m - 2 + n - 2 + r - 2)
        = 1985) :
  (m = 5 ∧ n = 7 ∧ r = 663) ∨
  (m = 5 ∧ n = 5 ∧ r = 1981) ∨
  (m = 3 ∧ n = 3 ∧ r = 1981) ∨
  (m = 1 ∧ n = 7 ∧ r = 399) ∨
  (m = 1 ∧ n = 3 ∧ r = 1987) := 
sorry

end prism_cubes_paint_condition_l2426_242696


namespace ratio_y_to_x_l2426_242612

-- Definitions based on conditions
variable (c : ℝ) -- Cost price
def x : ℝ := 0.8 * c -- Selling price for a loss of 20%
def y : ℝ := 1.25 * c -- Selling price for a gain of 25%

-- Statement to prove the ratio of y to x
theorem ratio_y_to_x : y / x = 25 / 16 := by
  -- skip the proof
  sorry

end ratio_y_to_x_l2426_242612


namespace kelsey_remaining_half_speed_l2426_242651

variable (total_hours : ℝ) (first_half_speed : ℝ) (total_distance : ℝ) (remaining_half_time : ℝ) (remaining_half_distance : ℝ)

axiom h1 : total_hours = 10
axiom h2 : first_half_speed = 25
axiom h3 : total_distance = 400
axiom h4 : remaining_half_time = total_hours - total_distance / (2 * first_half_speed)
axiom h5 : remaining_half_distance = total_distance / 2

theorem kelsey_remaining_half_speed :
  remaining_half_distance / remaining_half_time = 100
:=
by
  sorry

end kelsey_remaining_half_speed_l2426_242651


namespace line_through_fixed_point_fixed_points_with_constant_slope_l2426_242601

-- Point structure definition
structure Point :=
(x : ℝ)
(y : ℝ)

-- Define curves C1 and C2
def curve_C1 (p : Point) : Prop :=
  p.x^2 + (p.y - 1/4)^2 = 1 ∧ p.y ≥ 1/4

def curve_C2 (p : Point) : Prop :=
  p.x^2 = 8 * p.y - 1 ∧ abs p.x ≥ 1

-- Line passing through fixed point for given perpendicularity condition
theorem line_through_fixed_point (A B M : Point) (l : ℝ → ℝ → Prop) :
  curve_C2 A → curve_C2 B →
  (∃ k b, ∀ x y, l x y ↔ y = k * x + b) →
  (M = ⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩) →
  ((M.x = A.x ∧ M.y = (A.y + B.y) / 2) → A.x * B.x = -16) →
  ∀ x y, l x y → y = (17 / 8) := sorry

-- Existence of two fixed points on y-axis with constant slope product
theorem fixed_points_with_constant_slope (P T1 T2 M : Point) (l : ℝ → ℝ → Prop) :
  curve_C1 P →
  (T1 = ⟨0, -1⟩) →
  (T2 = ⟨0, 1⟩) →
  l P.x P.y →
  (∃ k b, ∀ x y, l x y ↔ y = k * x + b) →
  (M.y^2 - (M.x^2 / 16) = 1) →
  (M.x ≠ 0) →
  ((M.y + 1) / M.x) * ((M.y - 1) / M.x) = (1 / 16) := sorry

end line_through_fixed_point_fixed_points_with_constant_slope_l2426_242601
