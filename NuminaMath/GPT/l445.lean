import Mathlib

namespace savings_account_amount_l445_44586

theorem savings_account_amount (stimulus : ℕ) (wife_ratio first_son_ratio wife_share first_son_share second_son_share : ℕ) : 
    stimulus = 2000 →
    wife_ratio = 2 / 5 →
    first_son_ratio = 2 / 5 →
    wife_share = wife_ratio * stimulus →
    first_son_share = first_son_ratio * (stimulus - wife_share) →
    second_son_share = 40 / 100 * (stimulus - wife_share - first_son_share) →
    (stimulus - wife_share - first_son_share - second_son_share) = 432 :=
by
  sorry

end savings_account_amount_l445_44586


namespace tina_pink_pens_l445_44527

def number_pink_pens (P G B : ℕ) : Prop :=
  G = P - 9 ∧
  B = P - 6 ∧
  P + G + B = 21

theorem tina_pink_pens :
  ∃ (P G B : ℕ), number_pink_pens P G B ∧ P = 12 :=
by
  sorry

end tina_pink_pens_l445_44527


namespace sample_size_correct_l445_44540

-- Definitions following the conditions in the problem
def total_products : Nat := 80
def sample_products : Nat := 10

-- Statement of the proof problem
theorem sample_size_correct : sample_products = 10 :=
by
  -- The proof is replaced with a placeholder sorry to skip the proof step
  sorry

end sample_size_correct_l445_44540


namespace shaded_region_area_l445_44526

open Real

noncomputable def area_of_shaded_region (r : ℝ) (s : ℝ) (d : ℝ) : ℝ := 
  (1/4) * π * r^2 + (1/2) * (d - s)^2

theorem shaded_region_area :
  let r := 3
  let s := 2
  let d := sqrt 5
  area_of_shaded_region r s d = 9 * π / 4 + (9 - 4 * sqrt 5) / 2 :=
by
  sorry

end shaded_region_area_l445_44526


namespace find_x_l445_44559

variable (x : ℝ)  -- Current distance Teena is behind Loe in miles
variable (t : ℝ) -- Time period in hours
variable (speed_teena : ℝ) -- Speed of Teena in miles per hour
variable (speed_loe : ℝ) -- Speed of Loe in miles per hour
variable (d_ahead : ℝ) -- Distance Teena will be ahead of Loe in 1.5 hours

axiom conditions : speed_teena = 55 ∧ speed_loe = 40 ∧ t = 1.5 ∧ d_ahead = 15

theorem find_x : (speed_teena * t - speed_loe * t = x + d_ahead) → x = 7.5 :=
by
  intro h
  sorry

end find_x_l445_44559


namespace hexagon_perimeter_is_42_l445_44542

-- Define the side length of the hexagon
def side_length : ℕ := 7

-- Define the number of sides of the hexagon
def num_sides : ℕ := 6

-- Define the perimeter of the hexagon
def hexagon_perimeter (side_length : ℕ) (num_sides : ℕ) : ℕ :=
  num_sides * side_length

-- The theorem to prove
theorem hexagon_perimeter_is_42 : hexagon_perimeter side_length num_sides = 42 :=
by
  sorry

end hexagon_perimeter_is_42_l445_44542


namespace elevator_height_after_20_seconds_l445_44569

-- Conditions
def starting_height : ℕ := 120
def descending_speed : ℕ := 4
def time_elapsed : ℕ := 20

-- Statement to prove
theorem elevator_height_after_20_seconds : 
  starting_height - descending_speed * time_elapsed = 40 := 
by 
  sorry

end elevator_height_after_20_seconds_l445_44569


namespace function_is_zero_l445_44515

theorem function_is_zero (f : ℝ → ℝ)
  (H : ∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) :
  ∀ x : ℝ, f x = 0 :=
by
  sorry

end function_is_zero_l445_44515


namespace verify_monomial_properties_l445_44567

def monomial : ℚ := -3/5 * (1:ℚ)^1 * (2:ℚ)^2

def coefficient (m : ℚ) : ℚ := -3/5  -- The coefficient of the monomial
def degree (m : ℚ) : ℕ := 3          -- The degree of the monomial

theorem verify_monomial_properties :
  coefficient monomial = -3/5 ∧ degree monomial = 3 :=
by
  sorry

end verify_monomial_properties_l445_44567


namespace candies_per_basket_l445_44563

noncomputable def chocolate_bars : ℕ := 5
noncomputable def mms : ℕ := 7 * chocolate_bars
noncomputable def marshmallows : ℕ := 6 * mms
noncomputable def total_candies : ℕ := chocolate_bars + mms + marshmallows
noncomputable def baskets : ℕ := 25

theorem candies_per_basket : total_candies / baskets = 10 :=
by
  sorry

end candies_per_basket_l445_44563


namespace find_number_of_10_bills_from_mother_l445_44581

variable (m10 : ℕ)  -- number of $10 bills given by Luke's mother

def mother_total : ℕ := 50 + 2*20 + 10*m10
def father_total : ℕ := 4*50 + 20 + 10
def total : ℕ := mother_total m10 + father_total

theorem find_number_of_10_bills_from_mother
  (fee : ℕ := 350)
  (m10 : ℕ) :
  total m10 = fee → m10 = 3 := 
by
  sorry

end find_number_of_10_bills_from_mother_l445_44581


namespace find_f_prime_one_l445_44593

theorem find_f_prime_one (f : ℝ → ℝ) (f' : ℝ → ℝ) (h : ∀ x, f' x = 2 * f' 1 + 1 / x) (h_fx : ∀ x, f x = 2 * x * f' 1 + Real.log x) : f' 1 = -1 := 
by 
  sorry

end find_f_prime_one_l445_44593


namespace heart_digit_proof_l445_44584

noncomputable def heart_digit : ℕ := 3

theorem heart_digit_proof (heartsuit : ℕ) (h : heartsuit * 9 + 6 = heartsuit * 10 + 3) : 
  heartsuit = heart_digit := 
by
  sorry

end heart_digit_proof_l445_44584


namespace equation_of_l3_line_l1_through_fixed_point_existence_of_T_l445_44579

-- Question 1: The equation of the line \( l_{3} \)
theorem equation_of_l3 
  (F : ℝ × ℝ) 
  (H_focus : F = (2, 0))
  (k : ℝ) 
  (H_slope : k = 1) : 
  (∀ x y : ℝ, y = k * x + -2 ↔ y = x - 2) := 
sorry

-- Question 2: Line \( l_{1} \) passes through the fixed point (8, 0)
theorem line_l1_through_fixed_point 
  (k m1 : ℝ)
  (H_km1 : k * m1 ≠ 0)
  (H_m1lt : m1 < -t)
  (H_condition : ∃ x y : ℝ, y = k * x + m1 ∧ x^2 + (8/k) * x + (8 * m1 / k) = 0 ∧ ((x, y) = A1 ∨ (x, y) = B1))
  (H_dot_product : (x1 - 0)*(x2 - 0) + (y1 - 0)*(y2 - 0) = 0) : 
  ∀ P : ℝ × ℝ, P = (8, 0) := 
sorry

-- Question 3: Existence of point T such that S_i and d_i form geometric sequences
theorem existence_of_T
  (k : ℝ)
  (H_k : k = 1)
  (m1 m2 m3 : ℝ)
  (H_m_ordered : m1 < m2 ∧ m2 < m3 ∧ m3 < -t)
  (t : ℝ)
  (S1 S2 S3 d1 d2 d3 : ℝ)
  (H_S_geom_seq : S2^2 = S1 * S3)
  (H_d_geom_seq : d2^2 = d1 * d3)
  : ∃ t : ℝ, t = -2 :=
sorry

end equation_of_l3_line_l1_through_fixed_point_existence_of_T_l445_44579


namespace number_of_teams_l445_44573

theorem number_of_teams (x : ℕ) (h : x * (x - 1) / 2 = 15) : x = 6 :=
sorry

end number_of_teams_l445_44573


namespace intersection_correct_l445_44562

def A : Set ℝ := { x | 0 < x ∧ x < 3 }
def B : Set ℝ := { x | x^2 ≥ 4 }
def intersection : Set ℝ := { x | 2 ≤ x ∧ x < 3 }

theorem intersection_correct : A ∩ B = intersection := by
  sorry

end intersection_correct_l445_44562


namespace calculation_result_l445_44520

theorem calculation_result : (4^2)^3 - 4 = 4092 :=
by
  sorry

end calculation_result_l445_44520


namespace inequality_abc_l445_44557

theorem inequality_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1) (h3 : 0 ≤ b) (h4 : b ≤ 1) (h5 : 0 ≤ c) (h6 : c ≤ 1) :
  (a / (b * c + 1)) + (b / (a * c + 1)) + (c / (a * b + 1)) ≤ 2 := by
  sorry

end inequality_abc_l445_44557


namespace num_six_digit_asc_digits_l445_44561

theorem num_six_digit_asc_digits : 
  ∃ n : ℕ, n = (Nat.choose 9 3) ∧ n = 84 := 
by
  sorry

end num_six_digit_asc_digits_l445_44561


namespace minimum_value_f_is_correct_l445_44501

noncomputable def f (x : ℝ) := 
  Real.sqrt (15 - 12 * Real.cos x) + 
  Real.sqrt (4 - 2 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (7 - 4 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (10 - 4 * Real.sqrt 3 * Real.sin x - 6 * Real.cos x)

theorem minimum_value_f_is_correct :
  ∃ x : ℝ, f x = (9 / 2) * Real.sqrt 2 :=
sorry

end minimum_value_f_is_correct_l445_44501


namespace ratio_of_costs_l445_44533

-- Definitions based on conditions
def quilt_length : Nat := 16
def quilt_width : Nat := 20
def patch_area : Nat := 4
def first_10_patch_cost : Nat := 10
def total_cost : Nat := 450

-- Theorem we need to prove
theorem ratio_of_costs : (total_cost - 10 * first_10_patch_cost) / (10 * first_10_patch_cost) = 7 / 2 := by
  sorry

end ratio_of_costs_l445_44533


namespace jessica_deposit_fraction_l445_44547

-- Definitions based on conditions
variable (initial_balance : ℝ)
variable (fraction_withdrawn : ℝ) (withdrawn_amount : ℝ)
variable (final_balance remaining_balance fraction_deposit : ℝ)

-- Conditions
def conditions := 
  fraction_withdrawn = 2 / 5 ∧
  withdrawn_amount = 400 ∧
  remaining_balance = initial_balance - withdrawn_amount ∧
  remaining_balance = initial_balance * (1 - fraction_withdrawn) ∧
  final_balance = 750 ∧
  final_balance = remaining_balance + fraction_deposit * remaining_balance

-- The proof problem
theorem jessica_deposit_fraction : 
  conditions initial_balance fraction_withdrawn withdrawn_amount final_balance remaining_balance fraction_deposit →
  fraction_deposit = 1 / 4 :=
by
  intro h
  sorry

end jessica_deposit_fraction_l445_44547


namespace geometric_first_term_l445_44506

theorem geometric_first_term (a r : ℝ) (h1 : a * r^3 = 720) (h2 : a * r^6 = 5040) : 
a = 720 / 7 :=
by
  sorry

end geometric_first_term_l445_44506


namespace marble_221_is_green_l445_44550

def marble_sequence_color (n : ℕ) : String :=
  let cycle_length := 15
  let red_count := 6
  let green_start := red_count + 1
  let green_end := red_count + 5
  let position := n % cycle_length
  if position ≠ 0 then
    let cycle_position := position
    if cycle_position <= red_count then "red"
    else if cycle_position <= green_end then "green"
    else "blue"
  else "blue"

theorem marble_221_is_green : marble_sequence_color 221 = "green" :=
by
  -- proof to be filled in
  sorry

end marble_221_is_green_l445_44550


namespace fraction_of_budget_is_31_percent_l445_44532

def coffee_pastry_cost (B : ℝ) (c : ℝ) (p : ℝ) :=
  c = 0.25 * (B - p) ∧ p = 0.10 * (B - c)

theorem fraction_of_budget_is_31_percent (B c p : ℝ) (h : coffee_pastry_cost B c p) :
  c + p = 0.31 * B :=
sorry

end fraction_of_budget_is_31_percent_l445_44532


namespace total_cost_of_roads_l445_44539

/-- A rectangular lawn with dimensions 150 m by 80 m with two roads running 
through the middle, one parallel to the length and one parallel to the breadth. 
The first road has a width of 12 m, a base cost of Rs. 4 per sq m, and an additional section 
through a hill costing 25% more for a section of length 60 m. The second road has a width 
of 8 m and a cost of Rs. 5 per sq m. Prove that the total cost for both roads is Rs. 14000. -/
theorem total_cost_of_roads :
  let lawn_length := 150
  let lawn_breadth := 80
  let road1_width := 12
  let road2_width := 8
  let road1_base_cost := 4
  let road1_hill_length := 60
  let road1_hill_cost := road1_base_cost + (road1_base_cost / 4)
  let road2_cost := 5
  let road1_length := lawn_length
  let road2_length := lawn_breadth

  let road1_area_non_hill := road1_length * road1_width
  let road1_area_hill := road1_hill_length * road1_width
  let road1_cost_non_hill := road1_area_non_hill * road1_base_cost
  let road1_cost_hill := road1_area_hill * road1_hill_cost

  let total_road1_cost := road1_cost_non_hill + road1_cost_hill

  let road2_area := road2_length * road2_width
  let road2_total_cost := road2_area * road2_cost

  let total_cost := total_road1_cost + road2_total_cost

  total_cost = 14000 := by sorry

end total_cost_of_roads_l445_44539


namespace fraction_addition_l445_44560

theorem fraction_addition : (1 + 3 + 5)/(2 + 4 + 6) + (2 + 4 + 6)/(1 + 3 + 5) = 25/12 := by
  sorry

end fraction_addition_l445_44560


namespace infinite_k_Q_ineq_l445_44518

def Q (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem infinite_k_Q_ineq :
  ∃ᶠ k in at_top, Q (3 ^ k) > Q (3 ^ (k + 1)) := sorry

end infinite_k_Q_ineq_l445_44518


namespace evaluate_expression_at_minus_half_l445_44548

noncomputable def complex_expression (x : ℚ) : ℚ :=
  (x - 3)^2 + (x + 3) * (x - 3) - 2 * x * (x - 2) + 1

theorem evaluate_expression_at_minus_half :
  complex_expression (-1 / 2) = 2 :=
by
  sorry

end evaluate_expression_at_minus_half_l445_44548


namespace compute_avg_interest_rate_l445_44576

variable (x : ℝ)

/-- The total amount of investment is $5000 - x at 3% and x at 7%. The incomes are equal 
thus we are asked to compute the average rate of interest -/
def avg_interest_rate : Prop :=
  let i_3 := 0.03 * (5000 - x)
  let i_7 := 0.07 * x
  i_3 = i_7 ∧
  (2 * i_3) / 5000 = 0.042

theorem compute_avg_interest_rate 
  (condition : ∃ x : ℝ, 0.03 * (5000 - x) = 0.07 * x) :
  avg_interest_rate x :=
by
  sorry

end compute_avg_interest_rate_l445_44576


namespace number_of_arrangements_l445_44556

theorem number_of_arrangements :
  ∃ (n k : ℕ), n = 10 ∧ k = 5 ∧ Nat.choose n k = 252 := by
  sorry

end number_of_arrangements_l445_44556


namespace find_f_log_l445_44531

noncomputable def f : ℝ → ℝ := sorry

-- Given Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x
axiom f_def : ∀ x : ℝ, 0 < x ∧ x < 1 → f x = 2^x - 2

-- Theorem to be proved
theorem find_f_log : f (Real.log 6 / Real.log (1/2)) = 1 / 2 :=
by
  sorry

end find_f_log_l445_44531


namespace range_of_a_l445_44591

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 ≥ 0

def neg_p : Prop := ∃ x : ℝ, a * x^2 + a * x + 1 < 0

theorem range_of_a (h : neg_p a) : a ∈ Set.Iio 0 ∪ Set.Ioi 4 :=
  sorry

end range_of_a_l445_44591


namespace quadratic_roots_l445_44517

theorem quadratic_roots (m : ℝ) (h_eq : ∃ α β : ℝ, (α + β = -4) ∧ (α * β = m) ∧ (|α - β| = 2)) : m = 5 :=
sorry

end quadratic_roots_l445_44517


namespace base8_to_base10_l445_44521

theorem base8_to_base10 :
  ∀ (n : ℕ), (n = 2 * 8^2 + 4 * 8^1 + 3 * 8^0) → (n = 163) :=
by
  intros n hn
  sorry

end base8_to_base10_l445_44521


namespace unwilted_roses_proof_l445_44530

-- Conditions
def initial_roses : Nat := 2 * 12
def traded_roses : Nat := 12
def first_day_roses (r: Nat) : Nat := r / 2
def second_day_roses (r: Nat) : Nat := r / 2

-- Initial number of roses
def total_roses : Nat := initial_roses + traded_roses

-- Number of unwilted roses after two days
def unwilted_roses : Nat := second_day_roses (first_day_roses total_roses)

-- Formal statement to prove
theorem unwilted_roses_proof : unwilted_roses = 9 := by
  sorry

end unwilted_roses_proof_l445_44530


namespace find_angle_l445_44599

-- Given the complement condition
def complement_condition (x : ℝ) : Prop :=
  x + 2 * (4 * x + 10) = 90

-- Proving the degree measure of the angle
theorem find_angle (x : ℝ) : complement_condition x → x = 70 / 9 := by
  intro hc
  sorry

end find_angle_l445_44599


namespace opposite_of_neg_three_fifths_l445_44552

theorem opposite_of_neg_three_fifths :
  -(-3 / 5) = 3 / 5 :=
by
  sorry

end opposite_of_neg_three_fifths_l445_44552


namespace find_a_for_square_binomial_l445_44546

theorem find_a_for_square_binomial (a r s : ℝ) 
  (h1 : ax^2 + 18 * x + 9 = (r * x + s)^2)
  (h2 : a = r^2)
  (h3 : 2 * r * s = 18)
  (h4 : s^2 = 9) : 
  a = 9 := 
by sorry

end find_a_for_square_binomial_l445_44546


namespace minimum_value_of_fraction_sum_l445_44571

open Real

theorem minimum_value_of_fraction_sum (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (h : a + b + c + d = 2) : 
    6 ≤ (1 / (a + b) + 1 / (a + c) + 1 / (a + d) + 1 / (b + c) + 1 / (b + d) + 1 / (c + d)) := by 
  sorry

end minimum_value_of_fraction_sum_l445_44571


namespace ElaCollected13Pounds_l445_44595

def KimberleyCollection : ℕ := 10
def HoustonCollection : ℕ := 12
def TotalCollection : ℕ := 35

def ElaCollection : ℕ := TotalCollection - KimberleyCollection - HoustonCollection

theorem ElaCollected13Pounds : ElaCollection = 13 := sorry

end ElaCollected13Pounds_l445_44595


namespace math_problem_l445_44582

theorem math_problem
  (n : ℕ) (d : ℕ)
  (h1 : d ≤ 9)
  (h2 : 3 * n^2 + 2 * n + d = 263)
  (h3 : 3 * n^2 + 2 * n + 4 = 253 + 6 * d) :
  n + d = 11 := 
sorry

end math_problem_l445_44582


namespace arith_prog_sum_eq_l445_44587

variable (a d : ℕ → ℤ)

def S (n : ℕ) : ℤ := (n / 2) * (2 * a 1 + (n - 1) * d 1)

theorem arith_prog_sum_eq (n : ℕ) : 
  S a d (n + 3) - 3 * S a d (n + 2) + 3 * S a d (n + 1) - S a d n = 0 := 
sorry

end arith_prog_sum_eq_l445_44587


namespace minimum_value_is_two_sqrt_two_l445_44508

noncomputable def minimum_value_expression (x : ℝ) : ℝ :=
  (Real.sqrt (x^2 + (2 - x)^2)) + (Real.sqrt ((2 - x)^2 + x^2))

theorem minimum_value_is_two_sqrt_two :
  ∃ x : ℝ, minimum_value_expression x = 2 * Real.sqrt 2 :=
by 
  sorry

end minimum_value_is_two_sqrt_two_l445_44508


namespace find_y_l445_44534

def diamond (a b : ℝ) : ℝ := a * b + 3 * b - a

theorem find_y : ∃ y : ℝ, diamond 4 y = 44 ∧ y = 48 / 7 :=
by
  sorry

end find_y_l445_44534


namespace area_of_common_region_l445_44513

noncomputable def common_area (length : ℝ) (width : ℝ) (radius : ℝ) : ℝ :=
  let pi := Real.pi
  let sector_area := (pi * radius^2 / 4) * 4
  let triangle_area := (1 / 2) * (width / 2) * (length / 2) * 4
  sector_area - triangle_area

theorem area_of_common_region :
  common_area 10 (Real.sqrt 18) 3 = 9 * (Real.pi) - 9 :=
by
  sorry

end area_of_common_region_l445_44513


namespace range_of_a_l445_44577

noncomputable def A : Set ℝ := { x : ℝ | x > 5 }
noncomputable def B (a : ℝ) : Set ℝ := { x : ℝ | x > a }

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a < 5 :=
  sorry

end range_of_a_l445_44577


namespace find_line_equation_l445_44585

def parameterized_line (t : ℝ) : ℝ × ℝ :=
  (3 * t + 2, 5 * t - 3)

theorem find_line_equation (x y : ℝ) (t : ℝ) (h : parameterized_line t = (x, y)) :
  y = (5 / 3) * x - 19 / 3 :=
sorry

end find_line_equation_l445_44585


namespace composite_quadratic_l445_44575

theorem composite_quadratic (a b : Int) (x1 x2 : Int)
  (h1 : x1 + x2 = -a)
  (h2 : x1 * x2 = b)
  (h3 : abs x1 > 2)
  (h4 : abs x2 > 2) :
  ∃ m n : Int, a + b + 1 = m * n ∧ m > 1 ∧ n > 1 :=
by
  sorry

end composite_quadratic_l445_44575


namespace total_distance_hiked_l445_44578

-- Defining the distances Terrell hiked on Saturday and Sunday
def distance_Saturday : Real := 8.2
def distance_Sunday : Real := 1.6

-- Stating the theorem to prove the total distance
theorem total_distance_hiked : distance_Saturday + distance_Sunday = 9.8 := by
  sorry

end total_distance_hiked_l445_44578


namespace twice_son_plus_father_is_70_l445_44583

section
variable {s f : ℕ}

-- Conditions
def son_age : ℕ := 15
def father_age : ℕ := 40

-- Statement to prove
theorem twice_son_plus_father_is_70 : (2 * son_age + father_age) = 70 :=
by
  sorry
end

end twice_son_plus_father_is_70_l445_44583


namespace max_students_l445_44503

theorem max_students 
  (x : ℕ) 
  (h_lt : x < 100)
  (h_mod8 : x % 8 = 5) 
  (h_mod5 : x % 5 = 3) 
  : x = 93 := 
sorry

end max_students_l445_44503


namespace unique_solution_7tuples_l445_44564

theorem unique_solution_7tuples : 
  ∃! (x : Fin 7 → ℝ), (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1/8 :=
sorry

end unique_solution_7tuples_l445_44564


namespace coefficient_x_squared_l445_44507

variable {a w c d : ℝ}

/-- The coefficient of x^2 in the expanded form of the equation (ax + w)(cx + d) = 6x^2 + x - 12 -/
theorem coefficient_x_squared (h1 : (a * x + w) * (c * x + d) = 6 * x^2 + x - 12)
                             (h2 : abs a + abs w + abs c + abs d = 12) :
  a * c = 6 :=
  sorry

end coefficient_x_squared_l445_44507


namespace exists_nat_concat_is_perfect_square_l445_44551

theorem exists_nat_concat_is_perfect_square :
  ∃ A : ℕ, ∃ n : ℕ, ∃ B : ℕ, (B * B = (10^n + 1) * A) :=
by sorry

end exists_nat_concat_is_perfect_square_l445_44551


namespace average_score_in_all_matches_l445_44596

theorem average_score_in_all_matches (runs_match1_match2 : ℤ) (runs_other_matches : ℤ) (total_matches : ℤ) 
  (average1 : ℤ) (average2 : ℤ)
  (h1 : average1 = 40) (h2 : average2 = 10) (h3 : runs_match1_match2 = 2 * average1)
  (h4 : runs_other_matches = 3 * average2) (h5 : total_matches = 5) :
  ((runs_match1_match2 + runs_other_matches) / total_matches) = 22 := 
by
  sorry

end average_score_in_all_matches_l445_44596


namespace quadratic_roots_difference_square_l445_44512

theorem quadratic_roots_difference_square (a b : ℝ) (h : 2 * a^2 - 8 * a + 6 = 0 ∧ 2 * b^2 - 8 * b + 6 = 0) :
  (a - b) ^ 2 = 4 :=
sorry

end quadratic_roots_difference_square_l445_44512


namespace solve_for_nabla_l445_44555

theorem solve_for_nabla (nabla : ℤ) (h : 3 * (-2) = nabla + 2) : nabla = -8 := 
by
  sorry

end solve_for_nabla_l445_44555


namespace pirate_coins_l445_44570

theorem pirate_coins (x : ℕ) (hn : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 15 → ∃ y : ℕ, y = (2 * k * x) / 15) : 
  ∃ y : ℕ, y = 630630 :=
by sorry

end pirate_coins_l445_44570


namespace determine_height_impossible_l445_44525

-- Definitions used in the conditions
def shadow_length_same (xiao_ming_height xiao_qiang_height xiao_ming_distance xiao_qiang_distance : ℝ) : Prop :=
  xiao_ming_height / xiao_ming_distance = xiao_qiang_height / xiao_qiang_distance

-- The proof problem: given that the shadow lengths are the same under the same street lamp,
-- prove that it is impossible to determine who is taller.
theorem determine_height_impossible (xiao_ming_height xiao_qiang_height xiao_ming_distance xiao_qiang_distance : ℝ) :
  shadow_length_same xiao_ming_height xiao_qiang_height xiao_ming_distance xiao_qiang_distance →
  ¬ (xiao_ming_height ≠ xiao_qiang_height ↔ true) :=
by
  intro h
  sorry -- Proof not required as per instructions

end determine_height_impossible_l445_44525


namespace farmer_total_acres_l445_44543

/--
A farmer used some acres of land for beans, wheat, and corn in the ratio of 5 : 2 : 4, respectively.
There were 376 acres used for corn. Prove that the farmer used a total of 1034 acres of land.
-/
theorem farmer_total_acres (x : ℕ) (h : 4 * x = 376) :
    5 * x + 2 * x + 4 * x = 1034 :=
by
  sorry

end farmer_total_acres_l445_44543


namespace solution_A_to_B_ratio_l445_44574

def ratio_solution_A_to_B (V_A V_B : ℝ) : Prop :=
  (21 / 25) * V_A + (2 / 5) * V_B = (3 / 5) * (V_A + V_B) → V_A / V_B = 5 / 6

theorem solution_A_to_B_ratio (V_A V_B : ℝ) (h : (21 / 25) * V_A + (2 / 5) * V_B = (3 / 5) * (V_A + V_B)) :
  V_A / V_B = 5 / 6 :=
sorry

end solution_A_to_B_ratio_l445_44574


namespace factorization_of_difference_of_squares_l445_44588

theorem factorization_of_difference_of_squares (x : ℝ) : 
  x^2 - 1 = (x + 1) * (x - 1) :=
sorry

end factorization_of_difference_of_squares_l445_44588


namespace mass_of_1m3_l445_44544

/-- The volume of 1 gram of the substance in cubic centimeters cms_per_gram is 1.3333333333333335 cm³. -/
def cms_per_gram : ℝ := 1.3333333333333335

/-- There are 1,000,000 cubic centimeters in 1 cubic meter. -/
def cm3_per_m3 : ℕ := 1000000

/-- Given the volume of 1 gram of the substance, find the mass of 1 cubic meter of the substance. -/
theorem mass_of_1m3 (h1 : cms_per_gram = 1.3333333333333335) (h2 : cm3_per_m3 = 1000000) :
  ∃ m : ℝ, m = 750 :=
by
  sorry

end mass_of_1m3_l445_44544


namespace solution_system_eq_l445_44565

theorem solution_system_eq (x y : ℝ) :
  (x^2 * y + x * y^2 - 2 * x - 2 * y + 10 = 0) ∧
  (x^3 * y - x * y^3 - 2 * x^2 + 2 * y^2 - 30 = 0) ↔ 
  (x = -4 ∧ y = -1) :=
by sorry

end solution_system_eq_l445_44565


namespace value_of_y_l445_44502

theorem value_of_y (x y : ℝ) (h1 : x + y = 5) (h2 : x = 3) : y = 2 :=
by
  sorry

end value_of_y_l445_44502


namespace sin_squared_sum_eq_one_l445_44537

theorem sin_squared_sum_eq_one (α β γ : ℝ) 
  (h₁ : 0 ≤ α ∧ α ≤ π/2) 
  (h₂ : 0 ≤ β ∧ β ≤ π/2) 
  (h₃ : 0 ≤ γ ∧ γ ≤ π/2) 
  (h₄ : Real.sin α + Real.sin β + Real.sin γ = 1)
  (h₅ : Real.sin α * Real.cos (2 * α) + Real.sin β * Real.cos (2 * β) + Real.sin γ * Real.cos (2 * γ) = -1) :
  Real.sin α ^ 2 + Real.sin β ^ 2 + Real.sin γ ^ 2 = 1 := 
sorry

end sin_squared_sum_eq_one_l445_44537


namespace general_formula_for_nth_term_exists_c_makes_bn_arithmetic_l445_44545

-- Definitions and conditions
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Noncomputable sum of the first n terms of an arithmetic sequence
noncomputable def arithmetic_sum (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 0 + a (n - 1))) / 2

variables {a : ℕ → ℤ} {S : ℕ → ℤ} {d : ℤ}
variable (c : ℤ)

axiom h1 : is_arithmetic_sequence a d
axiom h2 : d > 0
axiom h3 : a 1 * a 2 = 45
axiom h4 : a 0 + a 4 = 18

-- General formula for the nth term
theorem general_formula_for_nth_term :
  ∃ a1 d, a 0 = a1 ∧ d > 0 ∧ (∀ n, a n = a1 + n * d) :=
sorry

-- Arithmetic sequence from Sn/(n+c)
theorem exists_c_makes_bn_arithmetic :
  ∃ (c : ℤ), c ≠ 0 ∧ (∀ n, n > 0 → (arithmetic_sum a n) / (n + c) - (arithmetic_sum a (n - 1)) / (n - 1 + c) = d) :=
sorry

end general_formula_for_nth_term_exists_c_makes_bn_arithmetic_l445_44545


namespace simplify_fraction_l445_44505

theorem simplify_fraction : (180 : ℚ) / 1260 = 1 / 7 :=
by
  sorry

end simplify_fraction_l445_44505


namespace eq_970299_l445_44535

theorem eq_970299 : 98^3 + 3 * 98^2 + 3 * 98 + 1 = 970299 :=
by
  sorry

end eq_970299_l445_44535


namespace machine_A_sprockets_per_hour_l445_44580

theorem machine_A_sprockets_per_hour 
  (A T_Q : ℝ)
  (h1 : 550 = 1.1 * A * T_Q)
  (h2 : 550 = A * (T_Q + 10)) 
  : A = 5 :=
by
  sorry

end machine_A_sprockets_per_hour_l445_44580


namespace dot_product_a_b_l445_44524

def vector_a : ℝ × ℝ := (1, 1)
def vector_b : ℝ × ℝ := (-1, 2)

theorem dot_product_a_b : vector_a.1 * vector_b.1 + vector_a.2 * vector_b.2 = 1 := by
  sorry

end dot_product_a_b_l445_44524


namespace find_a_b_l445_44590

def satisfies_digit_conditions (n a b : ℕ) : Prop :=
  n = 2000 + 100 * a + 90 + b ∧
  n / 1000 % 10 = 2 ∧
  n / 100 % 10 = a ∧
  n / 10 % 10 = 9 ∧
  n % 10 = b

theorem find_a_b : ∃ (a b : ℕ), 2^a * 9^b = 2000 + 100*a + 90 + b ∧ satisfies_digit_conditions (2^a * 9^b) a b :=
by
  sorry

end find_a_b_l445_44590


namespace parcel_post_cost_l445_44529

def indicator (P : ℕ) : ℕ := if P >= 5 then 1 else 0

theorem parcel_post_cost (P : ℕ) : 
  P ≥ 0 →
  (C : ℕ) = 15 + 5 * (P - 1) - 8 * indicator P :=
sorry

end parcel_post_cost_l445_44529


namespace conditions_for_star_commute_l445_44553

-- Define the operation star
def star (a b : ℝ) : ℝ := a^3 * b^2 - a * b^3

-- Theorem stating the equivalence
theorem conditions_for_star_commute :
  ∀ (x y : ℝ), (star x y = star y x) ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) :=
sorry

end conditions_for_star_commute_l445_44553


namespace length_of_train_l445_44554

-- We state the problem as a theorem in Lean
theorem length_of_train (bridge_length : ℝ) (crossing_time : ℝ) (train_speed_kmh : ℝ)
  (h_bridge_length : bridge_length = 150)
  (h_crossing_time : crossing_time = 32)
  (h_train_speed_kmh : train_speed_kmh = 45) :
  ∃ (train_length : ℝ), train_length = 250 := 
by
  -- We assume the necessary conditions as given
  have train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  have total_distance : ℝ := train_speed_ms * crossing_time
  have train_length : ℝ := total_distance - bridge_length
  -- Conclude the length of the train is 250
  use train_length
  -- The proof steps are skipped using 'sorry'
  sorry

end length_of_train_l445_44554


namespace one_third_eleven_y_plus_three_l445_44568

theorem one_third_eleven_y_plus_three (y : ℝ) : 
  (1/3) * (11 * y + 3) = 11 * y / 3 + 1 :=
by
  sorry

end one_third_eleven_y_plus_three_l445_44568


namespace find_number_l445_44558

theorem find_number (x : ℝ) : x + 5 * 12 / (180 / 3) = 61 ↔ x = 60 := by
  sorry

end find_number_l445_44558


namespace Jane_Hector_meet_point_C_l445_44589

theorem Jane_Hector_meet_point_C (s t : ℝ) (h_start : ℝ) (j_start : ℝ) (loop_length : ℝ) 
  (h_speed : ℝ) (j_speed : ℝ) (h_dest : ℝ) (j_dest : ℝ)
  (h_speed_eq : h_speed = s) (j_speed_eq : j_speed = 3 * s) (loop_len_eq : loop_length = 30)
  (start_point_eq : h_start = 0 ∧ j_start = 0)
  (opposite_directions : h_dest + j_dest = loop_length)
  (meet_time_eq : t = 15 / (2 * s)) :
  h_dest = 7.5 ∧ j_dest = 22.5 → (h_dest = 7.5 ∧ j_dest = 22.5) :=
by
  sorry

end Jane_Hector_meet_point_C_l445_44589


namespace slope_of_chord_l445_44592

theorem slope_of_chord (x y : ℝ) (h : (x^2 / 16) + (y^2 / 9) = 1) (h_midpoint : (x₁ + x₂ = 2) ∧ (y₁ + y₂ = 4)) :
  ∃ k : ℝ, k = -9 / 32 :=
by
  sorry

end slope_of_chord_l445_44592


namespace taxi_fare_charge_l445_44538

theorem taxi_fare_charge :
  let initial_fee := 2.25
  let total_distance := 3.6
  let total_charge := 4.95
  let increments := total_distance / (2 / 5)
  let distance_charge := total_charge - initial_fee
  let charge_per_increment := distance_charge / increments
  charge_per_increment = 0.30 :=
by
  sorry

end taxi_fare_charge_l445_44538


namespace exact_one_solves_l445_44509

variables (p1 p2 : ℝ)

/-- The probability that exactly one of two persons solves the problem
    when their respective probabilities are p1 and p2. -/
theorem exact_one_solves (h1 : 0 ≤ p1) (h2 : p1 ≤ 1) (h3 : 0 ≤ p2) (h4 : p2 ≤ 1) :
  (p1 * (1 - p2) + p2 * (1 - p1)) = (p1 + p2 - 2 * p1 * p2) := 
sorry

end exact_one_solves_l445_44509


namespace factored_quadratic_even_b_l445_44536

theorem factored_quadratic_even_b
  (c d e f y : ℤ)
  (h1 : c * e = 45)
  (h2 : d * f = 45) 
  (h3 : ∃ b, 45 * y^2 + b * y + 45 = (c * y + d) * (e * y + f)) :
  ∃ b, (45 * y^2 + b * y + 45 = (c * y + d) * (e * y + f)) ∧ (b % 2 = 0) :=
by
  sorry

end factored_quadratic_even_b_l445_44536


namespace molecular_weight_of_compound_l445_44566

theorem molecular_weight_of_compound (C H O n : ℕ) 
    (atomic_weight_C : ℝ) (atomic_weight_H : ℝ) (atomic_weight_O : ℝ) 
    (total_weight : ℝ) 
    (h_C : C = 2) (h_H : H = 4) 
    (h_atomic_weight_C : atomic_weight_C = 12.01) 
    (h_atomic_weight_H : atomic_weight_H = 1.008) 
    (h_atomic_weight_O : atomic_weight_O = 16.00) 
    (h_total_weight : total_weight = 60) : 
    C * atomic_weight_C + H * atomic_weight_H + n * atomic_weight_O = total_weight → 
    n = 2 := 
sorry

end molecular_weight_of_compound_l445_44566


namespace total_winter_clothing_l445_44504

theorem total_winter_clothing (boxes : ℕ) (scarves_per_box mittens_per_box : ℕ) (h_boxes : boxes = 8) (h_scarves : scarves_per_box = 4) (h_mittens : mittens_per_box = 6) : 
  boxes * (scarves_per_box + mittens_per_box) = 80 := 
by
  sorry

end total_winter_clothing_l445_44504


namespace height_of_box_l445_44598

def base_area : ℕ := 20 * 20
def cost_per_box : ℝ := 1.30
def total_volume : ℕ := 3060000
def amount_spent : ℝ := 663

theorem height_of_box : ∃ h : ℕ, 400 * h = total_volume / (amount_spent / cost_per_box) := sorry

end height_of_box_l445_44598


namespace product_of_fractions_l445_44500

theorem product_of_fractions :
  (1 / 3) * (3 / 5) * (5 / 7) = 1 / 7 :=
  sorry

end product_of_fractions_l445_44500


namespace count_integers_between_cubes_l445_44516

theorem count_integers_between_cubes (a b : ℝ) (h1 : a = 10.5) (h2 : b = 10.6) : 
  let lower_bound := a^3
  let upper_bound := b^3
  let first_integer := Int.ceil lower_bound
  let last_integer := Int.floor upper_bound
  (last_integer - first_integer + 1) = 33 :=
by
  -- Definitions for clarity
  let lower_bound := a^3
  let upper_bound := b^3
  let first_integer := Int.ceil lower_bound
  let last_integer := Int.floor upper_bound
  
  -- Skipping the proof
  sorry

end count_integers_between_cubes_l445_44516


namespace total_distance_traveled_l445_44523

def distance_from_earth_to_planet_x : ℝ := 0.5
def distance_from_planet_x_to_planet_y : ℝ := 0.1
def distance_from_planet_y_to_earth : ℝ := 0.1

theorem total_distance_traveled : 
  distance_from_earth_to_planet_x + distance_from_planet_x_to_planet_y + distance_from_planet_y_to_earth = 0.7 :=
by
  sorry

end total_distance_traveled_l445_44523


namespace divides_two_pow_n_minus_one_l445_44597

theorem divides_two_pow_n_minus_one {n : ℕ} (h : n > 0) (divides : n ∣ 2^n - 1) : n = 1 :=
sorry

end divides_two_pow_n_minus_one_l445_44597


namespace pipe_A_fill_time_l445_44514

theorem pipe_A_fill_time 
  (t : ℝ)
  (ht : (1 / t - 1 / 6) = 4 / 15.000000000000005) : 
  t = 30 / 13 :=  
sorry

end pipe_A_fill_time_l445_44514


namespace determine_f_5_l445_44549

theorem determine_f_5 (f : ℝ → ℝ) (h1 : f 1 = 3) 
  (h2 : ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x + f y)) : f 5 = 45 :=
sorry

end determine_f_5_l445_44549


namespace ABD_collinear_l445_44519

noncomputable def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (p2.1 - p1.1) * k = p3.1 - p1.1 ∧ (p2.2 - p1.2) * k = p3.2 - p1.2

noncomputable def vector (x y : ℝ) : ℝ × ℝ := (x, y)

variables {a b : ℝ × ℝ}
variables {A B C D : ℝ × ℝ}

axiom a_ne_zero : a ≠ (0, 0)
axiom b_ne_zero : b ≠ (0, 0)
axiom a_b_not_collinear : ∀ k : ℝ, a ≠ k • b
axiom AB_def : B = (A.1 + a.1 + b.1, A.2 + a.2 + b.2)
axiom BC_def : C = (B.1 + a.1 + 10 * b.1, B.2 + a.2 + 10 * b.2)
axiom CD_def : D = (C.1 + 3 * (a.1 - 2 * b.1), C.2 + 3 * (a.2 - 2 * b.2))

theorem ABD_collinear : collinear A B D :=
by
  sorry

end ABD_collinear_l445_44519


namespace problem_statement_l445_44511

theorem problem_statement (a b c : ℝ) (h₀ : 4 * a - 4 * b + c > 0) (h₁ : a + 2 * b + c < 0) : b^2 > a * c :=
sorry

end problem_statement_l445_44511


namespace sum_of_integers_l445_44541

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 120) : x + y = 15 :=
by {
    sorry
}

end sum_of_integers_l445_44541


namespace optionB_is_a9_l445_44522

-- Definitions of the expressions
def optionA (a : ℤ) : ℤ := a^3 + a^6
def optionB (a : ℤ) : ℤ := a^3 * a^6
def optionC (a : ℤ) : ℤ := a^10 - a
def optionD (a α : ℤ) : ℤ := α^12 / a^2

-- Theorem stating which option equals a^9
theorem optionB_is_a9 (a α : ℤ) : optionA a ≠ a^9 ∧ optionB a = a^9 ∧ optionC a ≠ a^9 ∧ optionD a α ≠ a^9 :=
by
  sorry

end optionB_is_a9_l445_44522


namespace a_2016_mod_2017_l445_44572

-- Defining the sequence
def seq (a : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧
  a 1 = 2 ∧
  ∀ n, a (n + 2) = 2 * a (n + 1) + 41 * a n

theorem a_2016_mod_2017 (a : ℕ → ℕ) (h : seq a) : 
  a 2016 % 2017 = 0 := 
sorry

end a_2016_mod_2017_l445_44572


namespace check_basis_l445_44594

structure Vector2D :=
  (x : ℤ)
  (y : ℤ)

def are_collinear (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.y - v2.x * v1.y = 0

def can_be_basis (v1 v2 : Vector2D) : Prop :=
  ¬ are_collinear v1 v2

theorem check_basis :
  can_be_basis ⟨-1, 2⟩ ⟨5, 7⟩ ∧
  ¬ can_be_basis ⟨0, 0⟩ ⟨1, -2⟩ ∧
  ¬ can_be_basis ⟨3, 5⟩ ⟨6, 10⟩ ∧
  ¬ can_be_basis ⟨2, -3⟩ ⟨(1 : ℤ)/2, -(3 : ℤ)/4⟩ :=
by
  sorry

end check_basis_l445_44594


namespace solve_equation_l445_44528

theorem solve_equation :
  (∀ x : ℝ, x ≠ 2 / 3 → (7 * x + 3) / (3 * x ^ 2 + 7 * x - 6) = (3 * x) / (3 * x - 2) → (x = 1 / 3) ∨ (x = -3)) :=
by
  sorry

end solve_equation_l445_44528


namespace tangent_line_equation_at_x_1_intervals_of_monotonic_increase_l445_44510

noncomputable def f (x : ℝ) := x^3 - x + 3
noncomputable def df (x : ℝ) := 3 * x^2 - 1

theorem tangent_line_equation_at_x_1 : 
  let k := df 1
  let y := f 1
  (2 = k) ∧ (y = 3) ∧ ∀ x y, y - 3 = 2 * (x - 1) ↔ 2 * x - y + 1 = 0 := 
by 
  sorry

theorem intervals_of_monotonic_increase : 
  let x1 := - (Real.sqrt 3) / 3
  let x2 := (Real.sqrt 3) / 3
  ∀ x, (df x > 0 ↔ (x < x1) ∨ (x > x2)) ∧ 
       (df x < 0 ↔ (x1 < x ∧ x < x2)) := 
by 
  sorry

end tangent_line_equation_at_x_1_intervals_of_monotonic_increase_l445_44510
