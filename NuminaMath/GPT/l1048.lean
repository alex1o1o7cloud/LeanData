import Mathlib

namespace cylinder_height_l1048_104829

theorem cylinder_height (r h : ℝ) (SA : ℝ) (h₀ : r = 3) (h₁ : SA = 36 * Real.pi) (h₂ : SA = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) : h = 3 :=
by
  -- The proof will be constructed here
  sorry

end cylinder_height_l1048_104829


namespace james_january_income_l1048_104820

variable (January February March : ℝ)
variable (h1 : February = 2 * January)
variable (h2 : March = February - 2000)
variable (h3 : January + February + March = 18000)

theorem james_january_income : January = 4000 := by
  sorry

end james_january_income_l1048_104820


namespace quadrilateral_perimeter_proof_l1048_104888

noncomputable def perimeter_quadrilateral (AB BC CD AD : ℝ) : ℝ :=
  AB + BC + CD + AD

theorem quadrilateral_perimeter_proof
  (AB BC CD AD : ℝ)
  (h1 : AB = 15)
  (h2 : BC = 10)
  (h3 : CD = 6)
  (h4 : AB = AD)
  (h5 : AD = Real.sqrt 181)
  : perimeter_quadrilateral AB BC CD AD = 31 + Real.sqrt 181 := by
  unfold perimeter_quadrilateral
  rw [h1, h2, h3, h5]
  sorry

end quadrilateral_perimeter_proof_l1048_104888


namespace multiple_of_distance_l1048_104857

namespace WalkProof

variable (H R M : ℕ)

/-- Rajesh walked 10 kilometers less than a certain multiple of the distance that Hiro walked. 
    Together they walked 25 kilometers. Rajesh walked 18 kilometers. 
    Prove that the multiple of the distance Hiro walked that Rajesh walked less than is 4. -/
theorem multiple_of_distance (h1 : R = M * H - 10) 
                             (h2 : H + R = 25)
                             (h3 : R = 18) :
                             M = 4 :=
by
  sorry

end WalkProof

end multiple_of_distance_l1048_104857


namespace consecutive_sunny_days_l1048_104845

theorem consecutive_sunny_days (n_sunny_days : ℕ) (n_days_year : ℕ) (days_to_stay : ℕ) (condition1 : n_sunny_days = 350) (condition2 : n_days_year = 365) :
  days_to_stay = 32 :=
by
  sorry

end consecutive_sunny_days_l1048_104845


namespace probability_is_8point64_percent_l1048_104844

/-- Define the probabilities based on given conditions -/
def p_excel : ℝ := 0.45
def p_night_shift_given_excel : ℝ := 0.32
def p_no_weekend_given_night_shift : ℝ := 0.60

/-- Calculate the combined probability -/
def combined_probability :=
  p_excel * p_night_shift_given_excel * p_no_weekend_given_night_shift

theorem probability_is_8point64_percent :
  combined_probability = 0.0864 :=
by
  -- We will skip the proof for now
  sorry

end probability_is_8point64_percent_l1048_104844


namespace intersection_of_A_and_B_l1048_104843

def A : Set ℤ := {1, 2, -3}
def B : Set ℤ := {1, -4, 5}

theorem intersection_of_A_and_B : A ∩ B = {1} :=
by sorry

end intersection_of_A_and_B_l1048_104843


namespace range_of_a_l1048_104883

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + (a-1)*x + 1 ≤ 0) → (-1 < a ∧ a < 3) :=
by
  sorry

end range_of_a_l1048_104883


namespace solve_abs_eq_l1048_104825

theorem solve_abs_eq (x : ℝ) (h : |x - 1| = 2 * x) : x = 1 / 3 :=
by
  sorry

end solve_abs_eq_l1048_104825


namespace necessary_and_sufficient_condition_l1048_104830

theorem necessary_and_sufficient_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (|a + b| = |a| + |b|) ↔ (a * b > 0) :=
sorry

end necessary_and_sufficient_condition_l1048_104830


namespace age_of_new_teacher_l1048_104832

theorem age_of_new_teacher (sum_of_20_teachers : ℕ)
  (avg_age_20_teachers : ℕ)
  (total_teachers_after_new_teacher : ℕ)
  (new_avg_age_after_new_teacher : ℕ)
  (h1 : sum_of_20_teachers = 20 * 49)
  (h2 : avg_age_20_teachers = 49)
  (h3 : total_teachers_after_new_teacher = 21)
  (h4 : new_avg_age_after_new_teacher = 48) :
  ∃ (x : ℕ), x = 28 :=
by
  sorry

end age_of_new_teacher_l1048_104832


namespace sum_of_first_twelve_multiples_of_18_l1048_104852

-- Given conditions
def sum_of_first_n_positives (n : ℕ) : ℕ := n * (n + 1) / 2

def first_twelve_multiples_sum (k : ℕ) : ℕ := k * (sum_of_first_n_positives 12)

-- The question to prove
theorem sum_of_first_twelve_multiples_of_18 : first_twelve_multiples_sum 18 = 1404 :=
by
  sorry

end sum_of_first_twelve_multiples_of_18_l1048_104852


namespace find_angle_C_l1048_104823

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (triangle_ABC : Type)

-- Given conditions
axiom ten_a_cos_B_eq_three_b_cos_A : 10 * a * Real.cos B = 3 * b * Real.cos A
axiom cos_A_value : Real.cos A = 5 * Real.sqrt 26 / 26

-- Required to prove
theorem find_angle_C : C = 3 * Real.pi / 4 := by
  sorry

end find_angle_C_l1048_104823


namespace four_integers_product_sum_l1048_104860

theorem four_integers_product_sum (a b c d : ℕ) (h1 : a * b * c * d = 2002) (h2 : a + b + c + d < 40) :
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) ∨
  (a = 2 ∧ b = 7 ∧ c = 11 ∧ d = 13) ∨ (a = 1 ∧ b = 14 ∧ c = 11 ∧ d = 13) :=
sorry

end four_integers_product_sum_l1048_104860


namespace geometric_sequence_sum_l1048_104802

theorem geometric_sequence_sum (q a₁ : ℝ) (hq : q > 1) (h₁ : a₁ + a₁ * q^3 = 18) (h₂ : a₁^2 * q^3 = 32) :
  (a₁ * (1 - q^8) / (1 - q) = 510) :=
by
  sorry

end geometric_sequence_sum_l1048_104802


namespace choose_president_and_vice_president_l1048_104868

theorem choose_president_and_vice_president :
  let total_members := 24
  let boys := 8
  let girls := 16
  let senior_members := 4
  let senior_boys := 2
  let senior_girls := 2
  let president_choices := senior_members
  let vice_president_choices_boy_pres := girls
  let vice_president_choices_girl_pres := boys - senior_boys
  let total_ways :=
    (senior_boys * vice_president_choices_boy_pres) + 
    (senior_girls * vice_president_choices_girl_pres)
  total_ways = 44 := 
by
  sorry

end choose_president_and_vice_president_l1048_104868


namespace simplify_expression_l1048_104856

theorem simplify_expression (x : ℝ) : 3 * (5 - 2 * x) - 2 * (4 + 3 * x) = 7 - 12 * x := by
  sorry

end simplify_expression_l1048_104856


namespace circle_numbers_exist_l1048_104876

theorem circle_numbers_exist :
  ∃ (a b c d e f : ℚ),
    a = 2 ∧
    b = 3 ∧
    c = 3 / 2 ∧
    d = 1 / 2 ∧
    e = 1 / 3 ∧
    f = 2 / 3 ∧
    a = b * f ∧
    b = a * c ∧
    c = b * d ∧
    d = c * e ∧
    e = d * f ∧
    f = e * a := by
  sorry

end circle_numbers_exist_l1048_104876


namespace volume_of_soil_extracted_l1048_104878

-- Definition of the conditions
def Length : ℝ := 20
def Width : ℝ := 10
def Depth : ℝ := 8

-- Statement of the proof problem
theorem volume_of_soil_extracted : Length * Width * Depth = 1600 := by
  -- Proof skipped
  sorry

end volume_of_soil_extracted_l1048_104878


namespace isosceles_triangles_l1048_104854

noncomputable def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem isosceles_triangles (a b c : ℝ) (h : a ≥ b ∧ b ≥ c ∧ c > 0)
    (H : ∀ n : ℕ, a ^ n + b ^ n > c ^ n ∧ b ^ n + c ^ n > a ^ n ∧ c ^ n + a ^ n > b ^ n) :
    is_isosceles_triangle a b c :=
  sorry

end isosceles_triangles_l1048_104854


namespace quadratic_equation_unique_solution_l1048_104818

theorem quadratic_equation_unique_solution 
  (a c : ℝ) (h1 : ∃ x : ℝ, a * x^2 + 8 * x + c = 0)
  (h2 : a + c = 10)
  (h3 : a < c) :
  (a, c) = (2, 8) := 
sorry

end quadratic_equation_unique_solution_l1048_104818


namespace part1_part2_l1048_104865

open Complex

-- Define the first proposition p
def p (m : ℝ) : Prop :=
  (m - 1 < 0) ∧ (m + 3 > 0)

-- Define the second proposition q
def q (m : ℝ) : Prop :=
  abs (Complex.mk 1 (m - 2)) ≤ Real.sqrt 10

-- Prove the first part of the problem
theorem part1 (m : ℝ) (hp : p m) : -3 < m ∧ m < 1 :=
sorry

-- Prove the second part of the problem
theorem part2 (m : ℝ) (h : ¬ (p m ∧ q m) ∧ (p m ∨ q m)) : (-3 < m ∧ m < -1) ∨ (1 ≤ m ∧ m ≤ 5) :=
sorry

end part1_part2_l1048_104865


namespace flour_per_new_bread_roll_l1048_104801

theorem flour_per_new_bread_roll (p1 f1 p2 f2 c : ℚ)
  (h1 : p1 = 40)
  (h2 : f1 = 1 / 8)
  (h3 : p2 = 25)
  (h4 : c = p1 * f1)
  (h5 : c = p2 * f2) :
  f2 = 1 / 5 :=
by
  sorry

end flour_per_new_bread_roll_l1048_104801


namespace catch_up_time_l1048_104842

def A_departure_time : ℕ := 8 * 60 -- in minutes
def B_departure_time : ℕ := 6 * 60 -- in minutes
def relative_speed (v : ℕ) : ℕ := 5 * v / 4 -- (2.5v effective) converted to integer math
def initial_distance (v : ℕ) : ℕ := 2 * v * 2 -- 4v distance (B's 2 hours lead)

theorem catch_up_time (v : ℕ) :  A_departure_time + ((initial_distance v * 4) / (relative_speed v - v)) = 1080 :=
by
  sorry

end catch_up_time_l1048_104842


namespace solve_equation_real_l1048_104838

theorem solve_equation_real (x : ℝ) (h : (x ^ 2 - x + 1) * (3 * x ^ 2 - 10 * x + 3) = 20 * x ^ 2) :
    x = (5 + Real.sqrt 21) / 2 ∨ x = (5 - Real.sqrt 21) / 2 :=
by
  sorry

end solve_equation_real_l1048_104838


namespace find_b_for_parallel_lines_l1048_104872

theorem find_b_for_parallel_lines :
  (∀ (b : ℝ), (∃ (f g : ℝ → ℝ),
  (∀ x, f x = 3 * x + b) ∧
  (∀ x, g x = (b + 9) * x - 2) ∧
  (∀ x, f x = g x → False)) →
  b = -6) :=
sorry

end find_b_for_parallel_lines_l1048_104872


namespace quadratic_equal_roots_l1048_104869

theorem quadratic_equal_roots :
  ∀ x : ℝ, 4 * x^2 - 4 * x + 1 = 0 → (0 ≤ 0) ∧ 
  (∀ a b : ℝ, 0 = b^2 - 4 * a * 1 → (x = -b / (2 * a))) :=
by
  sorry

end quadratic_equal_roots_l1048_104869


namespace equal_area_intersection_l1048_104858

variable (p q r s : ℚ)
noncomputable def intersection_point (x y : ℚ) : Prop :=
  4 * x + 5 * p / q = 12 * p / q ∧ 8 * y = p 

theorem equal_area_intersection :
  intersection_point p q r s /\
  p + q + r + s = 60 := 
by 
  sorry

end equal_area_intersection_l1048_104858


namespace find_remaining_area_l1048_104849

theorem find_remaining_area 
    (base_RST : ℕ) 
    (height_RST : ℕ) 
    (base_RSC : ℕ) 
    (height_RSC : ℕ) 
    (area_RST : ℕ := (1 / 2) * base_RST * height_RST) 
    (area_RSC : ℕ := (1 / 2) * base_RSC * height_RSC) 
    (remaining_area : ℕ := area_RST - area_RSC) 
    (h_base_RST : base_RST = 5) 
    (h_height_RST : height_RST = 4) 
    (h_base_RSC : base_RSC = 1) 
    (h_height_RSC : height_RSC = 4) : 
    remaining_area = 8 := 
by 
  sorry

end find_remaining_area_l1048_104849


namespace total_amount_spent_l1048_104837

def price_per_deck (n : ℕ) : ℝ :=
if n <= 3 then 8 else if n <= 6 then 7 else 6

def promotion_price (price : ℝ) : ℝ :=
price * 0.5

def total_cost (decks_victor decks_friend : ℕ) : ℝ :=
let cost_victor :=
  if decks_victor % 2 = 0 then
    let pairs := decks_victor / 2
    price_per_deck decks_victor * pairs + promotion_price (price_per_deck decks_victor) * pairs
  else sorry
let cost_friend :=
  if decks_friend = 2 then
    price_per_deck decks_friend + promotion_price (price_per_deck decks_friend)
  else sorry
cost_victor + cost_friend

theorem total_amount_spent : total_cost 6 2 = 43.5 := sorry

end total_amount_spent_l1048_104837


namespace dan_spent_amount_l1048_104895

-- Defining the prices of items
def candy_bar_price : ℝ := 7
def chocolate_price : ℝ := 6
def gum_price : ℝ := 3
def chips_price : ℝ := 4

-- Defining the discount and tax rates
def candy_bar_discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.05

-- Defining the steps to calculate the total price including discount and tax
def total_before_discount_and_tax := candy_bar_price + chocolate_price + gum_price + chips_price
def candy_bar_discount := candy_bar_discount_rate * candy_bar_price
def candy_bar_after_discount := candy_bar_price - candy_bar_discount
def total_after_discount := candy_bar_after_discount + chocolate_price + gum_price + chips_price
def tax := tax_rate * total_after_discount
def total_with_discount_and_tax := total_after_discount + tax

theorem dan_spent_amount : total_with_discount_and_tax = 20.27 :=
by sorry

end dan_spent_amount_l1048_104895


namespace cheesecake_factory_working_days_l1048_104889

-- Define the savings rates
def robby_saves := 2 / 5
def jaylen_saves := 3 / 5
def miranda_saves := 1 / 2

-- Define their hourly rate and daily working hours
def hourly_rate := 10 -- dollars per hour
def work_hours_per_day := 10 -- hours per day

-- Define their combined savings after four weeks and the combined savings target
def four_weeks := 4 * 7
def combined_savings_target := 3000 -- dollars

-- Question: Prove that the number of days they work per week is 7
theorem cheesecake_factory_working_days (d : ℕ) (h : d * 400 = combined_savings_target / 4) : d = 7 := sorry

end cheesecake_factory_working_days_l1048_104889


namespace least_number_to_multiply_for_multiple_of_112_l1048_104880

theorem least_number_to_multiply_for_multiple_of_112 (n : ℕ) : 
  (Nat.lcm 72 112) / 72 = 14 := 
sorry

end least_number_to_multiply_for_multiple_of_112_l1048_104880


namespace number_of_students_l1048_104819

theorem number_of_students (N T : ℕ) (h1 : T = 80 * N)
  (h2 : (T - 100) / (N - 5) = 90) : N = 35 := 
by 
  sorry

end number_of_students_l1048_104819


namespace rooks_control_chosen_squares_l1048_104859

theorem rooks_control_chosen_squares (n : Nat) 
  (chessboard : Fin (2 * n) × Fin (2 * n)) 
  (chosen_squares : Finset (Fin (2 * n) × Fin (2 * n))) 
  (h : chosen_squares.card = 3 * n) :
  ∃ rooks : Finset (Fin (2 * n) × Fin (2 * n)), rooks.card = n ∧
  ∀ (square : Fin (2 * n) × Fin (2 * n)), square ∈ chosen_squares → 
  (square ∈ rooks ∨ ∃ (rook : Fin (2 * n) × Fin (2 * n)) (hr : rook ∈ rooks), 
  rook.1 = square.1 ∨ rook.2 = square.2) :=
sorry

end rooks_control_chosen_squares_l1048_104859


namespace parallel_lines_eq_a2_l1048_104800

theorem parallel_lines_eq_a2
  (a : ℝ)
  (h : ∀ x y : ℝ, x + a * y - 1 = 0 → (a - 1) * x + a * y + 1 = 0)
  : a = 2 := 
  sorry

end parallel_lines_eq_a2_l1048_104800


namespace percentage_of_men_l1048_104898

variables {M W : ℝ}
variables (h1 : M + W = 100)
variables (h2 : 0.20 * M + 0.40 * W = 34)

theorem percentage_of_men :
  M = 30 :=
by
  sorry

end percentage_of_men_l1048_104898


namespace factor_congruence_l1048_104882

theorem factor_congruence (n : ℕ) (hn : n ≠ 0) :
  ∀ p : ℕ, p ∣ (2 * n)^(2^n) + 1 → p ≡ 1 [MOD 2^(n+1)] :=
sorry

end factor_congruence_l1048_104882


namespace at_least_one_non_negative_l1048_104840

variable (x : ℝ)
def a : ℝ := x^2 - 1
def b : ℝ := 2*x + 2

theorem at_least_one_non_negative (x : ℝ) : ¬ (a x < 0 ∧ b x < 0) :=
by
  sorry

end at_least_one_non_negative_l1048_104840


namespace man_is_older_by_22_l1048_104804

/-- 
Given the present age of the son is 20 years and in two years the man's age will be 
twice the age of his son, prove that the man is 22 years older than his son.
-/
theorem man_is_older_by_22 (S M : ℕ) (h1 : S = 20) (h2 : M + 2 = 2 * (S + 2)) : M - S = 22 :=
by
  sorry  -- Proof will be provided here

end man_is_older_by_22_l1048_104804


namespace value_of_x_y_squared_l1048_104863

theorem value_of_x_y_squared (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = 5) : (x - y)^2 = 16 :=
by
  sorry

end value_of_x_y_squared_l1048_104863


namespace model_distance_comparison_l1048_104833

theorem model_distance_comparison (m h c x y z : ℝ) (hm : 0 < m) (hh : 0 < h) (hc : 0 < c) (hz : 0 < z) (hx : 0 < x) (hy : 0 < y)
    (h_eq : (x - c) * z = (y - c) * (z + m) + h) :
    (if h > c * m then (x * z > y * (z + m))
     else if h < c * m then (x * z < y * (z + m))
     else (h = c * m → x * z = y * (z + m))) :=
by
  sorry

end model_distance_comparison_l1048_104833


namespace amy_carl_distance_after_2_hours_l1048_104824

-- Conditions
def amy_rate : ℤ := 1
def carl_rate : ℤ := 2
def amy_interval : ℤ := 20
def carl_interval : ℤ := 30
def time_hours : ℤ := 2
def minutes_per_hour : ℤ := 60

-- Derived values
def time_minutes : ℤ := time_hours * minutes_per_hour
def amy_distance : ℤ := time_minutes / amy_interval * amy_rate
def carl_distance : ℤ := time_minutes / carl_interval * carl_rate

-- Question and answer pair
def distance_amy_carl : ℤ := amy_distance + carl_distance
def expected_distance : ℤ := 14

-- The theorem to prove
theorem amy_carl_distance_after_2_hours : distance_amy_carl = expected_distance := by
  sorry

end amy_carl_distance_after_2_hours_l1048_104824


namespace shift_right_inverse_exp_eq_ln_l1048_104826

variable (f : ℝ → ℝ)

theorem shift_right_inverse_exp_eq_ln :
  (∀ x, f (x - 1) = Real.log x) → ∀ x, f x = Real.log (x + 1) :=
by
  sorry

end shift_right_inverse_exp_eq_ln_l1048_104826


namespace gain_per_year_is_correct_l1048_104894

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

theorem gain_per_year_is_correct :
  let borrowed_amount := 7000
  let borrowed_rate := 0.04
  let borrowed_time := 2
  let borrowed_compound_freq := 1 -- annually
  
  let lent_amount := 7000
  let lent_rate := 0.06
  let lent_time := 2
  let lent_compound_freq := 2 -- semi-annually
  
  let amount_owed := compound_interest borrowed_amount borrowed_rate borrowed_compound_freq borrowed_time
  let amount_received := compound_interest lent_amount lent_rate lent_compound_freq lent_time
  let total_gain := amount_received - amount_owed
  let gain_per_year := total_gain / lent_time
  
  gain_per_year = 153.65 :=
by
  sorry

end gain_per_year_is_correct_l1048_104894


namespace hyperbola_center_l1048_104822

theorem hyperbola_center (x1 y1 x2 y2 : ℝ) (h₁ : x1 = 3) (h₂ : y1 = 2) (h₃ : x2 = 11) (h₄ : y2 = 6) :
  (x1 + x2) / 2 = 7 ∧ (y1 + y2) / 2 = 4 :=
by
  -- Use the conditions h₁, h₂, h₃, and h₄ to substitute values and prove the statement
  sorry

end hyperbola_center_l1048_104822


namespace digits_product_l1048_104891

-- Define the conditions
variables (A B : ℕ)

-- Define the main problem statement using the conditions and expected answer
theorem digits_product (h1 : A + B = 12) (h2 : (10 * A + B) % 3 = 0) : A * B = 35 := 
by
  sorry

end digits_product_l1048_104891


namespace find_total_photos_l1048_104885

noncomputable def total_photos (T : ℕ) (Paul Tim Tom : ℕ) : Prop :=
  Tim = T - 100 ∧ Paul = Tim + 10 ∧ Tom = 38 ∧ Tom + Tim + Paul = T

theorem find_total_photos : ∃ T, total_photos T (T - 90) (T - 100) 38 :=
sorry

end find_total_photos_l1048_104885


namespace find_k_l1048_104846

variables (k : ℝ)
def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (-3, 2)
def vector_k_a_plus_b (k : ℝ) : ℝ × ℝ := (k*1 + (-3), k*2 + 2)
def vector_a_minus_2b : ℝ × ℝ := (1 - 2*(-3), 2 - 2*2)

theorem find_k (h : (vector_k_a_plus_b k).fst * (vector_a_minus_2b).snd = (vector_k_a_plus_b k).snd * (vector_a_minus_2b).fst) : k = -1/2 :=
sorry

end find_k_l1048_104846


namespace sam_last_30_minutes_speed_l1048_104866

/-- 
Given the total distance of 96 miles driven in 1.5 hours, 
with the first 30 minutes at an average speed of 60 mph, 
and the second 30 minutes at an average speed of 65 mph,
we need to show that the average speed during the last 30 minutes was 67 mph.
-/
theorem sam_last_30_minutes_speed (total_distance : ℤ) (time1 time2 : ℤ) (speed1 speed2 speed_last segment_time : ℤ)
  (h_total_distance : total_distance = 96)
  (h_total_time : time1 + time2 + segment_time = 90)
  (h_segment_time : segment_time = 30)
  (convert_time1 : time1 = 30)
  (convert_time2 : time2 = 30)
  (h_speed1 : speed1 = 60)
  (h_speed2 : speed2 = 65)
  (h_average_speed : ((60 + 65 + speed_last) / 3) = 64) :
  speed_last = 67 := 
sorry

end sam_last_30_minutes_speed_l1048_104866


namespace first_train_speed_l1048_104816

theorem first_train_speed:
  ∃ v : ℝ, 
    (∀ t : ℝ, t = 1 → (v * t) + (4 * v) = 200) ∧ 
    (∀ t : ℝ, t = 4 → 50 * t = 200) → 
    v = 40 :=
by {
 sorry
}

end first_train_speed_l1048_104816


namespace water_consumption_correct_l1048_104848

theorem water_consumption_correct (w n r : ℝ) 
  (hw : w = 21428) 
  (hn : n = 26848.55) 
  (hr : r = 302790.13) :
  w = 21428 ∧ n = 26848.55 ∧ r = 302790.13 :=
by 
  sorry

end water_consumption_correct_l1048_104848


namespace pushups_total_l1048_104821

theorem pushups_total (z d e : ℕ)
  (hz : z = 44) 
  (hd : d = z + 58) 
  (he : e = 2 * d) : 
  z + d + e = 350 := by
  sorry

end pushups_total_l1048_104821


namespace total_earnings_from_peaches_l1048_104879

-- Definitions of the conditions
def total_peaches : ℕ := 15
def peaches_sold_to_friends : ℕ := 10
def price_per_peach_friends : ℝ := 2
def peaches_sold_to_relatives : ℕ :=  4
def price_per_peach_relatives : ℝ := 1.25
def peaches_for_self : ℕ := 1

-- We aim to prove the following statement
theorem total_earnings_from_peaches :
  (peaches_sold_to_friends * price_per_peach_friends) +
  (peaches_sold_to_relatives * price_per_peach_relatives) = 25 := by
  -- proof goes here
  sorry

end total_earnings_from_peaches_l1048_104879


namespace total_fish_catch_l1048_104827

noncomputable def Johnny_fishes : ℕ := 8
noncomputable def Sony_fishes : ℕ := 4 * Johnny_fishes
noncomputable def total_fishes : ℕ := Sony_fishes + Johnny_fishes

theorem total_fish_catch : total_fishes = 40 := by
  sorry

end total_fish_catch_l1048_104827


namespace right_triangle_area_l1048_104871

theorem right_triangle_area (h : Real) (a : Real) (b : Real) (c : Real) (h_is_hypotenuse : h = 13) (a_is_leg : a = 5) (pythagorean_theorem : a^2 + b^2 = h^2) : (1 / 2) * a * b = 30 := 
by 
  sorry

end right_triangle_area_l1048_104871


namespace prove_distance_uphill_l1048_104805

noncomputable def distance_uphill := 
  let flat_speed := 20
  let uphill_speed := 12
  let extra_flat_distance := 30
  let uphill_time (D : ℝ) := D / uphill_speed
  let flat_time (D : ℝ) := (D + extra_flat_distance) / flat_speed
  ∃ D : ℝ, uphill_time D = flat_time D ∧ D = 45

theorem prove_distance_uphill : distance_uphill :=
sorry

end prove_distance_uphill_l1048_104805


namespace find_number_l1048_104812

theorem find_number (x : ℝ) : (5 / 3) * x = 45 → x = 27 := by
  sorry

end find_number_l1048_104812


namespace catherine_initial_pens_l1048_104809

-- Defining the conditions
def equal_initial_pencils_and_pens (P : ℕ) : Prop := true
def pens_given_away_per_friend : ℕ := 8
def pencils_given_away_per_friend : ℕ := 6
def number_of_friends : ℕ := 7
def remaining_pens_and_pencils : ℕ := 22

-- The total number of items given away
def total_pens_given_away : ℕ := pens_given_away_per_friend * number_of_friends
def total_pencils_given_away : ℕ := pencils_given_away_per_friend * number_of_friends

-- The problem statement in Lean 4
theorem catherine_initial_pens (P : ℕ) 
  (h1 : equal_initial_pencils_and_pens P)
  (h2 : P - total_pens_given_away + P - total_pencils_given_away = remaining_pens_and_pencils) : 
  P = 60 :=
sorry

end catherine_initial_pens_l1048_104809


namespace xyz_squared_eq_one_l1048_104815

theorem xyz_squared_eq_one (x y z : ℝ) (h_distinct : x ≠ y ∧ y ≠ z ∧ z ≠ x)
    (h_eq : ∃ k, x + (1 / y) = k ∧ y + (1 / z) = k ∧ z + (1 / x) = k) : 
    x^2 * y^2 * z^2 = 1 := 
  sorry

end xyz_squared_eq_one_l1048_104815


namespace jacks_remaining_capacity_l1048_104870

noncomputable def jacks_basket_full_capacity : ℕ := 12
noncomputable def jills_basket_full_capacity : ℕ := 2 * jacks_basket_full_capacity
noncomputable def jacks_current_apples (x : ℕ) : Prop := 3 * x = jills_basket_full_capacity

theorem jacks_remaining_capacity {x : ℕ} (hx : jacks_current_apples x) :
  jacks_basket_full_capacity - x = 4 :=
by sorry

end jacks_remaining_capacity_l1048_104870


namespace billy_age_is_45_l1048_104831

variable (Billy_age Joe_age : ℕ)

-- Given conditions
def condition1 := Billy_age = 3 * Joe_age
def condition2 := Billy_age + Joe_age = 60
def condition3 := Billy_age > 60 / 2

-- Prove Billy's age is 45
theorem billy_age_is_45 (h1 : condition1 Billy_age Joe_age) (h2 : condition2 Billy_age Joe_age) (h3 : condition3 Billy_age) : Billy_age = 45 :=
by
  sorry

end billy_age_is_45_l1048_104831


namespace sum_of_other_endpoint_coordinates_l1048_104874

theorem sum_of_other_endpoint_coordinates (x y : ℤ) :
  (7 + x) / 2 = 5 ∧ (4 + y) / 2 = -8 → x + y = -17 :=
by 
  sorry

end sum_of_other_endpoint_coordinates_l1048_104874


namespace find_a_even_function_l1048_104893

theorem find_a_even_function (a : ℝ) :
  (∀ x : ℝ, (x ^ 2 + a * x - 4) = ((-x) ^ 2 + a * (-x) - 4)) → a = 0 :=
by
  intro h
  sorry

end find_a_even_function_l1048_104893


namespace josh_500_coins_impossible_l1048_104841

theorem josh_500_coins_impossible : ¬ ∃ (x y : ℕ), x + y ≤ 500 ∧ 36 * x + 6 * y + (500 - x - y) = 3564 := 
sorry

end josh_500_coins_impossible_l1048_104841


namespace systematic_sampling_correct_l1048_104899

-- Definitions for the conditions
def total_products := 60
def group_count := 5
def products_per_group := total_products / group_count

-- systematic sampling condition: numbers are in increments of products_per_group
def systematic_sample (start : ℕ) (count : ℕ) : List ℕ := List.range' start products_per_group count

-- Given sequences
def A : List ℕ := [5, 10, 15, 20, 25]
def B : List ℕ := [5, 12, 31, 39, 57]
def C : List ℕ := [5, 17, 29, 41, 53]
def D : List ℕ := [5, 15, 25, 35, 45]

-- Correct solution defined
def correct_solution := [5, 17, 29, 41, 53]

-- Problem Statement
theorem systematic_sampling_correct :
  systematic_sample 5 group_count = correct_solution :=
by
  sorry

end systematic_sampling_correct_l1048_104899


namespace largest_divisor_8_l1048_104875

theorem largest_divisor_8 (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) (h : q < p) : 
  8 ∣ (p^2 - q^2 + 2*p - 2*q) := 
sorry

end largest_divisor_8_l1048_104875


namespace moles_CO2_is_one_l1048_104873

noncomputable def moles_CO2_formed (moles_HNO3 moles_NaHCO3 : ℕ) : ℕ :=
  if moles_HNO3 = 1 ∧ moles_NaHCO3 = 1 then 1 else 0

theorem moles_CO2_is_one :
  moles_CO2_formed 1 1 = 1 :=
by
  sorry

end moles_CO2_is_one_l1048_104873


namespace root_of_linear_equation_l1048_104862

theorem root_of_linear_equation (b c : ℝ) (hb : b ≠ 0) :
  ∃ x : ℝ, 0 * x^2 + b * x + c = 0 → x = -c / b :=
by
  -- The proof steps would typically go here
  sorry

end root_of_linear_equation_l1048_104862


namespace rectangular_solid_surface_area_l1048_104839

theorem rectangular_solid_surface_area (a b c : ℕ) (h_a_prime : Nat.Prime a) (h_b_prime : Nat.Prime b) (h_c_prime : Nat.Prime c) 
  (volume_eq : a * b * c = 273) :
  2 * (a * b + b * c + c * a) = 302 := 
sorry

end rectangular_solid_surface_area_l1048_104839


namespace unique_function_solution_l1048_104877

variable (f : ℝ → ℝ)

theorem unique_function_solution :
  (∀ x y : ℝ, f (f x - y^2) = f x ^ 2 - 2 * f x * y^2 + f (f y))
  → (∀ x : ℝ, f x = x^2) :=
by
  sorry

end unique_function_solution_l1048_104877


namespace multiplication_of_mixed_number_l1048_104817

theorem multiplication_of_mixed_number :
  7 * (9 + 2/5 : ℚ) = 65 + 4/5 :=
by
  -- to start the proof
  sorry

end multiplication_of_mixed_number_l1048_104817


namespace vector_subtraction_scalar_mul_l1048_104861

theorem vector_subtraction_scalar_mul :
  let v₁ := (3, -8) 
  let scalar := -5 
  let v₂ := (4, 6)
  v₁.1 - scalar * v₂.1 = 23 ∧ v₁.2 - scalar * v₂.2 = 22 := by
    sorry

end vector_subtraction_scalar_mul_l1048_104861


namespace janet_freelancer_income_difference_l1048_104853

theorem janet_freelancer_income_difference :
  let hours_per_week := 40
  let current_job_hourly_rate := 30
  let freelancer_hourly_rate := 40
  let fica_taxes_per_week := 25
  let healthcare_premiums_per_month := 400
  let weeks_per_month := 4
  
  let current_job_weekly_income := hours_per_week * current_job_hourly_rate
  let current_job_monthly_income := current_job_weekly_income * weeks_per_month
  
  let freelancer_weekly_income := hours_per_week * freelancer_hourly_rate
  let freelancer_monthly_income := freelancer_weekly_income * weeks_per_month
  
  let freelancer_monthly_fica_taxes := fica_taxes_per_week * weeks_per_month
  let freelancer_total_additional_costs := freelancer_monthly_fica_taxes + healthcare_premiums_per_month
  
  let freelancer_net_monthly_income := freelancer_monthly_income - freelancer_total_additional_costs
  
  freelancer_net_monthly_income - current_job_monthly_income = 1100 :=
by
  sorry

end janet_freelancer_income_difference_l1048_104853


namespace polygon_sides_l1048_104886

theorem polygon_sides (n : ℕ) (hn : 3 ≤ n) (H : (n * (n - 3)) / 2 = 15) : n = 7 :=
by
  sorry

end polygon_sides_l1048_104886


namespace sum_of_divisors_of_11_squared_l1048_104890

theorem sum_of_divisors_of_11_squared (a b c : ℕ) (h1 : a ∣ 11^2) (h2 : b ∣ 11^2) (h3 : c ∣ 11^2) (h4 : a * b * c = 11^2) (h5 : a ≠ b) (h6 : b ≠ c) (h7 : a ≠ c) :
  a + b + c = 23 :=
sorry

end sum_of_divisors_of_11_squared_l1048_104890


namespace B_completes_remaining_work_in_2_days_l1048_104892

theorem B_completes_remaining_work_in_2_days 
  (A_work_rate : ℝ) (B_work_rate : ℝ) (total_work : ℝ) 
  (A_days_to_complete : A_work_rate = 1 / 2) 
  (B_days_to_complete : B_work_rate = 1 / 6) 
  (combined_work_1_day : A_work_rate + B_work_rate = 2 / 3) : 
  (total_work - (A_work_rate + B_work_rate)) / B_work_rate = 2 := 
by
  sorry

end B_completes_remaining_work_in_2_days_l1048_104892


namespace xiaolong_correct_answers_l1048_104808

/-- There are 50 questions in the exam. Correct answers earn 3 points each,
incorrect answers deduct 1 point each, and unanswered questions score 0 points.
Xiaolong scored 120 points. Prove that the maximum number of questions 
Xiaolong answered correctly is 42. -/
theorem xiaolong_correct_answers :
  ∃ (x y : ℕ), 3 * x - y = 120 ∧ x + y = 48 ∧ x ≤ 50 ∧ y ≤ 50 ∧ x = 42 :=
by
  sorry

end xiaolong_correct_answers_l1048_104808


namespace problem_statement_l1048_104850

theorem problem_statement (a b : ℝ) (h1 : a^3 - b^3 = 2) (h2 : a^5 - b^5 ≥ 4) : a^2 + b^2 ≥ 2 := 
sorry

end problem_statement_l1048_104850


namespace frequency_of_zero_in_3021004201_l1048_104897

def digit_frequency (n : Nat) (d : Nat) :  Rat :=
  let digits := n.digits 10
  let count_d := digits.count d
  (count_d : Rat) / digits.length

theorem frequency_of_zero_in_3021004201 : 
  digit_frequency 3021004201 0 = 0.4 := 
by 
  sorry

end frequency_of_zero_in_3021004201_l1048_104897


namespace evaluate_ceiling_expression_l1048_104828

theorem evaluate_ceiling_expression:
  (Int.ceil ((23 : ℚ) / 9 - Int.ceil ((35 : ℚ) / 23)))
  / (Int.ceil ((35 : ℚ) / 9 + Int.ceil ((9 * 23 : ℚ) / 35))) = 1 / 12 := by
  sorry

end evaluate_ceiling_expression_l1048_104828


namespace part_a_part_b_l1048_104836

-- Definition of the function f and the condition it satisfies
variable (f : ℕ → ℕ)
variable (k n : ℕ)

theorem part_a (h1 : ∀ k n : ℕ, (k * f n) ≤ f (k * n) ∧ f (k * n) ≤ (k * f n) + k - 1)
  (a b : ℕ) :
  f a + f b ≤ f (a + b) ∧ f (a + b) ≤ f a + f b + 1 :=
by
  exact sorry  -- Proof to be supplied

theorem part_b (h1 : ∀ k n : ℕ, (k * f n) ≤ f (k * n) ∧ f (k * n) ≤ (k * f n) + k - 1)
  (h2 : ∀ n : ℕ, f (2007 * n) ≤ 2007 * f n + 200) :
  ∃ c : ℕ, f (2007 * c) = 2007 * f c :=
by
  exact sorry  -- Proof to be supplied

end part_a_part_b_l1048_104836


namespace max_XG_l1048_104835

theorem max_XG :
  ∀ (G X Y Z : ℝ),
    Y - X = 5 ∧ Z - Y = 3 ∧ (1 / G + 1 / (G - 5) + 1 / (G - 8) = 0) →
    G = 20 / 3 :=
by
  sorry

end max_XG_l1048_104835


namespace tony_initial_money_l1048_104847

theorem tony_initial_money (ticket_cost hotdog_cost money_left initial_money : ℕ) 
  (h_ticket : ticket_cost = 8)
  (h_hotdog : hotdog_cost = 3) 
  (h_left : money_left = 9)
  (h_spent : initial_money = ticket_cost + hotdog_cost + money_left) :
  initial_money = 20 := 
by 
  sorry

end tony_initial_money_l1048_104847


namespace matrix_det_problem_l1048_104813

-- Define the determinant of a 2x2 matrix
def det (a b c d : ℤ) : ℤ := a * d - b * c

-- State the problem in Lean
theorem matrix_det_problem : 2 * det 5 7 2 3 = 2 := by
  sorry

end matrix_det_problem_l1048_104813


namespace integer_solutions_pxy_eq_xy_l1048_104867

theorem integer_solutions_pxy_eq_xy (p : ℤ) (hp : Prime p) :
  ∃ x y : ℤ, p * (x + y) = x * y ∧ 
  ((x, y) = (2 * p, 2 * p) ∨ 
  (x, y) = (0, 0) ∨ 
  (x, y) = (p + 1, p + p^2) ∨ 
  (x, y) = (p - 1, p - p^2) ∨ 
  (x, y) = (p + p^2, p + 1) ∨ 
  (x, y) = (p - p^2, p - 1)) :=
by
  sorry

end integer_solutions_pxy_eq_xy_l1048_104867


namespace saturday_earnings_l1048_104896

-- Lean 4 Statement

theorem saturday_earnings 
  (S Wednesday_earnings : ℝ)
  (h1 : S + Wednesday_earnings = 5182.50)
  (h2 : Wednesday_earnings = S - 142.50) 
  : S = 2662.50 := 
by
  sorry

end saturday_earnings_l1048_104896


namespace expression_evaluation_l1048_104814

theorem expression_evaluation : (6 * 111) - (2 * 111) = 444 :=
by
  sorry

end expression_evaluation_l1048_104814


namespace plan_y_cheaper_than_plan_x_l1048_104864

def cost_plan_x (z : ℕ) : ℕ := 15 * z

def cost_plan_y (z : ℕ) : ℕ :=
  if z > 500 then 3000 + 7 * z - 1000 else 3000 + 7 * z

theorem plan_y_cheaper_than_plan_x (z : ℕ) (h : z > 500) : cost_plan_y z < cost_plan_x z :=
by
  sorry

end plan_y_cheaper_than_plan_x_l1048_104864


namespace japanese_turtle_crane_problem_l1048_104855

theorem japanese_turtle_crane_problem (x y : ℕ) (h1 : x + y = 35) (h2 : 2 * x + 4 * y = 94) : x + y = 35 ∧ 2 * x + 4 * y = 94 :=
by
  sorry

end japanese_turtle_crane_problem_l1048_104855


namespace hadley_total_distance_l1048_104884

def distance_to_grocery := 2
def distance_to_pet_store := 2 - 1
def distance_back_home := 4 - 1

theorem hadley_total_distance : distance_to_grocery + distance_to_pet_store + distance_back_home = 6 :=
by
  -- Proof is omitted.
  sorry

end hadley_total_distance_l1048_104884


namespace true_proposition_l1048_104811

-- Define the propositions p and q
def p : Prop := 2 % 2 = 0
def q : Prop := 5 % 2 = 0

-- Define the problem statement
theorem true_proposition (hp : p) (hq : ¬ q) : p ∨ q :=
by
  sorry

end true_proposition_l1048_104811


namespace collinear_vectors_x_eq_neg_two_l1048_104810

theorem collinear_vectors_x_eq_neg_two (x : ℝ) (a b : ℝ×ℝ) :
  a = (1, 2) → b = (x, -4) → a.1 * b.2 = a.2 * b.1 → x = -2 :=
by
  intro ha hb hc
  sorry

end collinear_vectors_x_eq_neg_two_l1048_104810


namespace sequence_diff_l1048_104887

theorem sequence_diff (x : ℕ → ℕ)
  (h1 : ∀ n, x n < x (n + 1))
  (h2 : ∀ n, 2 * n + 1 ≤ x (2 * n + 1)) :
  ∀ k : ℕ, ∃ r s : ℕ, x r - x s = k :=
by
  sorry

end sequence_diff_l1048_104887


namespace sum_of_a_for_unique_solution_l1048_104806

theorem sum_of_a_for_unique_solution (a : ℝ) (h : (a + 12)^2 - 384 = 0) : 
  let a1 := -12 + 16 * Real.sqrt 6
  let a2 := -12 - 16 * Real.sqrt 6
  a1 + a2 = -24 := 
by
  sorry

end sum_of_a_for_unique_solution_l1048_104806


namespace strawberries_to_grapes_ratio_l1048_104803

-- Define initial conditions
def initial_grapes : ℕ := 100
def fruits_left : ℕ := 96

-- Define the number of strawberries initially
def strawberries_init (S : ℕ) : Prop :=
  (S - (2 * (1/5) * S) = fruits_left - initial_grapes + ((2 * (1/5)) * initial_grapes))

-- Define the ratio problem in Lean
theorem strawberries_to_grapes_ratio (S : ℕ) (h : strawberries_init S) : (S / initial_grapes = 3 / 5) :=
sorry

end strawberries_to_grapes_ratio_l1048_104803


namespace sum_of_coeffs_eq_one_l1048_104881

theorem sum_of_coeffs_eq_one (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) (x : ℝ) :
  (1 - 2 * x) ^ 10 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + 
                    a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 + a_9 * x^9 + a_10 * x^10 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 1 :=
  sorry

end sum_of_coeffs_eq_one_l1048_104881


namespace sum_of_coefficients_l1048_104851

-- Definition of the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem sum_of_coefficients (n : ℕ) (hn1 : 5 < n) (hn2 : n < 7)
  (coeff_cond : binom n 3 > binom n 2 ∧ binom n 3 > binom n 4) :
  (1 + 1)^n = 64 :=
by
  have h : n = 6 :=
    by sorry -- provided conditions force n to be 6
  show 2^n = 64
  rw [h]
  exact rfl

end sum_of_coefficients_l1048_104851


namespace arthur_additional_muffins_l1048_104834

/-- Define the number of muffins Arthur has already baked -/
def muffins_baked : ℕ := 80

/-- Define the multiplier for the total output Arthur wants -/
def desired_multiplier : ℝ := 2.5

/-- Define the equation representing the total desired muffins -/
def total_muffins : ℝ := muffins_baked * desired_multiplier

/-- Define the number of additional muffins Arthur needs to bake -/
def additional_muffins : ℝ := total_muffins - muffins_baked

theorem arthur_additional_muffins : additional_muffins = 120 := by
  sorry

end arthur_additional_muffins_l1048_104834


namespace billy_points_difference_l1048_104807

-- Condition Definitions
def billy_points : ℕ := 7
def friend_points : ℕ := 9

-- Theorem stating the problem and the solution
theorem billy_points_difference : friend_points - billy_points = 2 :=
by 
  sorry

end billy_points_difference_l1048_104807
