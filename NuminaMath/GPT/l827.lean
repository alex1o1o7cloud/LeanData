import Mathlib

namespace rectangle_length_l827_82767

variable (w l : ℝ)

def perimeter (w l : ℝ) : ℝ := 2 * w + 2 * l

theorem rectangle_length (h1 : l = w + 2) (h2 : perimeter w l = 20) : l = 6 :=
by sorry

end rectangle_length_l827_82767


namespace cookies_per_person_l827_82769

variable (x y z : ℕ)
variable (h_pos_z : z ≠ 0) -- Ensure z is not zero to avoid division by zero

theorem cookies_per_person (h_cookies : x * y / z = 35) : 35 / 5 = 7 := by
  sorry

end cookies_per_person_l827_82769


namespace caterer_min_people_l827_82766

theorem caterer_min_people (x : ℕ) : 150 + 18 * x > 250 + 15 * x → x ≥ 34 :=
by
  intro h
  sorry

end caterer_min_people_l827_82766


namespace slices_served_today_l827_82752

-- Definitions based on conditions from part a)
def slices_lunch_today : ℕ := 7
def slices_dinner_today : ℕ := 5

-- Proof statement based on part c)
theorem slices_served_today : slices_lunch_today + slices_dinner_today = 12 := 
by
  sorry

end slices_served_today_l827_82752


namespace instantaneous_velocity_at_2_l827_82726

def displacement (t : ℝ) : ℝ := 14 * t - t^2 

def velocity (t : ℝ) : ℝ :=
  sorry -- The velocity function which is the derivative of displacement

theorem instantaneous_velocity_at_2 :
  velocity 2 = 10 := 
  sorry

end instantaneous_velocity_at_2_l827_82726


namespace Tim_paid_amount_l827_82718

theorem Tim_paid_amount (original_price : ℝ) (discount_percentage : ℝ) (discounted_price : ℝ) 
    (h1 : original_price = 1200) (h2 : discount_percentage = 0.15) 
    (discount_amount : ℝ) (h3 : discount_amount = original_price * discount_percentage) 
    (h4 : discounted_price = original_price - discount_amount) : discounted_price = 1020 := 
    by {
        sorry
    }

end Tim_paid_amount_l827_82718


namespace remainder_division_l827_82703

theorem remainder_division (a b : ℕ) (h1 : a > b) (h2 : (a - b) % 6 = 5) : a % 6 = 5 :=
sorry

end remainder_division_l827_82703


namespace rational_inequality_solution_l827_82744

variable (x : ℝ)

def inequality_conditions : Prop := (2 * x - 1) / (x + 1) > 1

def inequality_solution : Prop := x < -1 ∨ x > 2

theorem rational_inequality_solution : inequality_conditions x → inequality_solution x :=
by
  sorry

end rational_inequality_solution_l827_82744


namespace responses_needed_l827_82787

theorem responses_needed (p : ℝ) (q : ℕ) (r : ℕ) : 
  p = 0.6 → q = 370 → r = 222 → 
  q * p = r := 
by
  intros hp hq hr
  rw [hp, hq] 
  sorry

end responses_needed_l827_82787


namespace apples_difference_l827_82735

-- Definitions for initial and remaining apples
def initial_apples : ℕ := 46
def remaining_apples : ℕ := 14

-- The theorem to prove the difference between initial and remaining apples is 32
theorem apples_difference : initial_apples - remaining_apples = 32 := by
  -- proof is omitted
  sorry

end apples_difference_l827_82735


namespace gcd_of_three_digit_palindromes_l827_82724

def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ n = 101 * a + 10 * b

theorem gcd_of_three_digit_palindromes :
  ∀ n, is_palindrome n → Nat.gcd n 1 = 1 := by
  sorry

end gcd_of_three_digit_palindromes_l827_82724


namespace part1_part2_l827_82710

noncomputable def f (x : ℝ) : ℝ := x^3 + 1 / (1 + x)

theorem part1 (x : ℝ) (h : 0 ≤ x) (h1 : x ≤ 1) : f x ≥ 1 - x + x^2 := sorry

theorem part2 (x : ℝ) (h : 0 ≤ x) (h1 : x ≤ 1) : 3 / 4 < f x ∧ f x ≤ 3 / 2 := sorry

end part1_part2_l827_82710


namespace HunterScoreIs45_l827_82792

variable (G J H : ℕ)
variable (h1 : G = J + 10)
variable (h2 : J = 2 * H)
variable (h3 : G = 100)

theorem HunterScoreIs45 : H = 45 := by
  sorry

end HunterScoreIs45_l827_82792


namespace simplify_expression_l827_82738

theorem simplify_expression (x y : ℤ) (h₁ : x = 2) (h₂ : y = -3) :
  ((2 * x - y) ^ 2 - (x - y) * (x + y) - 2 * y ^ 2) / x = 18 :=
by
  sorry

end simplify_expression_l827_82738


namespace squares_triangles_product_l827_82709

theorem squares_triangles_product :
  let S := 7
  let T := 10
  S * T = 70 :=
by
  let S := 7
  let T := 10
  show (S * T = 70)
  sorry

end squares_triangles_product_l827_82709


namespace at_most_one_perfect_square_l827_82764

theorem at_most_one_perfect_square (a : ℕ → ℕ) :
  (∀ n, a (n + 1) = a n ^ 3 + 103) →
  (∃ n1, ∃ n2, a n1 = k1^2 ∧ a n2 = k2^2) → n1 = n2 
    ∨ (∀ n, a n ≠ k1^2) 
    ∨ (∀ n, a n ≠ k2^2) :=
sorry

end at_most_one_perfect_square_l827_82764


namespace find_m_l827_82730

def g (n : Int) : Int :=
  if n % 2 ≠ 0 then n + 5 else 
  if n % 3 = 0 then n / 3 else n

theorem find_m (m : Int) 
  (h_odd : m % 2 ≠ 0) 
  (h_ggg : g (g (g m)) = 35) : 
  m = 85 := 
by
  sorry

end find_m_l827_82730


namespace combined_work_rate_l827_82739

def work_done_in_one_day (A B : ℕ) (work_to_days : ℕ -> ℕ) : ℚ :=
  (work_to_days A + work_to_days B)

theorem combined_work_rate (A : ℕ) (B : ℕ) (work_to_days : ℕ -> ℕ) :
  work_to_days A = 1/18 ∧ work_to_days B = 1/9 → work_done_in_one_day A B (work_to_days) = 1/6 :=
by
  sorry

end combined_work_rate_l827_82739


namespace find_definite_integers_l827_82779

theorem find_definite_integers (n d e f : ℕ) (h₁ : n = d + Int.sqrt (e + Int.sqrt f)) 
    (h₂: ∀ x : ℝ, x = d + Int.sqrt (e + Int.sqrt f) → 
        (4 / (x - 4) + 6 / (x - 6) + 18 / (x - 18) + 20 / (x - 20) = x^2 - 12 * x - 5))
        : d + e + f = 76 :=
sorry

end find_definite_integers_l827_82779


namespace ratio_of_amounts_l827_82798

theorem ratio_of_amounts
    (initial_cents : ℕ)
    (given_to_peter_cents : ℕ)
    (remaining_nickels : ℕ)
    (nickel_value : ℕ := 5)
    (nickels_initial := initial_cents / nickel_value)
    (nickels_to_peter := given_to_peter_cents / nickel_value)
    (nickels_remaining := nickels_initial - nickels_to_peter)
    (nickels_given_to_randi := nickels_remaining - remaining_nickels)
    (cents_to_randi := nickels_given_to_randi * nickel_value)
    (cents_initial : initial_cents = 95)
    (cents_peter : given_to_peter_cents = 25)
    (nickels_left : remaining_nickels = 4)
    :
    (cents_to_randi / given_to_peter_cents) = 2 :=
by
  sorry

end ratio_of_amounts_l827_82798


namespace theater_seats_l827_82720

theorem theater_seats (x y t : ℕ) (h1 : x = 532) (h2 : y = 218) (h3 : t = x + y) : t = 750 := 
by 
  rw [h1, h2] at h3
  exact h3

end theater_seats_l827_82720


namespace fish_cost_l827_82795

theorem fish_cost (F P : ℝ) (h1 : 4 * F + 2 * P = 530) (h2 : 7 * F + 3 * P = 875) : F = 80 := 
by
  sorry

end fish_cost_l827_82795


namespace quadratic_min_value_l827_82761

theorem quadratic_min_value (p q : ℝ) (h : ∀ x : ℝ, 3 * x^2 + p * x + q ≥ 4) : q = p^2 / 12 + 4 :=
sorry

end quadratic_min_value_l827_82761


namespace combined_population_correct_l827_82796

theorem combined_population_correct (W PP LH N : ℕ) 
  (hW : W = 900)
  (hPP : PP = 7 * W)
  (hLH : LH = 2 * W + 600)
  (hN : N = 3 * (PP - W)) :
  PP + LH + N = 24900 :=
by
  sorry

end combined_population_correct_l827_82796


namespace ratio_of_average_speeds_l827_82743

theorem ratio_of_average_speeds
    (time_eddy : ℝ) (distance_eddy : ℝ)
    (time_freddy : ℝ) (distance_freddy : ℝ) :
  time_eddy = 3 ∧ distance_eddy = 600 ∧ time_freddy = 4 ∧ distance_freddy = 460 →
  (distance_eddy / time_eddy) / (distance_freddy / time_freddy) = 200 / 115 :=
by
  sorry

end ratio_of_average_speeds_l827_82743


namespace part1_part2_l827_82749

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end part1_part2_l827_82749


namespace fraction_equals_seven_twentyfive_l827_82751

theorem fraction_equals_seven_twentyfive :
  (1722^2 - 1715^2) / (1731^2 - 1706^2) = (7 / 25) :=
by
  sorry

end fraction_equals_seven_twentyfive_l827_82751


namespace distance_between_points_l827_82784

theorem distance_between_points (A B : ℝ) (hA : |A| = 2) (hB : |B| = 7) :
  |A - B| = 5 ∨ |A - B| = 9 := 
sorry

end distance_between_points_l827_82784


namespace polar_to_rectangular_coordinates_l827_82758

theorem polar_to_rectangular_coordinates (r θ : ℝ) (hr : r = 5) (hθ : θ = (3 * Real.pi) / 2) :
    (r * Real.cos θ, r * Real.sin θ) = (0, -5) :=
by
  rw [hr, hθ]
  simp [Real.cos, Real.sin]
  sorry

end polar_to_rectangular_coordinates_l827_82758


namespace find_first_discount_l827_82765

-- Definitions for the given conditions
def list_price : ℝ := 150
def final_price : ℝ := 105
def second_discount : ℝ := 12.5

-- Statement representing the mathematical proof problem
theorem find_first_discount (x : ℝ) : 
  list_price * ((100 - x) / 100) * ((100 - second_discount) / 100) = final_price → x = 20 :=
by
  sorry

end find_first_discount_l827_82765


namespace negation_proposition_false_l827_82786

variable (a : ℝ)

theorem negation_proposition_false : ¬ (∃ a : ℝ, a ≤ 2 ∧ a^2 ≥ 4) :=
sorry

end negation_proposition_false_l827_82786


namespace geese_left_in_the_field_l827_82780

theorem geese_left_in_the_field 
  (initial_geese : ℕ) 
  (geese_flew_away : ℕ) 
  (geese_joined : ℕ)
  (h1 : initial_geese = 372)
  (h2 : geese_flew_away = 178)
  (h3 : geese_joined = 57) :
  initial_geese - geese_flew_away + geese_joined = 251 := by
  sorry

end geese_left_in_the_field_l827_82780


namespace find_f_l827_82777

-- Define the conditions as hypotheses
def cond1 (f : ℕ) (p : ℕ) : Prop := f + p = 75
def cond2 (f : ℕ) (p : ℕ) : Prop := (f + p) + p = 143

-- The theorem stating that given the conditions, f must be 7
theorem find_f (f p : ℕ) (h1 : cond1 f p) (h2 : cond2 f p) : f = 7 := 
  by
  sorry

end find_f_l827_82777


namespace range_of_x_for_valid_sqrt_l827_82700

theorem range_of_x_for_valid_sqrt (x : ℝ) (h : 2 * x - 4 ≥ 0) : x ≥ 2 :=
by
  sorry

end range_of_x_for_valid_sqrt_l827_82700


namespace y_coords_diff_of_ellipse_incircle_area_l827_82740

theorem y_coords_diff_of_ellipse_incircle_area
  (x1 y1 x2 y2 : ℝ)
  (F1 F2 : ℝ × ℝ)
  (a b : ℝ)
  (h1 : a^2 = 25)
  (h2 : b^2 = 9)
  (h3 : F1 = (-4, 0))
  (h4 : F2 = (4, 0))
  (h5 : 4 * (|y1 - y2|) = 20)
  (h6 : ∃ (x : ℝ), (x / 25)^2 + (y1 / 9)^2 = 1 ∧ (x / 25)^2 + (y2 / 9)^2 = 1) :
  |y1 - y2| = 5 :=
sorry

end y_coords_diff_of_ellipse_incircle_area_l827_82740


namespace mixture_replacement_l827_82741

theorem mixture_replacement (A B T x : ℝ)
  (h1 : A / (A + B) = 7 / 12)
  (h2 : A = 21)
  (h3 : (A / (B + x)) = 7 / 9) :
  x = 12 :=
by
  sorry

end mixture_replacement_l827_82741


namespace selling_price_is_correct_l827_82734

-- Definitions of the given conditions

def cost_of_string_per_bracelet := 1
def cost_of_beads_per_bracelet := 3
def number_of_bracelets_sold := 25
def total_profit := 50

def cost_of_bracelet := cost_of_string_per_bracelet + cost_of_beads_per_bracelet
def total_cost := cost_of_bracelet * number_of_bracelets_sold
def total_revenue := total_profit + total_cost
def selling_price_per_bracelet := total_revenue / number_of_bracelets_sold

-- Target theorem
theorem selling_price_is_correct : selling_price_per_bracelet = 6 :=
  by
  sorry

end selling_price_is_correct_l827_82734


namespace dot_product_result_l827_82791

open Real

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-1, 2)

def scale_vec (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add_vec (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_result :
  dot_product (add_vec (scale_vec 2 a) b) a = 6 :=
by
  sorry

end dot_product_result_l827_82791


namespace computer_price_after_9_years_l827_82781

theorem computer_price_after_9_years 
  (initial_price : ℝ) (decrease_factor : ℝ) (years : ℕ) 
  (initial_price_eq : initial_price = 8100)
  (decrease_factor_eq : decrease_factor = 1 - 1/3)
  (years_eq : years = 9) :
  initial_price * (decrease_factor ^ (years / 3)) = 2400 := 
by
  sorry

end computer_price_after_9_years_l827_82781


namespace best_model_l827_82775

theorem best_model (R1 R2 R3 R4 : ℝ) :
  R1 = 0.78 → R2 = 0.85 → R3 = 0.61 → R4 = 0.31 →
  (R2 = max R1 (max R2 (max R3 R4))) :=
by
  intros hR1 hR2 hR3 hR4
  sorry

end best_model_l827_82775


namespace third_term_of_arithmetic_sequence_is_negative_22_l827_82729

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ := a + n * d

theorem third_term_of_arithmetic_sequence_is_negative_22
  (a d : ℤ)
  (H1 : arithmetic_sequence a d 14 = 14)
  (H2 : arithmetic_sequence a d 15 = 17) :
  arithmetic_sequence a d 2 = -22 :=
sorry

end third_term_of_arithmetic_sequence_is_negative_22_l827_82729


namespace evelyn_total_marbles_l827_82748

def initial_marbles := 95
def marbles_from_henry := 9
def marbles_from_grace := 12
def number_of_cards := 6
def marbles_per_card := 4

theorem evelyn_total_marbles :
  initial_marbles + marbles_from_henry + marbles_from_grace + number_of_cards * marbles_per_card = 140 := 
by 
  sorry

end evelyn_total_marbles_l827_82748


namespace smallest_is_C_l827_82790

def A : ℚ := 1/2
def B : ℚ := 9/10
def C : ℚ := 2/5

theorem smallest_is_C : min (min A B) C = C := 
by
  sorry

end smallest_is_C_l827_82790


namespace henri_total_time_l827_82702

variable (m1 m2 : ℝ) (r w : ℝ)

theorem henri_total_time (H1 : m1 = 3.5) (H2 : m2 = 1.5) (H3 : r = 10) (H4 : w = 1800) :
    m1 + m2 + w / r / 60 = 8 := by
  sorry

end henri_total_time_l827_82702


namespace tom_calories_l827_82736

theorem tom_calories :
  let carrot_pounds := 1
  let broccoli_pounds := 2 * carrot_pounds
  let carrot_calories_per_pound := 51
  let broccoli_calories_per_pound := carrot_calories_per_pound / 3
  let total_carrot_calories := carrot_pounds * carrot_calories_per_pound
  let total_broccoli_calories := broccoli_pounds * broccoli_calories_per_pound
  let total_calories := total_carrot_calories + total_broccoli_calories
  total_calories = 85 :=
by
  sorry

end tom_calories_l827_82736


namespace find_x_l827_82759

variable {a b x : ℝ}
variable (h₀ : b ≠ 0)
variable (h₁ : (3 * a)^(2 * b) = a^b * x^b)

theorem find_x (h₀ : b ≠ 0) (h₁ : (3 * a)^(2 * b) = a^b * x^b) : x = 9 * a :=
sorry

end find_x_l827_82759


namespace find_linear_function_l827_82794

theorem find_linear_function (a m : ℝ) : 
  (∀ x y : ℝ, (x, y) = (-2, -3) ∨ (x, y) = (-1, -1) ∨ (x, y) = (0, m) ∨ (x, y) = (1, 3) ∨ (x, y) = (a, 5) → 
  y = 2 * x + 1) → 
  (m = 1 ∧ a = 2) :=
by
  sorry

end find_linear_function_l827_82794


namespace central_projection_intersect_l827_82742

def central_projection (lines : Set (Set Point)) : Prop :=
  ∃ point : Point, ∀ line ∈ lines, line (point)

theorem central_projection_intersect :
  ∀ lines : Set (Set Point), central_projection lines → ∃ point : Point, ∀ line ∈ lines, line (point) :=
by
  sorry

end central_projection_intersect_l827_82742


namespace no_eight_roots_for_nested_quadratics_l827_82722

theorem no_eight_roots_for_nested_quadratics
  (f g h : ℝ → ℝ)
  (hf : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)
  (hg : ∃ d e k : ℝ, ∀ x, g x = d * x^2 + e * x + k)
  (hh : ∃ p q r : ℝ, ∀ x, h x = p * x^2 + q * x + r)
  (hroots : ∀ x, f (g (h x)) = 0 → (x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4 ∨ x = 5 ∨ x = 6 ∨ x = 7 ∨ x = 8)) :
  false :=
by
  sorry

end no_eight_roots_for_nested_quadratics_l827_82722


namespace find_zero_function_l827_82768

noncomputable def satisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x ^ 714 + y) = f (x ^ 2019) + f (y ^ 122)

theorem find_zero_function (f : ℝ → ℝ) (h : satisfiesCondition f) :
  ∀ x : ℝ, f x = 0 :=
sorry

end find_zero_function_l827_82768


namespace total_amount_paid_l827_82708

theorem total_amount_paid :
  let chapati_cost := 6
  let rice_cost := 45
  let mixed_vegetable_cost := 70
  let ice_cream_cost := 40
  let chapati_quantity := 16
  let rice_quantity := 5
  let mixed_vegetable_quantity := 7
  let ice_cream_quantity := 6
  let total_cost := chapati_quantity * chapati_cost +
                    rice_quantity * rice_cost +
                    mixed_vegetable_quantity * mixed_vegetable_cost +
                    ice_cream_quantity * ice_cream_cost
  total_cost = 1051 := by
  sorry

end total_amount_paid_l827_82708


namespace todd_money_left_l827_82753

def candy_bar_cost : ℝ := 2.50
def chewing_gum_cost : ℝ := 1.50
def soda_cost : ℝ := 3
def discount : ℝ := 0.20
def initial_money : ℝ := 50
def number_of_candy_bars : ℕ := 7
def number_of_chewing_gum : ℕ := 5
def number_of_soda : ℕ := 3

noncomputable def total_candy_bar_cost : ℝ := number_of_candy_bars * candy_bar_cost
noncomputable def total_chewing_gum_cost : ℝ := number_of_chewing_gum * chewing_gum_cost
noncomputable def total_soda_cost : ℝ := number_of_soda * soda_cost
noncomputable def discount_amount : ℝ := total_soda_cost * discount
noncomputable def discounted_soda_cost : ℝ := total_soda_cost - discount_amount
noncomputable def total_cost : ℝ := total_candy_bar_cost + total_chewing_gum_cost + discounted_soda_cost
noncomputable def money_left : ℝ := initial_money - total_cost

theorem todd_money_left : money_left = 17.80 :=
by sorry

end todd_money_left_l827_82753


namespace find_geometric_sequence_values_l827_82714

theorem find_geometric_sequence_values :
  ∃ (a b c : ℤ), (∃ q : ℤ, q ≠ 0 ∧ 2 * q ^ 4 = 32 ∧ a = 2 * q ∧ b = 2 * q ^ 2 ∧ c = 2 * q ^ 3)
                 ↔ ((a = 4 ∧ b = 8 ∧ c = 16) ∨ (a = -4 ∧ b = 8 ∧ c = -16)) := by
  sorry

end find_geometric_sequence_values_l827_82714


namespace combined_weight_l827_82701

theorem combined_weight (y z : ℝ) 
  (h_avg : (25 + 31 + 35 + 33) / 4 = (25 + 31 + 35 + 33 + y + z) / 6) :
  y + z = 62 :=
by
  sorry

end combined_weight_l827_82701


namespace shirt_cost_l827_82733

-- Definitions and conditions
def num_ten_bills : ℕ := 2
def num_twenty_bills : ℕ := num_ten_bills + 1

def ten_bill_value : ℕ := 10
def twenty_bill_value : ℕ := 20

-- Statement to prove
theorem shirt_cost :
  (num_ten_bills * ten_bill_value) + (num_twenty_bills * twenty_bill_value) = 80 :=
by
  sorry

end shirt_cost_l827_82733


namespace least_k_inequality_l827_82757

theorem least_k_inequality :
  ∃ k : ℝ, (∀ a b c : ℝ, 
    ((2 * a / (a - b)) ^ 2 + (2 * b / (b - c)) ^ 2 + (2 * c / (c - a)) ^ 2 + k 
    ≥ 4 * (2 * a / (a - b) + 2 * b / (b - c) + 2 * c / (c - a)))) ∧ k = 8 :=
by
  sorry  -- proof is omitted

end least_k_inequality_l827_82757


namespace leak_empty_time_l827_82754

theorem leak_empty_time
  (R : ℝ) (L : ℝ)
  (hR : R = 1 / 8)
  (hRL : R - L = 1 / 10) :
  1 / L = 40 :=
by
  sorry

end leak_empty_time_l827_82754


namespace number_of_slices_with_both_l827_82750

def total_slices : ℕ := 20
def slices_with_pepperoni : ℕ := 12
def slices_with_mushrooms : ℕ := 14
def slices_with_both_toppings (n : ℕ) : Prop :=
  n + (slices_with_pepperoni - n) + (slices_with_mushrooms - n) = total_slices

theorem number_of_slices_with_both (n : ℕ) (h : slices_with_both_toppings n) : n = 6 :=
sorry

end number_of_slices_with_both_l827_82750


namespace Charlie_age_when_Jenny_twice_as_Bobby_l827_82712

theorem Charlie_age_when_Jenny_twice_as_Bobby (B C J : ℕ) 
  (h₁ : J = C + 5)
  (h₂ : C = B + 3)
  (h₃ : J = 2 * B) : 
  C = 11 :=
by
  sorry

end Charlie_age_when_Jenny_twice_as_Bobby_l827_82712


namespace minutes_watched_on_Thursday_l827_82788

theorem minutes_watched_on_Thursday 
  (n_total : ℕ) (n_Mon : ℕ) (n_Tue : ℕ) (n_Wed : ℕ) (n_Fri : ℕ) (n_weekend : ℕ)
  (h_total : n_total = 352)
  (h_Mon : n_Mon = 138)
  (h_Tue : n_Tue = 0)
  (h_Wed : n_Wed = 0)
  (h_Fri : n_Fri = 88)
  (h_weekend : n_weekend = 105) :
  n_total - (n_Mon + n_Tue + n_Wed + n_Fri + n_weekend) = 21 := by
  sorry

end minutes_watched_on_Thursday_l827_82788


namespace taller_tree_height_l827_82799

theorem taller_tree_height
  (h : ℕ)
  (h_shorter_ratio : h - 16 = (3 * h) / 4) : h = 64 := by
  sorry

end taller_tree_height_l827_82799


namespace counties_no_rain_l827_82728

theorem counties_no_rain 
  (P_A : ℝ) (P_B : ℝ) (P_A_and_B : ℝ) :
  P_A = 0.7 → P_B = 0.5 → P_A_and_B = 0.4 →
  (1 - (P_A + P_B - P_A_and_B) = 0.2) :=
by intros h1 h2 h3; sorry

end counties_no_rain_l827_82728


namespace simplify_inv_sum_l827_82797

variables {x y z : ℝ}

theorem simplify_inv_sum (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x⁻¹ + y⁻¹ + z⁻¹)⁻¹ = xyz / (yz + xz + xy) :=
by
  sorry

end simplify_inv_sum_l827_82797


namespace maximum_k_l827_82711

theorem maximum_k (m k : ℝ) (h0 : 0 < m) (h1 : m < 1/2) (h2 : (1/m + 2/(1-2*m)) ≥ k): k ≤ 8 :=
sorry

end maximum_k_l827_82711


namespace fraction_transform_l827_82783

theorem fraction_transform {x : ℤ} :
  (537 - x : ℚ) / (463 + x) = 1 / 9 ↔ x = 437 := by
sorry

end fraction_transform_l827_82783


namespace largest_divisor_of_even_square_difference_l827_82713

theorem largest_divisor_of_even_square_difference (m n : ℕ) (hm : m % 2 = 0) (hn : n % 2 = 0) (h : n < m) :
  ∃ (k : ℕ), k = 8 ∧ ∀ m n : ℕ, m % 2 = 0 → n % 2 = 0 → n < m → k ∣ (m^2 - n^2) := by
  sorry

end largest_divisor_of_even_square_difference_l827_82713


namespace find_a_l827_82723

theorem find_a (a : ℚ) (h : a + a / 3 + a / 4 = 4) : a = 48 / 19 := by
  sorry

end find_a_l827_82723


namespace quadratic_root_l827_82706

/-- If one root of the quadratic equation x^2 - 2x + n = 0 is 3, then n is -3. -/
theorem quadratic_root (n : ℝ) (h : (3 : ℝ)^2 - 2 * 3 + n = 0) : n = -3 :=
sorry

end quadratic_root_l827_82706


namespace choose_three_cards_of_different_suits_l827_82727

/-- The number of ways to choose 3 cards from a standard deck of 52 cards,
if all three cards must be of different suits -/
theorem choose_three_cards_of_different_suits :
  let n := 4
  let r := 3
  let suits_combinations := Nat.choose n r
  let cards_per_suit := 13
  let total_ways := suits_combinations * (cards_per_suit ^ r)
  total_ways = 8788 :=
by
  sorry

end choose_three_cards_of_different_suits_l827_82727


namespace intersection_complement_l827_82721

def U : Set ℝ := Set.univ
def A : Set ℝ := { x | x^2 < 1 }
def B : Set ℝ := { x | x^2 - 2 * x > 0 }

theorem intersection_complement (A B : Set ℝ) : 
  (A ∩ (U \ B)) = { x | 0 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_complement_l827_82721


namespace c_share_l827_82773

theorem c_share (S : ℝ) (b_share_per_rs c_share_per_rs : ℝ)
  (h1 : S = 246)
  (h2 : b_share_per_rs = 0.65)
  (h3 : c_share_per_rs = 0.40) :
  (c_share_per_rs * S) = 98.40 :=
by sorry

end c_share_l827_82773


namespace friends_meet_first_time_at_4pm_l827_82771

def lcm_four_times (a b c d : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) (Nat.lcm c d)

def first_meeting_time (start_time_minutes: ℕ) (lap_anna lap_stephanie lap_james lap_carlos: ℕ) : ℕ :=
  start_time_minutes + lcm_four_times lap_anna lap_stephanie lap_james lap_carlos

theorem friends_meet_first_time_at_4pm :
  first_meeting_time 600 5 8 9 12 = 960 :=
by
  -- where 600 represents 10:00 AM in minutes since midnight and 960 represents 4:00 PM
  sorry

end friends_meet_first_time_at_4pm_l827_82771


namespace clock_shows_l827_82717

-- Definitions for the hands and their positions
variables {A B C : ℕ} -- Representing hands A, B, and C as natural numbers for simplicity

-- Conditions based on the problem description:
-- 1. Hands A and B point exactly at the hour markers.
-- 2. Hand C is slightly off from an hour marker.
axiom hand_A_hour_marker : A % 12 = A
axiom hand_B_hour_marker : B % 12 = B
axiom hand_C_slightly_off : C % 12 ≠ C

-- Theorem stating that given these conditions, the clock shows the time 4:50
theorem clock_shows (h1: A % 12 = A) (h2: B % 12 = B) (h3: C % 12 ≠ C) : A = 50 ∧ B = 12 ∧ C = 4 :=
sorry

end clock_shows_l827_82717


namespace find_number_l827_82760

theorem find_number (x : ℝ) (h : 0.30 * x = 108.0) : x = 360 := 
sorry

end find_number_l827_82760


namespace order_of_magnitude_l827_82776

noncomputable def a : Real := 70.3
noncomputable def b : Real := 70.2
noncomputable def c : Real := Real.log 0.3

theorem order_of_magnitude : a > b ∧ b > c :=
by
  sorry

end order_of_magnitude_l827_82776


namespace find_b_l827_82737

-- Definitions for the conditions
variables (a b c d : ℝ)
def four_segments_proportional := a / b = c / d

theorem find_b (h1: a = 3) (h2: d = 4) (h3: c = 6) (h4: four_segments_proportional a b c d) : b = 2 :=
by
  sorry

end find_b_l827_82737


namespace find_angle_AOD_l827_82705

noncomputable def angleAOD (x : ℝ) : ℝ :=
4 * x

theorem find_angle_AOD (x : ℝ) (h1 : 4 * x = 180) : angleAOD x = 135 :=
by
  -- x = 45
  have h2 : x = 45 := by linarith

  -- angleAOD 45 = 4 * 45 = 135
  rw [angleAOD, h2]
  norm_num
  sorry

end find_angle_AOD_l827_82705


namespace complex_simplify_l827_82716

theorem complex_simplify :
  10.25 * Real.sqrt 6 * Complex.exp (Complex.I * 160 * Real.pi / 180)
  / (Real.sqrt 3 * Complex.exp (Complex.I * 40 * Real.pi / 180))
  = (-Real.sqrt 2 / 2) + Complex.I * (Real.sqrt 6 / 2) := by
  sorry

end complex_simplify_l827_82716


namespace car_highway_miles_per_tankful_l827_82763

-- Defining conditions as per given problem
def city_miles_per_tank : ℕ := 336
def city_miles_per_gallon : ℕ := 8
def difference_miles_per_gallon : ℕ := 3
def highway_miles_per_gallon := city_miles_per_gallon + difference_miles_per_gallon
def tank_size := city_miles_per_tank / city_miles_per_gallon
def highway_miles_per_tank := highway_miles_per_gallon * tank_size

-- Theorem statement to prove
theorem car_highway_miles_per_tankful :
  highway_miles_per_tank = 462 :=
sorry

end car_highway_miles_per_tankful_l827_82763


namespace angle_BMC_not_obtuse_angle_BAC_is_120_l827_82725

theorem angle_BMC_not_obtuse (α β γ : ℝ) (h : α + β + γ = 180) :
  0 < 90 - α / 2 ∧ 90 - α / 2 < 90 :=
sorry

theorem angle_BAC_is_120 (α β γ : ℝ) (h1 : α + β + γ = 180)
  (h2 : 90 - α / 2 = α / 2) : α = 120 :=
sorry

end angle_BMC_not_obtuse_angle_BAC_is_120_l827_82725


namespace B_won_third_four_times_l827_82793

noncomputable def first_place := 5
noncomputable def second_place := 2
noncomputable def third_place := 1

structure ContestantScores :=
  (A_score : ℕ)
  (B_score : ℕ)
  (C_score : ℕ)

def competition_results (A B C : ContestantScores) (a b c : ℕ) : Prop :=
  A.A_score = 26 ∧ B.B_score = 11 ∧ C.C_score = 11 ∧ 1 = 1 ∧ -- B won first place once is synonymous to holding true
  a > b ∧ b > c ∧ a = 5 ∧ b = 2 ∧ c = 1

theorem B_won_third_four_times :
  ∃ (A B C : ContestantScores), competition_results A B C first_place second_place third_place → 
  B.B_score = 4 * third_place + first_place := 
sorry

end B_won_third_four_times_l827_82793


namespace find_other_number_l827_82778

theorem find_other_number 
  (a b : ℕ)
  (h_lcm : Nat.lcm a b = 5040)
  (h_gcd : Nat.gcd a b = 24)
  (h_a : a = 240) : b = 504 := by
  sorry

end find_other_number_l827_82778


namespace peter_work_days_l827_82770

variable (W M P : ℝ)
variable (h1 : M + P = W / 20) -- Combined rate of Matt and Peter
variable (h2 : 12 * (W / 20) + 14 * P = W) -- Work done by Matt and Peter for 12 days + Peter's remaining work

theorem peter_work_days :
  P = W / 35 :=
by
  sorry

end peter_work_days_l827_82770


namespace run_time_is_48_minutes_l827_82755

noncomputable def cycling_speed : ℚ := 5 / 2
noncomputable def running_speed : ℚ := cycling_speed * 0.5
noncomputable def walking_speed : ℚ := running_speed * 0.5

theorem run_time_is_48_minutes (d : ℚ) (h : (d / cycling_speed) + (d / walking_speed) = 2) : 
  (60 * d / running_speed) = 48 :=
by
  sorry

end run_time_is_48_minutes_l827_82755


namespace square_remainder_is_square_l827_82704

theorem square_remainder_is_square (a : ℤ) : ∃ b : ℕ, (a^2 % 16 = b) ∧ (∃ c : ℕ, b = c^2) :=
by
  sorry

end square_remainder_is_square_l827_82704


namespace typist_original_salary_l827_82782

theorem typist_original_salary (x : ℝ) (h : (x * 1.10 * 0.95 = 4180)) : x = 4000 :=
by sorry

end typist_original_salary_l827_82782


namespace min_value_of_x_plus_y_l827_82731

open Real

theorem min_value_of_x_plus_y (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0)
  (a : ℝ × ℝ := (1 - x, 4)) (b : ℝ × ℝ := (x, -y))
  (h₃ : ∃ k : ℝ, k ≠ 0 ∧ a = (k * b.1, k * b.2)) :
  x + y = 9 :=
by
  sorry

end min_value_of_x_plus_y_l827_82731


namespace selling_price_750_max_daily_profit_l827_82732

noncomputable def profit (x : ℝ) : ℝ :=
  (x - 10) * (-10 * x + 300)

theorem selling_price_750 (x : ℝ) : profit x = 750 ↔ (x = 15 ∨ x = 25) :=
by sorry

theorem max_daily_profit : (∀ x : ℝ, profit x ≤ 1000) ∧ (profit 20 = 1000) :=
by sorry

end selling_price_750_max_daily_profit_l827_82732


namespace strudel_price_l827_82772

def initial_price := 80
def first_increment (P0 : ℕ) := P0 * 3 / 2
def second_increment (P1 : ℕ) := P1 * 3 / 2
def final_price (P2 : ℕ) := P2 / 2

theorem strudel_price (P0 : ℕ) (P1 : ℕ) (P2 : ℕ) (Pf : ℕ)
  (h0 : P0 = initial_price)
  (h1 : P1 = first_increment P0)
  (h2 : P2 = second_increment P1)
  (hf : Pf = final_price P2) :
  Pf = 90 :=
sorry

end strudel_price_l827_82772


namespace closest_point_on_line_l827_82746

structure Point (α : Type) :=
(x : α) (y : α) (z : α)

def line (s : ℚ) : Point ℚ :=
⟨3 + s, 2 - 3 * s, 4 * s⟩

def distance (p1 p2 : Point ℚ) : ℚ :=
(p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

def closestPoint : Point ℚ := ⟨37/17, 74/17, -56/17⟩

def givenPoint : Point ℚ := ⟨1, 4, -2⟩

theorem closest_point_on_line :
  ∃ s : ℚ, line s = closestPoint ∧ 
           ∀ t : ℚ, distance closestPoint givenPoint ≤ distance (line t) givenPoint :=
by
  sorry

end closest_point_on_line_l827_82746


namespace foci_and_directrices_of_ellipse_l827_82756

noncomputable def parametricEllipse
    (θ : ℝ) : ℝ × ℝ :=
  (3 * Real.cos θ + 1, 4 * Real.sin θ)

theorem foci_and_directrices_of_ellipse :
  (∀ θ : ℝ, parametricEllipse θ = (x, y)) →
  (∃ (f1 f2 : ℝ × ℝ) (d1 d2 : ℝ → Prop),
    f1 = (1, Real.sqrt 7) ∧
    f2 = (1, -Real.sqrt 7) ∧
    d1 = fun x => x = 1 + 9 / Real.sqrt 7 ∧
    d2 = fun x => x = 1 - 9 / Real.sqrt 7) := sorry

end foci_and_directrices_of_ellipse_l827_82756


namespace four_digit_numbers_divisible_by_11_with_sum_of_digits_11_l827_82719

noncomputable def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

noncomputable def is_divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

noncomputable def sum_of_digits_is_11 (n : ℕ) : Prop := 
  let d1 := n / 1000
  let r1 := n % 1000
  let d2 := r1 / 100
  let r2 := r1 % 100
  let d3 := r2 / 10
  let d4 := r2 % 10
  d1 + d2 + d3 + d4 = 11

theorem four_digit_numbers_divisible_by_11_with_sum_of_digits_11
  (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : is_divisible_by_11 n)
  (h3 : sum_of_digits_is_11 n) : 
  n = 2090 ∨ n = 3080 ∨ n = 4070 ∨ n = 5060 ∨ n = 6050 ∨ n = 7040 ∨ n = 8030 ∨ n = 9020 :=
sorry

end four_digit_numbers_divisible_by_11_with_sum_of_digits_11_l827_82719


namespace seating_arrangements_count_is_134_l827_82707

theorem seating_arrangements_count_is_134 (front_row_seats : ℕ) (back_row_seats : ℕ) (valid_arrangements_with_no_next_to_each_other : ℕ) : 
  front_row_seats = 6 → back_row_seats = 7 → valid_arrangements_with_no_next_to_each_other = 134 :=
by
  intros h1 h2
  sorry

end seating_arrangements_count_is_134_l827_82707


namespace ludvik_favorite_number_l827_82747

variable (a b : ℕ)
variable (ℓ : ℝ)

theorem ludvik_favorite_number (h1 : 2 * a = (b + 12) * ℓ)
(h2 : a - 42 = (b / 2) * ℓ) : ℓ = 7 :=
sorry

end ludvik_favorite_number_l827_82747


namespace no_common_perfect_squares_l827_82789

theorem no_common_perfect_squares (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  ¬ (∃ m n : ℕ, a^2 + 4 * b = m^2 ∧ b^2 + 4 * a = n^2) :=
by
  sorry

end no_common_perfect_squares_l827_82789


namespace range_tan_squared_plus_tan_plus_one_l827_82715

theorem range_tan_squared_plus_tan_plus_one :
  (∀ y, ∃ x : ℝ, x ≠ (k : ℤ) * Real.pi + Real.pi / 2 → y = Real.tan x ^ 2 + Real.tan x + 1) ↔ 
  ∀ y, y ∈ Set.Ici (3 / 4) :=
sorry

end range_tan_squared_plus_tan_plus_one_l827_82715


namespace geometric_sequence_sum_l827_82774

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_a1 : a 1 = 3)
  (h_sum_first_three : a 1 + a 2 + a 3 = 21) :
  a 4 + a 5 + a 6 = 168 := 
sorry

end geometric_sequence_sum_l827_82774


namespace no_equal_numbers_from_19_and_98_l827_82745

theorem no_equal_numbers_from_19_and_98 :
  ¬ (∃ s : ℕ, ∃ (a b : ℕ → ℕ), 
       (a 0 = 19) ∧ (b 0 = 98) ∧
       (∀ k, a (k + 1) = a k * a k ∨ a (k + 1) = a k + 1) ∧
       (∀ k, b (k + 1) = b k * b k ∨ b (k + 1) = b k + 1) ∧
       a s = b s) :=
sorry

end no_equal_numbers_from_19_and_98_l827_82745


namespace volume_of_fifth_section_l827_82785

theorem volume_of_fifth_section
  (a : ℕ → ℚ)
  (h_arith_seq : ∀ n, a (n + 1) - a n = a 1 - a 0)  -- Arithmetic sequence constraint
  (h_sum_top_four : a 0 + a 1 + a 2 + a 3 = 3)  -- Sum of the top four sections
  (h_sum_bottom_three : a 6 + a 7 + a 8 = 4)  -- Sum of the bottom three sections
  : a 4 = 67 / 66 := sorry

end volume_of_fifth_section_l827_82785


namespace interior_angle_of_regular_hexagon_l827_82762

theorem interior_angle_of_regular_hexagon : 
  ∀ (n : ℕ), n = 6 → (∃ (x : ℝ), x = ((n - 2) * 180) / n) → x = 120 :=
by
  intros n hn hx
  sorry

end interior_angle_of_regular_hexagon_l827_82762
