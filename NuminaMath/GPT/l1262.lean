import Mathlib

namespace height_difference_percentage_l1262_126211

theorem height_difference_percentage (q p : ℝ) (h : p = 0.6 * q) : (q - p) / p * 100 = 66.67 := 
by
  sorry

end height_difference_percentage_l1262_126211


namespace sale_first_month_l1262_126231

-- Declaration of all constant sales amounts in rupees
def sale_second_month : ℕ := 6927
def sale_third_month : ℕ := 6855
def sale_fourth_month : ℕ := 7230
def sale_fifth_month : ℕ := 6562
def sale_sixth_month : ℕ := 6791
def average_required : ℕ := 6800
def months : ℕ := 6

-- Total sales computed from the average sale requirement
def total_sales_needed : ℕ := months * average_required

-- The sum of sales for the second to sixth months
def total_sales_last_five_months := sale_second_month + sale_third_month + sale_fourth_month + sale_fifth_month + sale_sixth_month

-- Prove the sales in the first month given the conditions
theorem sale_first_month :
  total_sales_needed - total_sales_last_five_months = 6435 :=
by
  sorry

end sale_first_month_l1262_126231


namespace number_subsets_property_p_l1262_126241

def has_property_p (a b : ℕ) : Prop := 17 ∣ (a + b)

noncomputable def num_subsets_with_property_p : ℕ :=
  -- sorry, put computation result here using the steps above but skipping actual computation for brevity
  3928

theorem number_subsets_property_p :
  num_subsets_with_property_p = 3928 := sorry

end number_subsets_property_p_l1262_126241


namespace line_parabola_one_intersection_not_tangent_l1262_126268

theorem line_parabola_one_intersection_not_tangent {A B C D : ℝ} (h: ∀ x : ℝ, ((A * x ^ 2 + B * x + C) = D) → False) :
  ¬ ∃ x : ℝ, (A * x ^ 2 + B * x + C) = D ∧ 2 * x * A + B = 0 := sorry

end line_parabola_one_intersection_not_tangent_l1262_126268


namespace greatest_sum_solution_l1262_126203

theorem greatest_sum_solution (x y : ℤ) (h : x^2 + y^2 = 20) : 
  x + y ≤ 6 :=
sorry

end greatest_sum_solution_l1262_126203


namespace molecular_weight_CaO_l1262_126207

theorem molecular_weight_CaO (m : ℕ -> ℝ) (h : m 7 = 392) : m 1 = 56 :=
sorry

end molecular_weight_CaO_l1262_126207


namespace chapters_per_day_l1262_126234

theorem chapters_per_day (total_pages : ℕ) (total_chapters : ℕ) (total_days : ℕ)
  (h1 : total_pages = 193)
  (h2 : total_chapters = 15)
  (h3 : total_days = 660) :
  (total_chapters : ℝ) / total_days = 0.0227 :=
by 
  sorry

end chapters_per_day_l1262_126234


namespace pizza_pasta_cost_difference_l1262_126287

variable (x y z : ℝ)
variable (A1 : 2 * x + 3 * y + 4 * z = 53)
variable (A2 : 5 * x + 6 * y + 7 * z = 107)

theorem pizza_pasta_cost_difference :
  x - z = 1 :=
by
  sorry

end pizza_pasta_cost_difference_l1262_126287


namespace maximum_distance_area_of_ring_l1262_126277

def num_radars : ℕ := 9
def radar_radius : ℝ := 37
def ring_width : ℝ := 24

theorem maximum_distance (θ : ℝ) (hθ : θ = 20) 
  : (∀ d, d = radar_radius * (ring_width / 2 / (radar_radius^2 - (ring_width / 2)^2).sqrt)) →
    ( ∀ dist_from_center, dist_from_center = radar_radius / θ.sin) :=
sorry

theorem area_of_ring (θ : ℝ) (hθ : θ = 20) 
  : (∀ a, a = π * (ring_width * radar_radius * 2 / θ.tan)) →
    ( ∀ area, area = 1680 * π / θ.tan) :=
sorry

end maximum_distance_area_of_ring_l1262_126277


namespace giant_spider_leg_cross_sectional_area_l1262_126295

theorem giant_spider_leg_cross_sectional_area :
  let previous_spider_weight := 6.4
  let weight_multiplier := 2.5
  let pressure := 4
  let num_legs := 8

  let giant_spider_weight := weight_multiplier * previous_spider_weight
  let weight_per_leg := giant_spider_weight / num_legs
  let cross_sectional_area := weight_per_leg / pressure

  cross_sectional_area = 0.5 :=
by 
  sorry

end giant_spider_leg_cross_sectional_area_l1262_126295


namespace min_perimeter_lateral_face_l1262_126202

theorem min_perimeter_lateral_face (x h : ℝ) (V : ℝ) (P : ℝ): 
  (x > 0) → (h > 0) → (V = 4) → (V = x^2 * h) → 
  (∀ y : ℝ, y > 0 → 2*y + 2 * (4 / y^2) ≥ P) → P = 6 := 
by
  intro x_pos h_pos volume_eq volume_expr min_condition
  sorry

end min_perimeter_lateral_face_l1262_126202


namespace average_speed_palindrome_l1262_126238

theorem average_speed_palindrome :
  ∀ (initial_odometer final_odometer : ℕ) (hours : ℕ),
  initial_odometer = 123321 →
  final_odometer = 124421 →
  hours = 4 →
  (final_odometer - initial_odometer) / hours = 275 :=
by
  intros initial_odometer final_odometer hours h1 h2 h3
  sorry

end average_speed_palindrome_l1262_126238


namespace smallest_twice_perfect_square_three_times_perfect_cube_l1262_126254

theorem smallest_twice_perfect_square_three_times_perfect_cube :
  ∃ n : ℕ, (∃ k : ℕ, n = 2 * k^2) ∧ (∃ m : ℕ, n = 3 * m^3) ∧ n = 648 :=
by
  sorry

end smallest_twice_perfect_square_three_times_perfect_cube_l1262_126254


namespace koi_fish_in_pond_l1262_126253

theorem koi_fish_in_pond:
  ∃ k : ℕ, 2 * k - 14 = 64 ∧ k = 39 := sorry

end koi_fish_in_pond_l1262_126253


namespace right_triangle_area_l1262_126272

theorem right_triangle_area (a b c : ℝ) (h : c = 5) (h1 : a = 3) (h2 : c^2 = a^2 + b^2) : 
  1 / 2 * a * b = 6 :=
by
  sorry

end right_triangle_area_l1262_126272


namespace Edmund_earns_64_dollars_l1262_126275

-- Conditions
def chores_per_week : Nat := 12
def pay_per_extra_chore : Nat := 2
def chores_per_day : Nat := 4
def weeks : Nat := 2
def days_per_week : Nat := 7

-- Goal
theorem Edmund_earns_64_dollars :
  let total_chores_without_extra := chores_per_week * weeks
  let total_chores_with_extra := chores_per_day * (days_per_week * weeks)
  let extra_chores := total_chores_with_extra - total_chores_without_extra
  let earnings := pay_per_extra_chore * extra_chores
  earnings = 64 :=
by
  sorry

end Edmund_earns_64_dollars_l1262_126275


namespace cube_surface_area_l1262_126297

/-- Given a cube with a space diagonal of 6, the surface area is 72. -/
theorem cube_surface_area (s : ℝ) (h : s * Real.sqrt 3 = 6) : 6 * s^2 = 72 :=
by
  sorry

end cube_surface_area_l1262_126297


namespace valid_interval_for_a_l1262_126212

theorem valid_interval_for_a (a : ℝ) :
  (6 - 3 * a > 0) ∧ (a > 0) ∧ (3 * a^2 + a - 2 ≥ 0) ↔ (2 / 3 ≤ a ∧ a < 2 ∧ a ≠ 5 / 3) :=
by
  sorry

end valid_interval_for_a_l1262_126212


namespace money_left_l1262_126250

theorem money_left 
  (salary : ℝ)
  (spent_on_food : ℝ)
  (spent_on_rent : ℝ)
  (spent_on_clothes : ℝ)
  (total_spent : ℝ)
  (money_left : ℝ)
  (h_salary : salary = 170000)
  (h_food : spent_on_food = salary * (1 / 5))
  (h_rent : spent_on_rent = salary * (1 / 10))
  (h_clothes : spent_on_clothes = salary * (3 / 5))
  (h_total_spent : total_spent = spent_on_food + spent_on_rent + spent_on_clothes)
  (h_money_left : money_left = salary - total_spent) :
  money_left = 17000 :=
by
  sorry

end money_left_l1262_126250


namespace carbon_paper_count_l1262_126274

theorem carbon_paper_count (x : ℕ) (sheets : ℕ) (copies : ℕ) (h1 : sheets = 3) (h2 : copies = 2) :
  x = 1 :=
sorry

end carbon_paper_count_l1262_126274


namespace discriminant_of_quadratic_5x2_minus_2x_minus_7_l1262_126286

def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b ^ 2 - 4 * a * c

theorem discriminant_of_quadratic_5x2_minus_2x_minus_7 :
  quadratic_discriminant 5 (-2) (-7) = 144 :=
by
  sorry

end discriminant_of_quadratic_5x2_minus_2x_minus_7_l1262_126286


namespace margaret_mean_score_l1262_126247

def sum_of_scores (scores : List ℤ) : ℤ :=
  scores.sum

def mean_score (total_score : ℤ) (count : ℕ) : ℚ :=
  total_score / count

theorem margaret_mean_score :
  let scores := [85, 88, 90, 92, 94, 96, 100]
  let cyprian_mean := 92
  let cyprian_count := 4
  let total_score := sum_of_scores scores
  let cyprian_total_score := cyprian_mean * cyprian_count
  let margaret_total_score := total_score - cyprian_total_score
  let margaret_mean := mean_score margaret_total_score 3
  margaret_mean = 92.33 :=
by
  sorry

end margaret_mean_score_l1262_126247


namespace buses_more_than_vans_l1262_126257

-- Definitions based on conditions
def vans : Float := 6.0
def buses : Float := 8.0
def people_per_van : Float := 6.0
def people_per_bus : Float := 18.0

-- Calculate total people in vans and buses
def total_people_vans : Float := vans * people_per_van
def total_people_buses : Float := buses * people_per_bus

-- Prove the difference
theorem buses_more_than_vans : total_people_buses - total_people_vans = 108.0 :=
by
  sorry

end buses_more_than_vans_l1262_126257


namespace find_cost_price_of_clock_l1262_126294

namespace ClockCost

variable (C : ℝ)

def cost_price_each_clock (n : ℝ) (gain1 : ℝ) (gain2 : ℝ) (uniform_gain : ℝ) (price_difference : ℝ) :=
  let selling_price1 := 40 * C * (1 + gain1)
  let selling_price2 := 50 * C * (1 + gain2)
  let uniform_selling_price := n * C * (1 + uniform_gain)
  selling_price1 + selling_price2 - uniform_selling_price = price_difference

theorem find_cost_price_of_clock (C : ℝ) (h : cost_price_each_clock C 90 0.10 0.20 0.15 40) : C = 80 :=
  sorry

end ClockCost

end find_cost_price_of_clock_l1262_126294


namespace comparison_of_f_values_l1262_126220

noncomputable def f (x : ℝ) := Real.cos x - x

theorem comparison_of_f_values :
  f (8 * Real.pi / 9) > f Real.pi ∧ f Real.pi > f (10 * Real.pi / 9) :=
by
  sorry

end comparison_of_f_values_l1262_126220


namespace eighth_odd_multiple_of_5_is_75_l1262_126264

theorem eighth_odd_multiple_of_5_is_75 : ∃ n : ℕ, (n > 0 ∧ n % 2 = 1 ∧ n % 5 = 0 ∧ ∃ k : ℕ, k = 8 ∧ n = 10 * k - 5) :=
  sorry

end eighth_odd_multiple_of_5_is_75_l1262_126264


namespace ratio_fourth_to_sixth_l1262_126226

-- Definitions from the conditions
def fourth_level_students := 40
def sixth_level_students := 40
def seventh_level_students := 2 * fourth_level_students

-- Statement to prove
theorem ratio_fourth_to_sixth : 
  fourth_level_students / sixth_level_students = 1 :=
by
  -- Proof skipped
  sorry

end ratio_fourth_to_sixth_l1262_126226


namespace max_sum_of_three_integers_with_product_24_l1262_126289

theorem max_sum_of_three_integers_with_product_24 : ∃ (a b c : ℤ), (a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 24 ∧ a + b + c = 15) :=
by
  sorry

end max_sum_of_three_integers_with_product_24_l1262_126289


namespace range_of_m_l1262_126248

theorem range_of_m (m : ℝ) (x : ℝ) (hp : (x + 2) * (x - 10) ≤ 0)
  (hq : x^2 - 2 * x + 1 - m^2 ≤ 0) (hm : m > 0) : 0 < m ∧ m ≤ 3 :=
sorry

end range_of_m_l1262_126248


namespace find_smallest_k_l1262_126299

variable (k : ℕ)

theorem find_smallest_k :
  (∀ a : ℝ, 0 ≤ a ∧ a ≤ 1 → (∀ n : ℕ, n > 0 → a^k * (1-a)^n < 1 / (n+1)^3)) ↔ k = 4 :=
sorry

end find_smallest_k_l1262_126299


namespace perfect_square_trinomial_l1262_126293

theorem perfect_square_trinomial (m : ℝ) :
  ∃ (a : ℝ), (∀ (x : ℝ), x^2 - 2*(m-3)*x + 16 = (x - a)^2) ↔ (m = 7 ∨ m = -1) := by
  sorry

end perfect_square_trinomial_l1262_126293


namespace number_of_smaller_cubes_l1262_126239

theorem number_of_smaller_cubes (N : ℕ) : 
  (∀ a : ℕ, ∃ n : ℕ, n * a^3 = 125) ∧
  (∀ b : ℕ, b ≤ 5 → ∃ m : ℕ, m * b^3 ≤ 125) ∧
  (∃ x y : ℕ, x ≠ y) → 
  N = 118 :=
sorry

end number_of_smaller_cubes_l1262_126239


namespace union_S_T_l1262_126276

def S : Set ℝ := { x | 3 < x ∧ x ≤ 6 }
def T : Set ℝ := { x | x^2 - 4*x - 5 ≤ 0 }

theorem union_S_T : S ∪ T = { x | -1 ≤ x ∧ x ≤ 6 } := 
by 
  sorry

end union_S_T_l1262_126276


namespace solve_equation_solve_inequality_system_l1262_126213

theorem solve_equation :
  ∃ x, 2 * x^2 - 4 * x - 1 = 0 ∧ (x = (2 + Real.sqrt 6) / 2 ∨ x = (2 - Real.sqrt 6) / 2) :=
sorry

theorem solve_inequality_system : 
  ∀ x, (2 * x + 3 > 1 → -1 < x) ∧
       (x - 2 ≤ (1 / 2) * (x + 2) → x ≤ 6) ∧ 
       (2 * x + 3 > 1 ∧ x - 2 ≤ (1 / 2) * (x + 2) ↔ (-1 < x ∧ x ≤ 6)) :=
sorry

end solve_equation_solve_inequality_system_l1262_126213


namespace sale_in_fifth_month_l1262_126201

theorem sale_in_fifth_month 
    (a1 a2 a3 a4 a6 : ℕ) 
    (avg_sale : ℕ)
    (H_avg : avg_sale = 8500)
    (H_a1 : a1 = 8435) 
    (H_a2 : a2 = 8927) 
    (H_a3 : a3 = 8855) 
    (H_a4 : a4 = 9230) 
    (H_a6 : a6 = 6991) : 
    ∃ a5 : ℕ, (a1 + a2 + a3 + a4 + a5 + a6) / 6 = avg_sale ∧ a5 = 8562 := 
by
    sorry

end sale_in_fifth_month_l1262_126201


namespace find_b_l1262_126256

-- Define the conditions of the equations
def condition_1 (x y a : ℝ) : Prop := x * Real.cos a + y * Real.sin a + 3 ≤ 0
def condition_2 (x y b : ℝ) : Prop := x^2 + y^2 + 8 * x - 4 * y - b^2 + 6 * b + 11 = 0

-- Define the proof problem
theorem find_b (b : ℝ) :
  (∀ a x y, condition_1 x y a → condition_2 x y b) →
  b ∈ Set.Iic (-2 * Real.sqrt 5) ∪ Set.Ici (6 + 2 * Real.sqrt 5) :=
by
  sorry

end find_b_l1262_126256


namespace loss_percentage_25_l1262_126292

variable (C S : ℝ)
variable (h : 15 * C = 20 * S)

theorem loss_percentage_25 (h : 15 * C = 20 * S) : (C - S) / C * 100 = 25 := by
  sorry

end loss_percentage_25_l1262_126292


namespace fourth_square_area_l1262_126260

theorem fourth_square_area (AB BC CD AD AC : ℝ) (h1 : AB^2 = 25) (h2 : BC^2 = 49) (h3 : CD^2 = 64) (h4 : AC^2 = AB^2 + BC^2)
  (h5 : AD^2 = AC^2 + CD^2) : AD^2 = 138 :=
by
  sorry

end fourth_square_area_l1262_126260


namespace inequality_three_variables_l1262_126252

theorem inequality_three_variables (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) : 
  (1/x) + (1/y) + (1/z) ≥ 9 := 
by 
  sorry

end inequality_three_variables_l1262_126252


namespace remainder_polynomial_l1262_126204

noncomputable def p (x : ℝ) : ℝ := sorry
noncomputable def r (x : ℝ) : ℝ := x^2 + x

theorem remainder_polynomial (p : ℝ → ℝ) (r : ℝ → ℝ) :
  (p 2 = 6) ∧ (p 4 = 20) ∧ (p 6 = 42) →
  (r 2 = 2^2 + 2) ∧ (r 4 = 4^2 + 4) ∧ (r 6 = 6^2 + 6) :=
sorry

end remainder_polynomial_l1262_126204


namespace convert_108_kmph_to_mps_l1262_126225

-- Definitions and assumptions
def kmph_to_mps (speed_kmph : ℕ) : ℚ :=
  speed_kmph * (1000 / 3600)

-- Theorem statement
theorem convert_108_kmph_to_mps : kmph_to_mps 108 = 30 := 
by
  sorry

end convert_108_kmph_to_mps_l1262_126225


namespace find_m_l1262_126270

variables (a b : ℝ × ℝ) (m : ℝ)

def vectors := (a = (3, 4)) ∧ (b = (2, -1))

def perpendicular (a b : ℝ × ℝ) : Prop :=
a.1 * b.1 + a.2 * b.2 = 0

theorem find_m (h1 : vectors a b) (h2 : perpendicular (a.1 + m * b.1, a.2 + m * b.2) (a.1 - b.1, a.2 - b.2)) :
  m = 23 / 3 :=
sorry

end find_m_l1262_126270


namespace cuboid_to_cube_surface_area_l1262_126259

variable (h w l : ℝ)
variable (volume_decreases : 64 = w^3 - w^2 * h)

theorem cuboid_to_cube_surface_area 
  (h w l : ℝ) 
  (cube_condition : w = l ∧ h = w + 4)
  (volume_condition : w^2 * h - w^3 = 64) : 
  (6 * w^2 = 96) :=
by
  sorry

end cuboid_to_cube_surface_area_l1262_126259


namespace maximum_unique_numbers_in_circle_l1262_126243

theorem maximum_unique_numbers_in_circle :
  ∀ (n : ℕ) (numbers : ℕ → ℤ), n = 2023 →
  (∀ i, numbers i = numbers ((i + 1) % n) * numbers ((i + n - 1) % n)) →
  ∀ i j, numbers i = numbers j :=
by
  sorry

end maximum_unique_numbers_in_circle_l1262_126243


namespace second_machine_copies_per_minute_l1262_126271

-- Definitions based on conditions
def copies_per_minute_first := 35
def total_copies_half_hour := 3300
def time_minutes := 30

-- Theorem statement
theorem second_machine_copies_per_minute : 
  ∃ (x : ℕ), (copies_per_minute_first * time_minutes + x * time_minutes = total_copies_half_hour) ∧ (x = 75) := by
  sorry

end second_machine_copies_per_minute_l1262_126271


namespace inequality_holds_l1262_126214

theorem inequality_holds (a b c : ℝ) 
  (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a * b * c = 1) : 
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_holds_l1262_126214


namespace proof_problem_l1262_126283

def p : Prop := ∃ x : ℝ, x^2 - x + 1 ≥ 0
def q : Prop := ∀ (a b : ℝ), (a^2 < b^2) → (a < b)

theorem proof_problem (h₁ : p) (h₂ : ¬ q) : p ∧ ¬ q := by
  exact ⟨h₁, h₂⟩

end proof_problem_l1262_126283


namespace gifts_wrapped_with_third_roll_l1262_126251

def num_rolls : ℕ := 3
def num_gifts : ℕ := 12
def first_roll_gifts : ℕ := 3
def second_roll_gifts : ℕ := 5

theorem gifts_wrapped_with_third_roll : 
  first_roll_gifts + second_roll_gifts < num_gifts → 
  num_gifts - (first_roll_gifts + second_roll_gifts) = 4 := 
by
  intros h
  sorry

end gifts_wrapped_with_third_roll_l1262_126251


namespace ratio_p_r_l1262_126246

     variables (p q r s : ℚ)

     -- Given conditions
     def ratio_p_q := p / q = 3 / 5
     def ratio_r_s := r / s = 5 / 4
     def ratio_s_q := s / q = 1 / 3

     -- Statement to be proved
     theorem ratio_p_r 
       (h1 : ratio_p_q p q)
       (h2 : ratio_r_s r s) 
       (h3 : ratio_s_q s q) : 
       p / r = 36 / 25 :=
     sorry
     
end ratio_p_r_l1262_126246


namespace least_sugar_l1262_126279

theorem least_sugar (f s : ℚ) (h1 : f ≥ 10 + 3 * s / 4) (h2 : f ≤ 3 * s) :
  s ≥ 40 / 9 :=
  sorry

end least_sugar_l1262_126279


namespace inequality_not_always_true_l1262_126210

theorem inequality_not_always_true {a b c : ℝ}
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) (h₄ : c ≠ 0) : ¬ ∀ c : ℝ, (a / c > b / c) :=
by
  sorry

end inequality_not_always_true_l1262_126210


namespace license_plate_difference_l1262_126232

theorem license_plate_difference : 
    let alpha_plates := 26^4 * 10^4
    let beta_plates := 26^3 * 10^4
    alpha_plates - beta_plates = 10^4 * 26^3 * 25 := 
by sorry

end license_plate_difference_l1262_126232


namespace quadratic_distinct_real_roots_l1262_126205

-- Defining the main hypothesis
theorem quadratic_distinct_real_roots (k : ℝ) :
  (k < 4 / 3) ∧ (k ≠ 1) ↔ (∀ x : ℂ, ((k-1) * x^2 - 2 * x + 3 = 0) → ∃ x₁ x₂ : ℂ, x₁ ≠ x₂ ∧ ((k-1) * x₁ ^ 2 - 2 * x₁ + 3 = 0) ∧ ((k-1) * x₂ ^ 2 - 2 * x₂ + 3 = 0)) := by
sorry

end quadratic_distinct_real_roots_l1262_126205


namespace estimate_num_2016_digit_squares_l1262_126263

noncomputable def num_estimate_2016_digit_squares : ℕ := 2016

theorem estimate_num_2016_digit_squares :
  let t1 := (10 ^ (2016 / 2) - 10 ^ (2015 / 2) - 1)
  let t2 := (2017 ^ 10)
  let result := t1 / t2
  t1 > 10 ^ 1000 → 
  result > 10 ^ 900 →
  result == num_estimate_2016_digit_squares :=
by
  intros
  sorry

end estimate_num_2016_digit_squares_l1262_126263


namespace valid_integer_values_of_x_l1262_126227

theorem valid_integer_values_of_x (x : ℤ) 
  (h1 : 3 < x) (h2 : x < 10)
  (h3 : 5 < x) (h4 : x < 18)
  (h5 : -2 < x) (h6 : x < 9)
  (h7 : 0 < x) (h8 : x < 8) 
  (h9 : x + 1 < 9) : x = 6 ∨ x = 7 :=
by
  sorry

end valid_integer_values_of_x_l1262_126227


namespace toms_friend_decks_l1262_126218

theorem toms_friend_decks
  (cost_per_deck : ℕ)
  (tom_decks : ℕ)
  (total_spent : ℕ)
  (h1 : cost_per_deck = 8)
  (h2 : tom_decks = 3)
  (h3 : total_spent = 64) :
  (total_spent - tom_decks * cost_per_deck) / cost_per_deck = 5 := by
  sorry

end toms_friend_decks_l1262_126218


namespace polar_coordinates_of_2_neg2_l1262_126216

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  (ρ, θ)

theorem polar_coordinates_of_2_neg2 :
  polar_coordinates 2 (-2) = (2 * Real.sqrt 2, -Real.pi / 4) :=
by
  sorry

end polar_coordinates_of_2_neg2_l1262_126216


namespace age_problem_l1262_126249

theorem age_problem (A B : ℕ) 
  (h1 : A + 10 = 2 * (B - 10))
  (h2 : A = B + 12) :
  B = 42 :=
sorry

end age_problem_l1262_126249


namespace peter_invested_for_3_years_l1262_126284

-- Definitions of parameters
def P : ℝ := 650
def APeter : ℝ := 815
def ADavid : ℝ := 870
def tDavid : ℝ := 4

-- Simple interest formula for Peter
def simple_interest_peter (r : ℝ) (t : ℝ) : Prop :=
  APeter = P + P * r * t

-- Simple interest formula for David
def simple_interest_david (r : ℝ) : Prop :=
  ADavid = P + P * r * tDavid

-- The main theorem to find out how many years Peter invested his money
theorem peter_invested_for_3_years : ∃ t : ℝ, (∃ r : ℝ, simple_interest_peter r t ∧ simple_interest_david r) ∧ t = 3 :=
by
  sorry

end peter_invested_for_3_years_l1262_126284


namespace num_real_a_with_int_roots_l1262_126258

theorem num_real_a_with_int_roots :
  (∃ n : ℕ, n = 15 ∧ ∀ a : ℝ, (∃ r s : ℤ, (r + s = -a) ∧ (r * s = 12 * a) → true)) :=
sorry

end num_real_a_with_int_roots_l1262_126258


namespace product_of_consecutive_numbers_l1262_126235

theorem product_of_consecutive_numbers (n : ℕ) (k : ℕ) (h₁: n * (n + 1) * (n + 2) = 210) (h₂: n + (n + 1) = 11) : k = 3 :=
by
  sorry

end product_of_consecutive_numbers_l1262_126235


namespace division_theorem_l1262_126282

variable (x : ℤ)

def dividend := 8 * x ^ 4 + 7 * x ^ 3 + 3 * x ^ 2 - 5 * x - 8
def divisor := x - 1
def quotient := 8 * x ^ 3 + 15 * x ^ 2 + 18 * x + 13
def remainder := 5

theorem division_theorem : dividend x = divisor x * quotient x + remainder := by
  sorry

end division_theorem_l1262_126282


namespace skyscraper_anniversary_l1262_126291

theorem skyscraper_anniversary (built_years_ago : ℕ) (anniversary_years : ℕ) (years_before : ℕ) :
    built_years_ago = 100 → anniversary_years = 200 → years_before = 5 → 
    (anniversary_years - years_before) - built_years_ago = 95 := by
  intros h1 h2 h3
  sorry

end skyscraper_anniversary_l1262_126291


namespace percentage_increase_after_decrease_and_increase_l1262_126261

theorem percentage_increase_after_decrease_and_increase 
  (P : ℝ) 
  (h : 0.8 * P + (x / 100) * (0.8 * P) = 1.16 * P) : 
  x = 45 :=
by
  sorry

end percentage_increase_after_decrease_and_increase_l1262_126261


namespace divisor_in_first_division_l1262_126281

theorem divisor_in_first_division
  (N : ℕ)
  (D : ℕ)
  (Q : ℕ)
  (h1 : N = 8 * D)
  (h2 : N % 5 = 4) :
  D = 3 := 
sorry

end divisor_in_first_division_l1262_126281


namespace exists_multiple_with_equal_digit_sum_l1262_126223

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_multiple_with_equal_digit_sum (k : ℕ) (h : k > 0) : 
  ∃ n : ℕ, (n % k = 0) ∧ (sum_of_digits n = sum_of_digits (n * n)) :=
sorry

end exists_multiple_with_equal_digit_sum_l1262_126223


namespace integer_expression_l1262_126280

theorem integer_expression (m : ℤ) : ∃ k : ℤ, k = (m / 3) + (m^2 / 2) + (m^3 / 6) :=
sorry

end integer_expression_l1262_126280


namespace am_minus_gm_less_than_option_D_l1262_126242

variable (c d : ℝ)
variable (hc_pos : 0 < c) (hd_pos : 0 < d) (hcd_lt : c < d)

noncomputable def am : ℝ := (c + d) / 2
noncomputable def gm : ℝ := Real.sqrt (c * d)

theorem am_minus_gm_less_than_option_D :
  (am c d - gm c d) < ((d - c) ^ 3 / (8 * c)) :=
sorry

end am_minus_gm_less_than_option_D_l1262_126242


namespace geometric_sequence_S28_l1262_126217

noncomputable def geom_sequence_sum (S : ℕ → ℝ) (a : ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, S (n * (n + 1) / 2) = a * (1 - r^n) / (1 - r)

theorem geometric_sequence_S28 {S : ℕ → ℝ} (a r : ℝ)
  (h1 : geom_sequence_sum S a r)
  (h2 : S 14 = 3)
  (h3 : 3 * S 7 = 3) :
  S 28 = 15 :=
by
  sorry

end geometric_sequence_S28_l1262_126217


namespace nth_equation_l1262_126206

theorem nth_equation (n : ℕ) : 
  n^2 + (n + 1)^2 = (n * (n + 1) + 1)^2 - (n * (n + 1))^2 :=
by
  sorry

end nth_equation_l1262_126206


namespace find_dimes_l1262_126209

-- Definitions for the conditions
def total_dollars : ℕ := 13
def dollar_bills_1 : ℕ := 2
def dollar_bills_5 : ℕ := 1
def quarters : ℕ := 13
def nickels : ℕ := 8
def pennies : ℕ := 35
def value_dollar_bill_1 : ℝ := 1.0
def value_dollar_bill_5 : ℝ := 5.0
def value_quarter : ℝ := 0.25
def value_nickel : ℝ := 0.05
def value_penny : ℝ := 0.01
def value_dime : ℝ := 0.10

-- Theorem statement
theorem find_dimes (total_dollars dollar_bills_1 dollar_bills_5 quarters nickels pennies : ℕ)
  (value_dollar_bill_1 value_dollar_bill_5 value_quarter value_nickel value_penny value_dime : ℝ) :
  (2 * value_dollar_bill_1 + 1 * value_dollar_bill_5 + 13 * value_quarter + 8 * value_nickel + 35 * value_penny) + 
  (20 * value_dime) = ↑total_dollars :=
sorry

end find_dimes_l1262_126209


namespace mike_went_to_last_year_l1262_126236

def this_year_games : ℕ := 15
def games_missed_this_year : ℕ := 41
def total_games_attended : ℕ := 54
def last_year_games : ℕ := total_games_attended - this_year_games

theorem mike_went_to_last_year :
  last_year_games = 39 :=
  by sorry

end mike_went_to_last_year_l1262_126236


namespace BowlingAlleyTotalPeople_l1262_126267

/--
There are 31 groups of people at the bowling alley.
Each group has about 6 people.
Prove that the total number of people at the bowling alley is 186.
-/
theorem BowlingAlleyTotalPeople : 
  let groups := 31
  let people_per_group := 6
  groups * people_per_group = 186 :=
by
  sorry

end BowlingAlleyTotalPeople_l1262_126267


namespace sum_of_digits_M_l1262_126244

-- Definitions
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Conditions
variables (M : ℕ)
  (h1 : M % 2 = 0)  -- M is even
  (h2 : ∀ d ∈ M.digits 10, d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 5 ∨ d = 7 ∨ d = 9)  -- Digits of M
  (h3 : sum_of_digits (2 * M) = 31)  -- Sum of digits of 2M
  (h4 : sum_of_digits (M / 2) = 28)  -- Sum of digits of M/2

-- Goal
theorem sum_of_digits_M :
  sum_of_digits M = 29 :=
sorry

end sum_of_digits_M_l1262_126244


namespace value_of_expression_l1262_126215

theorem value_of_expression (x y : ℤ) (h1 : x = 1) (h2 : y = 630) : 
  2019 * x - 3 * y - 9 = 120 := 
by
  sorry

end value_of_expression_l1262_126215


namespace r_p_q_sum_l1262_126237

theorem r_p_q_sum (t p q r : ℕ) (h1 : (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4)
    (h2 : (1 - Real.sin t) * (1 - Real.cos t) = p / q - Real.sqrt r)
    (h3 : r > 0) (h4 : p > 0) (h5 : q > 0)
    (h6 : Nat.gcd p q = 1) : r + p + q = 5 := 
sorry

end r_p_q_sum_l1262_126237


namespace electronics_weight_l1262_126255

variable (B C E : ℝ)
variable (h1 : B / (B * (4 / 7) - 8) = 2 * (B / (B * (4 / 7))))
variable (h2 : C = B * (4 / 7))
variable (h3 : E = B * (3 / 7))

theorem electronics_weight : E = 12 := by
  sorry

end electronics_weight_l1262_126255


namespace mary_should_drink_6_glasses_l1262_126230

-- Definitions based on conditions
def daily_water_goal_liters : ℚ := 1.5
def glass_capacity_ml : ℚ := 250
def liter_to_milliliters : ℚ := 1000

-- Conversion from liters to milliliters
def daily_water_goal_milliliters : ℚ := daily_water_goal_liters * liter_to_milliliters

-- Proof problem to show Mary needs 6 glasses per day
theorem mary_should_drink_6_glasses :
  daily_water_goal_milliliters / glass_capacity_ml = 6 := by
  sorry

end mary_should_drink_6_glasses_l1262_126230


namespace mapping_has_output_l1262_126266

variable (M N : Type) (f : M → N)

theorem mapping_has_output (x : M) : ∃ y : N, f x = y :=
by
  sorry

end mapping_has_output_l1262_126266


namespace tan_product_identity_l1262_126200

theorem tan_product_identity : (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 3)) = 4 + 2 * Real.sqrt 3 :=
by
  sorry

end tan_product_identity_l1262_126200


namespace incircle_area_of_triangle_l1262_126269

noncomputable def hyperbola_params : Type :=
  sorry

noncomputable def point_on_hyperbola (P : hyperbola_params) : Prop :=
  sorry

noncomputable def in_first_quadrant (P : hyperbola_params) : Prop :=
  sorry

noncomputable def distance_ratio (PF1 PF2 : ℝ) : Prop :=
  PF1 / PF2 = 4 / 3

noncomputable def distance1_is_8 (PF1 : ℝ) : Prop :=
  PF1 = 8

noncomputable def distance2_is_6 (PF2 : ℝ) : Prop :=
  PF2 = 6

noncomputable def distance_between_foci (F1F2 : ℝ) : Prop :=
  F1F2 = 10

noncomputable def incircle_area (area : ℝ) : Prop :=
  area = 4 * Real.pi

theorem incircle_area_of_triangle (P : hyperbola_params) 
  (hP : point_on_hyperbola P) 
  (h1 : in_first_quadrant P)
  (PF1 PF2 : ℝ)
  (h2 : distance_ratio PF1 PF2)
  (h3 : distance1_is_8 PF1)
  (h4 : distance2_is_6 PF2)
  (F1F2 : ℝ) 
  (h5 : distance_between_foci F1F2) :
  ∃ r : ℝ, incircle_area (Real.pi * r^2) :=
by
  sorry

end incircle_area_of_triangle_l1262_126269


namespace total_strength_college_l1262_126265

-- Defining the conditions
def C : ℕ := 500
def B : ℕ := 600
def Both : ℕ := 220

-- Declaring the theorem
theorem total_strength_college : (C + B - Both) = 880 :=
by
  -- The proof is not required, put sorry
  sorry

end total_strength_college_l1262_126265


namespace x5_plus_y5_l1262_126222

theorem x5_plus_y5 (x y : ℝ) 
  (h1 : x + y = 3) 
  (h2 : 1 / (x + y^2) + 1 / (x^2 + y) = 1 / 2) : 
  x^5 + y^5 = 252 :=
by
  -- Placeholder for the proof
  sorry

end x5_plus_y5_l1262_126222


namespace largest_possible_square_area_l1262_126240

def rectangle_length : ℕ := 9
def rectangle_width : ℕ := 6
def largest_square_side : ℕ := rectangle_width
def largest_square_area : ℕ := largest_square_side * largest_square_side

theorem largest_possible_square_area :
  largest_square_area = 36 := by
    sorry

end largest_possible_square_area_l1262_126240


namespace intersection_M_N_l1262_126278

-- Define the set M and N
def M : Set ℝ := { x | x^2 ≤ 1 }
def N : Set ℝ := {-2, 0, 1}

-- Theorem stating that the intersection of M and N is {0, 1}
theorem intersection_M_N : M ∩ N = {0, 1} :=
by
  sorry

end intersection_M_N_l1262_126278


namespace angle_A_value_sin_2B_plus_A_l1262_126233

variable (a b c : ℝ)
variable (A B C : ℝ)
variable (h1 : a = 3)
variable (h2 : b = 2 * Real.sqrt 2)
variable (triangle_condition : b / (a + c) = 1 - (Real.sin C / (Real.sin A + Real.sin B)))

theorem angle_A_value : A = Real.pi / 3 :=
sorry

theorem sin_2B_plus_A (hA : A = Real.pi / 3) : 
  Real.sin (2 * B + A) = (2 * Real.sqrt 2 - Real.sqrt 3) / 6 :=
sorry

end angle_A_value_sin_2B_plus_A_l1262_126233


namespace range_of_a_l1262_126219

noncomputable def g (a x : ℝ) : ℝ := x ^ 2 - 2 * a * x + 3

theorem range_of_a 
  (h_mono_inc : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → g a x1 ≤ g a x2)
  (h_nonneg : ∀ x : ℝ, -1 < x ∧ x < 1 → 0 ≤ g a x) :
  (-2 : ℝ) ≤ a ∧ a ≤ -1 := by
  sorry

end range_of_a_l1262_126219


namespace solution_set_of_inequalities_l1262_126208

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l1262_126208


namespace graph_not_pass_second_quadrant_l1262_126290

theorem graph_not_pass_second_quadrant (a b : ℝ) (h1 : a > 1) (h2 : b < -1) :
  ¬ ∃ (x : ℝ), y = a^x + b ∧ x < 0 ∧ y > 0 :=
by
  sorry

end graph_not_pass_second_quadrant_l1262_126290


namespace possible_values_of_a_l1262_126228

theorem possible_values_of_a (x y a : ℝ)
  (h1 : x + y = a)
  (h2 : x^3 + y^3 = a)
  (h3 : x^5 + y^5 = a) :
  a = 0 ∨ a = 1 ∨ a = -1 ∨ a = 2 ∨ a = -2 :=
sorry

end possible_values_of_a_l1262_126228


namespace ratio_of_parts_l1262_126298

theorem ratio_of_parts (N : ℝ) (h1 : (1/4) * (2/5) * N = 14) (h2 : 0.40 * N = 168) : (2/5) * N / N = 1 / 2.5 :=
by
  sorry

end ratio_of_parts_l1262_126298


namespace solve_for_x_l1262_126262

theorem solve_for_x :
  exists x : ℝ, 11.98 * 11.98 + 11.98 * x + 0.02 * 0.02 = (11.98 + 0.02) ^ 2 ∧ x = 0.04 :=
by
  sorry

end solve_for_x_l1262_126262


namespace no_real_solutions_l1262_126221

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (x - 3 * x + 8)^2 + 4 = -2 * |x| :=
by
  sorry

end no_real_solutions_l1262_126221


namespace movie_of_the_year_condition_l1262_126288

theorem movie_of_the_year_condition (total_lists : ℕ) (fraction : ℚ) (num_lists : ℕ) 
  (h1 : total_lists = 775) (h2 : fraction = 1 / 4) (h3 : num_lists = ⌈fraction * total_lists⌉) : 
  num_lists = 194 :=
by
  -- Using the conditions given,
  -- total_lists = 775,
  -- fraction = 1 / 4,
  -- num_lists = ⌈fraction * total_lists⌉
  -- We need to show num_lists = 194.
  sorry

end movie_of_the_year_condition_l1262_126288


namespace true_proposition_among_provided_l1262_126296

theorem true_proposition_among_provided :
  ∃ (x0 : ℝ), |x0| ≤ 0 :=
by
  exists 0
  simp

end true_proposition_among_provided_l1262_126296


namespace cookies_per_kid_l1262_126245

theorem cookies_per_kid (total_calories_per_lunch : ℕ) (burger_calories : ℕ) (carrot_calories_per_stick : ℕ) (num_carrot_sticks : ℕ) (cookie_calories : ℕ) (num_cookies : ℕ) : 
  total_calories_per_lunch = 750 →
  burger_calories = 400 →
  carrot_calories_per_stick = 20 →
  num_carrot_sticks = 5 →
  cookie_calories = 50 →
  num_cookies = (total_calories_per_lunch - (burger_calories + num_carrot_sticks * carrot_calories_per_stick)) / cookie_calories →
  num_cookies = 5 :=
by
  sorry

end cookies_per_kid_l1262_126245


namespace find_number_l1262_126273

theorem find_number (x : ℤ) (h : x - 254 + 329 = 695) : x = 620 :=
sorry

end find_number_l1262_126273


namespace box_third_dimension_l1262_126285

theorem box_third_dimension (num_cubes : ℕ) (cube_volume box_vol : ℝ) (dim1 dim2 h : ℝ) (h_num_cubes : num_cubes = 24) (h_cube_volume : cube_volume = 27) (h_dim1 : dim1 = 9) (h_dim2 : dim2 = 12) (h_box_vol : box_vol = num_cubes * cube_volume) :
  box_vol = dim1 * dim2 * h → h = 6 := 
by
  sorry

end box_third_dimension_l1262_126285


namespace range_of_x_l1262_126224

variable (x y : ℝ)

def op (x y : ℝ) := x * (1 - y)

theorem range_of_x (h : op (x - 1) (x + 2) < 0) : x < -1 ∨ 1 < x :=
by
  dsimp [op] at h
  sorry

end range_of_x_l1262_126224


namespace soap_bubble_thickness_scientific_notation_l1262_126229

theorem soap_bubble_thickness_scientific_notation :
  (0.0007 * 0.001) = 7 * 10^(-7) := by
sorry

end soap_bubble_thickness_scientific_notation_l1262_126229
