import Mathlib

namespace roger_coins_left_l1347_134722

theorem roger_coins_left {pennies nickels dimes donated_coins initial_coins remaining_coins : ℕ} 
    (h1 : pennies = 42) 
    (h2 : nickels = 36) 
    (h3 : dimes = 15) 
    (h4 : donated_coins = 66) 
    (h5 : initial_coins = pennies + nickels + dimes) 
    (h6 : remaining_coins = initial_coins - donated_coins) : 
    remaining_coins = 27 := 
sorry

end roger_coins_left_l1347_134722


namespace sum_radical_conjugate_l1347_134799

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end sum_radical_conjugate_l1347_134799


namespace shaded_area_size_l1347_134753

noncomputable def total_shaded_area : ℝ :=
  let R := 9
  let r := R / 2
  let area_larger_circle := 81 * Real.pi
  let shaded_area_larger_circle := area_larger_circle / 2
  let area_smaller_circle := Real.pi * r^2
  let shaded_area_smaller_circle := area_smaller_circle / 2
  let total_shaded_area := shaded_area_larger_circle + shaded_area_smaller_circle
  total_shaded_area

theorem shaded_area_size:
  total_shaded_area = 50.625 * Real.pi := 
by
  sorry

end shaded_area_size_l1347_134753


namespace solve_diamond_l1347_134784

theorem solve_diamond (d : ℕ) (h : 9 * d + 5 = 10 * d + 2) : d = 3 :=
by
  sorry

end solve_diamond_l1347_134784


namespace calculate_area_bounded_figure_l1347_134705

noncomputable def area_of_bounded_figure (R : ℝ) : ℝ :=
  (R^2 / 9) * (3 * Real.sqrt 3 - 2 * Real.pi)

theorem calculate_area_bounded_figure (R : ℝ) :
  ∀ r, r = (R / 3) → area_of_bounded_figure R = (R^2 / 9) * (3 * Real.sqrt 3 - 2 * Real.pi) :=
by
  intros r hr
  subst hr
  exact rfl

end calculate_area_bounded_figure_l1347_134705


namespace original_salary_l1347_134702

theorem original_salary (S : ℝ) (h : (1.12) * (0.93) * (1.09) * (0.94) * S = 1212) : 
  S = 1212 / ((1.12) * (0.93) * (1.09) * (0.94)) :=
by
  sorry

end original_salary_l1347_134702


namespace question1_question2_l1347_134779

noncomputable def minimum_value (x y : ℝ) : ℝ := (1 / x) + (1 / y)

theorem question1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + y^2 = x + y) : 
  minimum_value x y = 2 :=
sorry

theorem question2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x^2 + y^2 = x + y) :
  (x + 1) * (y + 1) ≠ 5 :=
sorry

end question1_question2_l1347_134779


namespace quadrilateral_area_inequality_l1347_134773

theorem quadrilateral_area_inequality
  (a b c d S : ℝ)
  (hS : 0 ≤ S)
  (h : S = (a + b) / 4 * (c + d) / 4)
  : S ≤ (a + b) / 4 * (c + d) / 4 := by
  sorry

end quadrilateral_area_inequality_l1347_134773


namespace trajectory_and_min_area_l1347_134719

theorem trajectory_and_min_area (C : ℝ → ℝ → Prop) (P : ℝ × ℝ → Prop)
  (l : ℝ → ℝ) (F : ℝ × ℝ) (M : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ)
  (k : ℝ) : 
  (∀ x y, P (x, y) ↔ x ^ 2 = 4 * y) → 
  P (0, 1) →
  (∀ y, l y = -1) →
  F = (0, 1) →
  (∀ x1 y1 x2 y2, x1 + x2 = 4 * k → x1 * x2 = -4 →
    M (x1, y1) (x2, y2) = (2 * k, -1)) →
  (min_area : ℝ) → 
  min_area = 4 :=
by
  intros
  sorry

end trajectory_and_min_area_l1347_134719


namespace sector_area_l1347_134786

theorem sector_area (r α : ℝ) (h_r : r = 3) (h_α : α = 2) : (1/2 * r^2 * α) = 9 := by
  sorry

end sector_area_l1347_134786


namespace function_characterization_l1347_134720

noncomputable def f : ℝ → ℝ := sorry

theorem function_characterization (f : ℝ → ℝ) (k : ℝ) :
  (∀ x y : ℝ, f (x^2 + 2*x*y + y^2) = (x + y) * (f x + f y)) →
  (∀ x : ℝ, |f x - k * x| ≤ |x^2 - x|) →
  ∀ x : ℝ, f x = k * x :=
by
  sorry

end function_characterization_l1347_134720


namespace sean_div_julie_l1347_134757

-- Define the sum of the first n integers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define Sean's sum: sum of even integers from 2 to 600
def sean_sum : ℕ := 2 * sum_first_n 300

-- Define Julie's sum: sum of integers from 1 to 300
def julie_sum : ℕ := sum_first_n 300

-- Prove that Sean's sum divided by Julie's sum equals 2
theorem sean_div_julie : sean_sum / julie_sum = 2 := by
  sorry

end sean_div_julie_l1347_134757


namespace prime_squares_5000_9000_l1347_134735

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem prime_squares_5000_9000 : 
  ∃ (l : List ℕ), 
  (∀ p ∈ l, is_prime p ∧ 5000 ≤ p^2 ∧ p^2 ≤ 9000) ∧ 
  l.length = 6 := 
by
  sorry

end prime_squares_5000_9000_l1347_134735


namespace opposite_numbers_expression_l1347_134738

theorem opposite_numbers_expression (a b : ℤ) (h : a + b = 0) : 3 * a + 3 * b - 2 = -2 :=
by
  sorry

end opposite_numbers_expression_l1347_134738


namespace trig_expression_evaluation_l1347_134769

theorem trig_expression_evaluation (θ : ℝ) (h : Real.tan θ = 2) : 
  Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5 := 
  sorry

end trig_expression_evaluation_l1347_134769


namespace inequality_solution_set_l1347_134795

theorem inequality_solution_set :
  ∀ x : ℝ, (1 / (x^2 + 1) > 5 / x + 21 / 10) ↔ x ∈ Set.Ioo (-2 : ℝ) (0 : ℝ) :=
by
  sorry

end inequality_solution_set_l1347_134795


namespace students_water_count_l1347_134709

-- Define the given conditions
def pct_students_juice (total_students : ℕ) : ℕ := 70 * total_students / 100
def pct_students_water (total_students : ℕ) : ℕ := 30 * total_students / 100
def students_juice (total_students : ℕ) : Prop := pct_students_juice total_students = 140

-- Define the proposition that needs to be proven
theorem students_water_count (total_students : ℕ) (h1 : students_juice total_students) : 
  pct_students_water total_students = 60 := 
by
  sorry


end students_water_count_l1347_134709


namespace bruce_paid_amount_l1347_134721

def kg_of_grapes : ℕ := 8
def rate_per_kg_grapes : ℕ := 70
def kg_of_mangoes : ℕ := 10
def rate_per_kg_mangoes : ℕ := 55

def total_amount_paid : ℕ := (kg_of_grapes * rate_per_kg_grapes) + (kg_of_mangoes * rate_per_kg_mangoes)

theorem bruce_paid_amount : total_amount_paid = 1110 :=
by sorry

end bruce_paid_amount_l1347_134721


namespace sum_of_coordinates_of_B_l1347_134726

theorem sum_of_coordinates_of_B (x : ℝ) (y : ℝ) 
  (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A = (0,0)) 
  (hB : B = (x, 3))
  (hslope : (3 - 0) / (x - 0) = 4 / 5) :
  x + 3 = 6.75 := 
by
  sorry

end sum_of_coordinates_of_B_l1347_134726


namespace arithmetic_calculation_l1347_134711

theorem arithmetic_calculation : 3.5 * 0.3 + 1.2 * 0.4 = 1.53 :=
by
  sorry

end arithmetic_calculation_l1347_134711


namespace unit_prices_l1347_134731

theorem unit_prices (x y : ℕ) (h1 : 5 * x + 4 * y = 139) (h2 : 4 * x + 5 * y = 140) :
  x = 15 ∧ y = 16 :=
by
  -- Proof will go here
  sorry

end unit_prices_l1347_134731


namespace movie_theater_ticket_sales_l1347_134787

theorem movie_theater_ticket_sales 
  (A C : ℤ) 
  (h1 : A + C = 900) 
  (h2 : 7 * A + 4 * C = 5100) : 
  A = 500 := 
sorry

end movie_theater_ticket_sales_l1347_134787


namespace sufficient_but_not_necessary_not_necessary_l1347_134742

theorem sufficient_but_not_necessary (m x y a : ℝ) (h₀ : m > 0) (h₁ : |x - a| < m) (h₂ : |y - a| < m) : |x - y| < 2 * m :=
by
  sorry

theorem not_necessary (m : ℝ) (h₀ : m > 0) : ∃ x y a : ℝ, |x - y| < 2 * m ∧ ¬ (|x - a| < m ∧ |y - a| < m) :=
by
  sorry

end sufficient_but_not_necessary_not_necessary_l1347_134742


namespace lily_pads_doubling_l1347_134794

theorem lily_pads_doubling (patch_half_day: ℕ) (doubling_rate: ℝ)
  (H1: patch_half_day = 49)
  (H2: doubling_rate = 2): (patch_half_day + 1) = 50 :=
by 
  sorry

end lily_pads_doubling_l1347_134794


namespace gcd_polynomials_l1347_134765

-- Given condition: a is an even multiple of 1009
def is_even_multiple_of_1009 (a : ℤ) : Prop :=
  ∃ k : ℤ, a = 2 * 1009 * k

-- Statement: gcd(2a^2 + 31a + 58, a + 15) = 1
theorem gcd_polynomials (a : ℤ) (ha : is_even_multiple_of_1009 a) :
  gcd (2 * a^2 + 31 * a + 58) (a + 15) = 1 := 
sorry

end gcd_polynomials_l1347_134765


namespace shaded_grid_percentage_l1347_134781

theorem shaded_grid_percentage (total_squares shaded_squares : ℕ) (h1 : total_squares = 64) (h2 : shaded_squares = 48) : 
  ((shaded_squares : ℚ) / (total_squares : ℚ)) * 100 = 75 :=
by
  rw [h1, h2]
  norm_num

end shaded_grid_percentage_l1347_134781


namespace negation_proposition_l1347_134712

theorem negation_proposition :
  (∀ x : ℝ, 0 < x → x^2 + 1 ≥ 2 * x) ↔ (∃ x : ℝ, 0 < x ∧ x^2 + 1 < 2 * x) :=
by
  sorry

end negation_proposition_l1347_134712


namespace moles_of_NaHCO3_combined_l1347_134752

theorem moles_of_NaHCO3_combined (n_HNO3 n_NaHCO3 : ℕ) (mass_H2O : ℝ) : 
  n_HNO3 = 2 ∧ mass_H2O = 36 ∧ n_HNO3 = n_NaHCO3 → n_NaHCO3 = 2 := by
  sorry

end moles_of_NaHCO3_combined_l1347_134752


namespace acceptable_colorings_correct_l1347_134775

def acceptableColorings (n : ℕ) : ℕ :=
  (3^(n + 1) + (-1:ℤ)^(n + 1)).natAbs / 2

theorem acceptable_colorings_correct (n : ℕ) :
  acceptableColorings n = (3^(n + 1) + (-1:ℤ)^(n + 1)).natAbs / 2 :=
by
  sorry

end acceptable_colorings_correct_l1347_134775


namespace calculate_expr_at_3_l1347_134785

-- Definition of the expression
def expr (x : ℕ) : ℕ := (x + x * x^(x^2)) * 3

-- The proof statement
theorem calculate_expr_at_3 : expr 3 = 177156 := 
by
  sorry

end calculate_expr_at_3_l1347_134785


namespace range_of_a_I_minimum_value_of_a_II_l1347_134788

open Real

def f (x a : ℝ) : ℝ := abs (x - a)

theorem range_of_a_I (a : ℝ) :
  (∀ x, -1 ≤ x → x ≤ 3 → f x a ≤ 3) ↔ 0 ≤ a ∧ a ≤ 2 := sorry

theorem minimum_value_of_a_II :
  ∀ a : ℝ, (∀ x : ℝ, f (x - a) a + f (x + a) a ≥ 1 - 2 * a) ↔ a ≥ (1 / 4) :=
sorry

end range_of_a_I_minimum_value_of_a_II_l1347_134788


namespace average_age_of_combined_rooms_l1347_134792

theorem average_age_of_combined_rooms
  (num_people_A : ℕ) (avg_age_A : ℕ)
  (num_people_B : ℕ) (avg_age_B : ℕ)
  (num_people_C : ℕ) (avg_age_C : ℕ)
  (hA : num_people_A = 8) (hAA : avg_age_A = 35)
  (hB : num_people_B = 5) (hBB : avg_age_B = 30)
  (hC : num_people_C = 7) (hCC : avg_age_C = 50) :
  ((num_people_A * avg_age_A + num_people_B * avg_age_B + num_people_C * avg_age_C) / 
  (num_people_A + num_people_B + num_people_C) = 39) :=
by
  sorry

end average_age_of_combined_rooms_l1347_134792


namespace roberto_outfits_l1347_134717

-- Define the conditions
def trousers := 5
def shirts := 8
def jackets := 4

-- Define the total number of outfits
def total_outfits : ℕ := trousers * shirts * jackets

-- The theorem stating the actual problem and answer
theorem roberto_outfits : total_outfits = 160 :=
by
  -- skip the proof for now
  sorry

end roberto_outfits_l1347_134717


namespace system_equivalence_l1347_134727

theorem system_equivalence (f g : ℝ → ℝ) (x : ℝ) (h1 : f x > 0) (h2 : g x > 0) : f x + g x > 0 :=
sorry

end system_equivalence_l1347_134727


namespace solution_set_of_inequality_l1347_134754

theorem solution_set_of_inequality (x : ℝ) : 
  (x^2 - abs x - 2 < 0) ↔ (-2 < x ∧ x < 2) := 
sorry

end solution_set_of_inequality_l1347_134754


namespace sin_cos_of_tan_is_two_l1347_134744

theorem sin_cos_of_tan_is_two (α : ℝ) (h : Real.tan α = 2) : Real.sin α * Real.cos α = 2 / 5 :=
sorry

end sin_cos_of_tan_is_two_l1347_134744


namespace find_x_l1347_134736

theorem find_x (x : ℚ) (n : ℤ) (f : ℚ) (h1 : x = n + f) (h2 : n = ⌊x⌋) (h3 : f < 1): 
  ⌊x⌋ + x = 17 / 4 → x = 9 / 4 :=
by 
  sorry

end find_x_l1347_134736


namespace real_roots_quadratic_iff_l1347_134704

theorem real_roots_quadratic_iff (a : ℝ) : (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 1 = 0) ↔ a ≤ 2 := 
sorry

end real_roots_quadratic_iff_l1347_134704


namespace cost_per_square_meter_of_mat_l1347_134730

theorem cost_per_square_meter_of_mat {L W E : ℝ} : 
  L = 20 → W = 15 → E = 57000 → (E / (L * W)) = 190 :=
by
  intros hL hW hE
  rw [hL, hW, hE]
  sorry

end cost_per_square_meter_of_mat_l1347_134730


namespace smallest_positive_integer_l1347_134758

theorem smallest_positive_integer (x : ℕ) : 
  (5 * x ≡ 18 [MOD 33]) ∧ (x ≡ 4 [MOD 7]) → x = 10 := 
by 
  sorry

end smallest_positive_integer_l1347_134758


namespace correct_exponentiation_rule_l1347_134777

theorem correct_exponentiation_rule (x y : ℝ) : ((x^2)^3 = x^6) :=
  by sorry

end correct_exponentiation_rule_l1347_134777


namespace total_employees_l1347_134782

variable (E : ℕ) -- E is the total number of employees

-- Conditions given in the problem
variable (male_fraction : ℚ := 0.45) -- 45% of the total employees are males
variable (males_below_50 : ℕ := 1170) -- 1170 males are below 50 years old
variable (males_total : ℕ := 2340) -- Total number of male employees

-- Condition derived from the problem (calculation of total males)
lemma male_employees_equiv (h : males_total = 2 * males_below_50) : males_total = 2340 :=
  by sorry

-- Main theorem
theorem total_employees (h : male_fraction * E = males_total) : E = 5200 :=
  by sorry

end total_employees_l1347_134782


namespace branches_on_one_stem_l1347_134772

theorem branches_on_one_stem (x : ℕ) (h : 1 + x + x^2 = 31) : x = 5 :=
by {
  sorry
}

end branches_on_one_stem_l1347_134772


namespace initial_average_age_l1347_134764

theorem initial_average_age (A : ℝ) (n : ℕ) (h1 : n = 17) (h2 : n * A + 32 = (n + 1) * 15) : A = 14 := by
  sorry

end initial_average_age_l1347_134764


namespace fraction_halfway_l1347_134714

theorem fraction_halfway (a b c d : ℚ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : d = 6) :
  (1 / 2) * ((a / b) + (c / d)) = 19 / 24 := 
by
  sorry

end fraction_halfway_l1347_134714


namespace max_value_of_quadratic_exists_r_for_max_value_of_quadratic_l1347_134749

theorem max_value_of_quadratic (r : ℝ) : -7 * r ^ 2 + 50 * r - 20 ≤ 5 :=
by sorry

theorem exists_r_for_max_value_of_quadratic : ∃ r : ℝ, -7 * r ^ 2 + 50 * r - 20 = 5 :=
by sorry

end max_value_of_quadratic_exists_r_for_max_value_of_quadratic_l1347_134749


namespace points_after_perfect_games_l1347_134774

-- Given conditions
def perfect_score := 21
def num_games := 3

-- Theorem statement
theorem points_after_perfect_games : perfect_score * num_games = 63 := by
  sorry

end points_after_perfect_games_l1347_134774


namespace radius_of_hole_l1347_134766

-- Define the dimensions of the rectangular solid
def length1 : ℕ := 3
def length2 : ℕ := 8
def length3 : ℕ := 9

-- Define the radius of the hole
variable (r : ℕ)

-- Condition: The area of the 2 circles removed equals the lateral surface area of the cylinder
axiom area_condition : 2 * Real.pi * r^2 = 2 * Real.pi * r * length1

-- Prove that the radius of the cylindrical hole is 3
theorem radius_of_hole : r = 3 := by
  sorry

end radius_of_hole_l1347_134766


namespace number_of_smaller_pipes_l1347_134746

theorem number_of_smaller_pipes (D_L D_s : ℝ) (h1 : D_L = 8) (h2 : D_s = 2) (v: ℝ) :
  let A_L := (π * (D_L / 2)^2)
  let A_s := (π * (D_s / 2)^2)
  (A_L / A_s) = 16 :=
by {
  sorry
}

end number_of_smaller_pipes_l1347_134746


namespace frames_per_page_l1347_134707

theorem frames_per_page (total_frames : ℕ) (total_pages : ℝ) (h1 : total_frames = 1573) (h2 : total_pages = 11.0) : total_frames / total_pages = 143 := by
  sorry

end frames_per_page_l1347_134707


namespace probability_two_or_fewer_distinct_digits_l1347_134770

def digits : Set ℕ := {1, 2, 3}

def total_3_digit_numbers : ℕ := 27

def distinct_3_digit_numbers : ℕ := 6

def at_most_two_distinct_numbers : ℕ := total_3_digit_numbers - distinct_3_digit_numbers

theorem probability_two_or_fewer_distinct_digits :
  (at_most_two_distinct_numbers : ℚ) / total_3_digit_numbers = 7 / 9 := by
  sorry

end probability_two_or_fewer_distinct_digits_l1347_134770


namespace student_failed_by_l1347_134729

-- Definitions based on the problem conditions
def total_marks : ℕ := 500
def passing_percentage : ℕ := 40
def marks_obtained : ℕ := 150
def passing_marks : ℕ := (passing_percentage * total_marks) / 100

-- The theorem statement
theorem student_failed_by :
  (passing_marks - marks_obtained) = 50 :=
by
  -- The proof is omitted
  sorry

end student_failed_by_l1347_134729


namespace yellow_crayons_count_l1347_134706

def red_crayons := 14
def blue_crayons := red_crayons + 5
def yellow_crayons := 2 * blue_crayons - 6

theorem yellow_crayons_count : yellow_crayons = 32 := by
  sorry

end yellow_crayons_count_l1347_134706


namespace tim_total_spending_l1347_134789

def lunch_cost : ℝ := 50.50
def dessert_cost : ℝ := 8.25
def beverage_cost : ℝ := 3.75
def lunch_discount : ℝ := 0.10
def dessert_tax : ℝ := 0.07
def beverage_tax : ℝ := 0.05
def lunch_tip_rate : ℝ := 0.20
def other_items_tip_rate : ℝ := 0.15

def total_spending : ℝ := 
  let lunch_after_discount := lunch_cost * (1 - lunch_discount)
  let dessert_after_tax := dessert_cost * (1 + dessert_tax)
  let beverage_after_tax := beverage_cost * (1 + beverage_tax)
  let tip_on_lunch := lunch_after_discount * lunch_tip_rate
  let combined_other_items := dessert_after_tax + beverage_after_tax
  let tip_on_other_items := combined_other_items * other_items_tip_rate
  lunch_after_discount + dessert_after_tax + beverage_after_tax + tip_on_lunch + tip_on_other_items

theorem tim_total_spending :
  total_spending = 69.23 :=
by
  sorry

end tim_total_spending_l1347_134789


namespace hank_newspaper_reading_time_l1347_134728

theorem hank_newspaper_reading_time
  (n_days_weekday : ℕ := 5)
  (novel_reading_time_weekday : ℕ := 60)
  (n_days_weekend : ℕ := 2)
  (total_weekly_reading_time : ℕ := 810)
  (x : ℕ)
  (h1 : n_days_weekday * x + n_days_weekday * novel_reading_time_weekday +
        n_days_weekend * 2 * x + n_days_weekend * 2 * novel_reading_time_weekday = total_weekly_reading_time) :
  x = 30 := 
by {
  sorry -- Proof would go here
}

end hank_newspaper_reading_time_l1347_134728


namespace mark_gpa_probability_l1347_134797

theorem mark_gpa_probability :
  let A_points := 4
  let B_points := 3
  let C_points := 2
  let D_points := 1
  let GPA_required := 3.5
  let total_subjects := 4
  let total_points_required := GPA_required * total_subjects
  -- Points from guaranteed A's in Mathematics and Science
  let guaranteed_points := 8
  -- Required points from Literature and History
  let points_needed := total_points_required - guaranteed_points
  -- Probabilities for grades in Literature
  let prob_A_Lit := 1 / 3
  let prob_B_Lit := 1 / 3
  let prob_C_Lit := 1 / 3
  -- Probabilities for grades in History
  let prob_A_Hist := 1 / 5
  let prob_B_Hist := 1 / 4
  let prob_C_Hist := 11 / 20
  -- Combinations of grades to achieve the required points
  let prob_two_As := prob_A_Lit * prob_A_Hist
  let prob_A_Lit_B_Hist := prob_A_Lit * prob_B_Hist
  let prob_B_Lit_A_Hist := prob_B_Lit * prob_A_Hist
  let prob_two_Bs := prob_B_Lit * prob_B_Hist
  -- Total probability of achieving at least the required GPA
  let total_probability := prob_two_As + prob_A_Lit_B_Hist + prob_B_Lit_A_Hist + prob_two_Bs
  total_probability = 3 / 10 := sorry

end mark_gpa_probability_l1347_134797


namespace ratio_shortest_to_middle_tree_l1347_134725

theorem ratio_shortest_to_middle_tree (height_tallest : ℕ) 
  (height_middle : ℕ) (height_shortest : ℕ)
  (h1 : height_tallest = 150) 
  (h2 : height_middle = (2 * height_tallest) / 3) 
  (h3 : height_shortest = 50) : 
  height_shortest / height_middle = 1 / 2 := by sorry

end ratio_shortest_to_middle_tree_l1347_134725


namespace time_needed_n_l1347_134715

variable (n : Nat)
variable (d : Nat := n - 1)
variable (s : ℚ := 2 / 3 * (d))
variable (time_third_mile : ℚ := 3)
noncomputable def time_needed (n : Nat) : ℚ := (3 * (n - 1)) / 2

theorem time_needed_n: 
  (∀ (n : Nat), n > 2 → time_needed n = (3 * (n - 1)) / 2) :=
by
  intros n hn
  sorry

end time_needed_n_l1347_134715


namespace smallest_n_cond_l1347_134762

theorem smallest_n_cond (n : ℕ) (h1 : n >= 100 ∧ n < 1000) (h2 : n ≡ 3 [MOD 9]) (h3 : n ≡ 3 [MOD 4]) : n = 111 := 
sorry

end smallest_n_cond_l1347_134762


namespace condition_sufficient_but_not_necessary_l1347_134771

theorem condition_sufficient_but_not_necessary (x : ℝ) :
  (x^3 > 8 → |x| > 2) ∧ (|x| > 2 → ¬ (x^3 ≤ 8 ∨ x^3 ≥ 8)) := by
  sorry

end condition_sufficient_but_not_necessary_l1347_134771


namespace sin_phi_value_l1347_134798

theorem sin_phi_value 
  (φ α : ℝ)
  (hφ : φ = 2 * α)
  (hα1 : Real.sin α = (Real.sqrt 5) / 5)
  (hα2 : Real.cos α = 2 * (Real.sqrt 5) / 5) 
  : Real.sin φ = 4 / 5 := 
by 
  sorry

end sin_phi_value_l1347_134798


namespace max_height_of_ball_l1347_134756

noncomputable def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 45

theorem max_height_of_ball : ∃ t : ℝ, (h t) = 69.5 :=
sorry

end max_height_of_ball_l1347_134756


namespace lawn_mowing_rate_l1347_134768

-- Definitions based on conditions
def total_hours_mowed : ℕ := 2 * 7
def money_left_after_expenses (R : ℕ) : ℕ := (14 * R) / 4

-- The problem statement
theorem lawn_mowing_rate (h : money_left_after_expenses R = 49) : R = 14 := 
sorry

end lawn_mowing_rate_l1347_134768


namespace train_length_l1347_134708

theorem train_length (L : ℝ) (h1 : 46 - 36 = 10) (h2 : 45 * (10 / 3600) = 1 / 8) : L = 62.5 :=
by
  sorry

end train_length_l1347_134708


namespace lcm_15_18_l1347_134745

theorem lcm_15_18 : Nat.lcm 15 18 = 90 := by
  sorry

end lcm_15_18_l1347_134745


namespace cube_mono_l1347_134755

theorem cube_mono {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_mono_l1347_134755


namespace product_of_number_subtracting_7_equals_9_l1347_134737

theorem product_of_number_subtracting_7_equals_9 (x : ℤ) (h : x - 7 = 9) : x * 5 = 80 := by
  sorry

end product_of_number_subtracting_7_equals_9_l1347_134737


namespace total_shaded_area_l1347_134734

theorem total_shaded_area (S T : ℝ) (h1 : 12 / S = 4) (h2 : S / T = 4) : 
  S^2 + 12 * T^2 = 15.75 :=
by 
  sorry

end total_shaded_area_l1347_134734


namespace bag_cost_is_2_l1347_134776

-- Define the inputs and conditions
def carrots_per_day := 1
def days_per_year := 365
def carrots_per_bag := 5
def yearly_spending := 146

-- The final goal is to find the cost per bag
def cost_per_bag := yearly_spending / ((carrots_per_day * days_per_year) / carrots_per_bag)

-- Prove that the cost per bag is $2
theorem bag_cost_is_2 : cost_per_bag = 2 := by
  -- Using sorry to complete the proof
  sorry

end bag_cost_is_2_l1347_134776


namespace partial_fraction_sum_l1347_134710

theorem partial_fraction_sum :
  ∃ P Q R : ℚ, 
    P * ((-1 : ℚ) * (-2 : ℚ)) + Q * ((-3 : ℚ) * (-2 : ℚ)) + R * ((-3 : ℚ) * (1 : ℚ))
    = 14 ∧ 
    R * (1 : ℚ) * (3 : ℚ) + Q * ((-4 : ℚ) * (-3 : ℚ)) + P * ((3 : ℚ) * (1 : ℚ)) 
      = 12 ∧ 
    P + Q + R = 115 / 30 := by
  sorry

end partial_fraction_sum_l1347_134710


namespace z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha_l1347_134767

-- Given conditions and definitions
variables {α : ℝ} {z : ℂ} 
  (hz : z + 1/z = 2 * Real.cos α)

-- The target statement
theorem z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha (n : ℕ) (hz : z + 1/z = 2 * Real.cos α) : 
  z ^ n + 1 / (z ^ n) = 2 * Real.cos (n * α) := 
  sorry

end z_pow_n_add_inv_pow_n_eq_two_cos_n_alpha_l1347_134767


namespace log_a_interval_l1347_134783

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem log_a_interval (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  {a | log_a a 3 - log_a a 1 = 2} = {Real.sqrt 3, Real.sqrt 3 / 3} :=
by
  sorry

end log_a_interval_l1347_134783


namespace arithmetic_sequence_values_l1347_134713

theorem arithmetic_sequence_values (a b c : ℤ) 
  (h1 : 2 * b = a + c)
  (h2 : 2 * a = b + 1)
  (h3 : 2 * c = b + 9) 
  (h4 : a + b + c = -15) :
  b = -5 ∧ a * c = 21 :=
by
  sorry

end arithmetic_sequence_values_l1347_134713


namespace new_average_weight_is_27_3_l1347_134791

-- Define the given conditions as variables/constants in Lean
noncomputable def original_students : ℕ := 29
noncomputable def original_average_weight : ℝ := 28
noncomputable def new_student_weight : ℝ := 7

-- The total weight of the original students
noncomputable def original_total_weight : ℝ := original_students * original_average_weight
-- The new total number of students
noncomputable def new_total_students : ℕ := original_students + 1
-- The new total weight after new student is added
noncomputable def new_total_weight : ℝ := original_total_weight + new_student_weight

-- The theorem to prove that the new average weight is 27.3 kg
theorem new_average_weight_is_27_3 : (new_total_weight / new_total_students) = 27.3 := 
by
  sorry -- The proof will be provided here

end new_average_weight_is_27_3_l1347_134791


namespace frac_div_l1347_134701

theorem frac_div : (3 / 7) / (4 / 5) = 15 / 28 := by
  sorry

end frac_div_l1347_134701


namespace determine_transportation_mode_l1347_134759

def distance : ℝ := 60 -- in kilometers
def time : ℝ := 3 -- in hours
def speed_of_walking : ℝ := 5 -- typical speed in km/h
def speed_of_bicycle_riding : ℝ := 15 -- lower bound of bicycle speed in km/h
def speed_of_driving_a_car : ℝ := 20 -- typical minimum speed in km/h

theorem determine_transportation_mode : (distance / time) = speed_of_driving_a_car ∧ speed_of_driving_a_car ≥ speed_of_walking + speed_of_bicycle_riding - speed_of_driving_a_car := sorry

end determine_transportation_mode_l1347_134759


namespace total_people_in_school_l1347_134751

def number_of_girls := 315
def number_of_boys := 309
def number_of_teachers := 772
def total_number_of_people := number_of_girls + number_of_boys + number_of_teachers

theorem total_people_in_school :
  total_number_of_people = 1396 :=
by sorry

end total_people_in_school_l1347_134751


namespace min_sum_of_factors_of_144_is_neg_145_l1347_134723

theorem min_sum_of_factors_of_144_is_neg_145 
  (a b : ℤ) 
  (h : a * b = 144) : 
  a + b ≥ -145 := 
sorry

end min_sum_of_factors_of_144_is_neg_145_l1347_134723


namespace chandler_bike_purchase_weeks_l1347_134750

theorem chandler_bike_purchase_weeks (bike_cost birthday_money weekly_earnings total_weeks : ℕ) 
  (h_bike_cost : bike_cost = 600)
  (h_birthday_money : birthday_money = 60 + 40 + 20 + 30)
  (h_weekly_earnings : weekly_earnings = 18)
  (h_total_weeks : total_weeks = 25) :
  birthday_money + weekly_earnings * total_weeks = bike_cost :=
by {
  sorry
}

end chandler_bike_purchase_weeks_l1347_134750


namespace solve_marble_problem_l1347_134793

noncomputable def marble_problem : Prop :=
  ∃ k : ℕ, k ≥ 0 ∧ k ≤ 50 ∧ 
  (∀ initial_white initial_black : ℕ, initial_white = 50 ∧ initial_black = 50 → 
  ∃ w b : ℕ, w = 50 + k - initial_black ∧ b = 50 - k ∧ (w, b) = (2, 0))

theorem solve_marble_problem: marble_problem :=
sorry

end solve_marble_problem_l1347_134793


namespace price_of_first_oil_l1347_134716

variable {x : ℝ}
variable {price1 volume1 price2 volume2 mix_price mix_volume : ℝ}

theorem price_of_first_oil:
  volume1 = 10 →
  price2 = 68 →
  volume2 = 5 →
  mix_volume = 15 →
  mix_price = 56 →
  (volume1 * x + volume2 * price2 = mix_volume * mix_price) →
  x = 50 :=
by
  intros h1 h2 h3 h4 h5 h6
  have h1 : volume1 = 10 := h1
  have h2 : price2 = 68 := h2
  have h3 : volume2 = 5 := h3
  have h4 : mix_volume = 15 := h4
  have h5 : mix_price = 56 := h5
  have h6 : volume1 * x + volume2 * price2 = mix_volume * mix_price := h6
  sorry

end price_of_first_oil_l1347_134716


namespace total_heartbeats_during_race_l1347_134748

-- Definitions for conditions
def heart_rate_per_minute : ℕ := 120
def pace_minutes_per_km : ℕ := 4
def race_distance_km : ℕ := 120

-- Lean statement of the proof problem
theorem total_heartbeats_during_race :
  120 * (4 * 120) = 57600 := by
  sorry

end total_heartbeats_during_race_l1347_134748


namespace train_speed_l1347_134743

theorem train_speed
  (num_carriages : ℕ)
  (length_carriage length_engine : ℕ)
  (bridge_length_km : ℝ)
  (crossing_time_min : ℝ)
  (h1 : num_carriages = 24)
  (h2 : length_carriage = 60)
  (h3 : length_engine = 60)
  (h4 : bridge_length_km = 4.5)
  (h5 : crossing_time_min = 6) :
  (num_carriages * length_carriage + length_engine) / 1000 + bridge_length_km / (crossing_time_min / 60) = 60 :=
by
  sorry

end train_speed_l1347_134743


namespace remainder_of_division_l1347_134763

open Polynomial

noncomputable def p : Polynomial ℤ := 3 * X^3 - 20 * X^2 + 45 * X + 23
noncomputable def d : Polynomial ℤ := (X - 3)^2

theorem remainder_of_division :
  ∃ q r : Polynomial ℤ, p = q * d + r ∧ degree r < degree d ∧ r = 6 * X + 41 := sorry

end remainder_of_division_l1347_134763


namespace average_of_new_sequence_l1347_134741

variable (c : ℕ)  -- c is a positive integer
variable (d : ℕ)  -- d is the average of the sequence starting from c 

def average_of_sequence (seq : List ℕ) : ℕ :=
  if h : seq.length ≠ 0 then seq.sum / seq.length else 0

theorem average_of_new_sequence (h : d = average_of_sequence [c, c+1, c+2, c+3, c+4, c+5, c+6]) :
  average_of_sequence [d, d+1, d+2, d+3, d+4, d+5, d+6] = c + 6 := 
sorry

end average_of_new_sequence_l1347_134741


namespace g_f_neg5_l1347_134761

-- Define the function f
def f (x : ℝ) := 2 * x ^ 2 - 4

-- Define the function g with the known condition g(f(5)) = 12
axiom g : ℝ → ℝ
axiom g_f5 : g (f 5) = 12

-- Now state the main theorem we need to prove
theorem g_f_neg5 : g (f (-5)) = 12 := by
  sorry

end g_f_neg5_l1347_134761


namespace complex_value_of_z_six_plus_z_inv_six_l1347_134733

open Complex

theorem complex_value_of_z_six_plus_z_inv_six (z : ℂ) (h : z + z⁻¹ = 1) : z^6 + (z⁻¹)^6 = 2 := by
  sorry

end complex_value_of_z_six_plus_z_inv_six_l1347_134733


namespace sum_of_fractions_l1347_134796

theorem sum_of_fractions :
  (3 / 15) + (6 / 15) + (9 / 15) + (12 / 15) + (15 / 15) + 
  (18 / 15) + (21 / 15) + (24 / 15) + (27 / 15) + (75 / 15) = 14 :=
by
  sorry

end sum_of_fractions_l1347_134796


namespace area_rectangle_relation_l1347_134747

theorem area_rectangle_relation (x y : ℝ) (h : x * y = 12) : y = 12 / x :=
by
  sorry

end area_rectangle_relation_l1347_134747


namespace largest_inscribed_triangle_area_l1347_134780

theorem largest_inscribed_triangle_area
  (D : Type) 
  (radius : ℝ) 
  (r_eq : radius = 8) 
  (triangle_area : ℝ)
  (max_area : triangle_area = 64) :
  ∃ (base height : ℝ), (base = 2 * radius) ∧ (height = radius) ∧ (triangle_area = (1 / 2) * base * height) := 
by
  sorry

end largest_inscribed_triangle_area_l1347_134780


namespace frames_per_page_l1347_134732

theorem frames_per_page (total_frames : ℕ) (pages : ℕ) (frames : ℕ) 
  (h1 : total_frames = 143) 
  (h2 : pages = 13) 
  (h3 : frames = total_frames / pages) : 
  frames = 11 := 
by 
  sorry

end frames_per_page_l1347_134732


namespace find_x_l1347_134718

-- Define vectors a and b
def a : ℝ × ℝ := (1, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Define the parallel condition
def parallel (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

-- Lean statement asserting that if a is parallel to b for some x, then x = 2
theorem find_x (x : ℝ) (h : parallel a (b x)) : x = 2 := 
by sorry

end find_x_l1347_134718


namespace product_of_two_integers_l1347_134778

theorem product_of_two_integers (x y : ℕ) (h1 : x + y = 22) (h2 : x^2 - y^2 = 44) : x * y = 120 :=
by
  sorry

end product_of_two_integers_l1347_134778


namespace horatio_sonnets_l1347_134739

theorem horatio_sonnets (num_lines_per_sonnet : ℕ) (heard_sonnets : ℕ) (unheard_lines : ℕ) (h1 : num_lines_per_sonnet = 16) (h2 : heard_sonnets = 9) (h3 : unheard_lines = 126) :
  ∃ total_sonnets : ℕ, total_sonnets = 16 :=
by
  -- Note: The proof is not required, hence 'sorry' is included to skip it.
  sorry

end horatio_sonnets_l1347_134739


namespace payment_ratio_l1347_134724

theorem payment_ratio (m p t : ℕ) (hm : m = 14) (hp : p = 84) (ht : t = m * 12) :
  (p : ℚ) / ((t : ℚ) - p) = 1 :=
by
  sorry

end payment_ratio_l1347_134724


namespace abs_opposite_of_three_eq_5_l1347_134760

theorem abs_opposite_of_three_eq_5 : ∀ (a : ℤ), a = -3 → |a - 2| = 5 := by
  sorry

end abs_opposite_of_three_eq_5_l1347_134760


namespace car_dealership_theorem_l1347_134740

def car_dealership_problem : Prop :=
  let initial_cars := 100
  let new_shipment := 150
  let initial_silver_percentage := 0.20
  let new_silver_percentage := 0.40
  let initial_silver := initial_silver_percentage * initial_cars
  let new_silver := new_silver_percentage * new_shipment
  let total_silver := initial_silver + new_silver
  let total_cars := initial_cars + new_shipment
  let silver_percentage := (total_silver / total_cars) * 100
  silver_percentage = 32

theorem car_dealership_theorem : car_dealership_problem :=
by {
  sorry
}

end car_dealership_theorem_l1347_134740


namespace max_distance_l1347_134790

theorem max_distance (x y : ℝ) (u v w : ℝ)
  (h1 : u = Real.sqrt (x^2 + y^2))
  (h2 : v = Real.sqrt ((x - 1)^2 + y^2))
  (h3 : w = Real.sqrt ((x - 1)^2 + (y - 1)^2))
  (h4 : u^2 + v^2 = w^2) :
  ∃ (P : ℝ), P = 2 + Real.sqrt 2 :=
sorry

end max_distance_l1347_134790


namespace xyz_value_l1347_134700

theorem xyz_value (x y z : ℝ) (h1 : y = x + 1) (h2 : x + y = 2 * z) (h3 : x = 3) : x * y * z = 42 :=
by
  -- proof here
  sorry

end xyz_value_l1347_134700


namespace problem_statement_l1347_134703

theorem problem_statement
  (a b c : ℝ)
  (h1 : a + 2 * b + 3 * c = 12)
  (h2 : a^2 + b^2 + c^2 = a * b + a * c + b * c) :
  a + b^2 + c^3 = 14 := 
sorry

end problem_statement_l1347_134703
