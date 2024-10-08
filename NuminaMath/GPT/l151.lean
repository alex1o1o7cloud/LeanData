import Mathlib

namespace elena_bouquet_petals_l151_151835

def num_petals (count : ℕ) (petals_per_flower : ℕ) : ℕ :=
  count * petals_per_flower

theorem elena_bouquet_petals :
  let num_lilies := 4
  let lilies_petal_count := num_petals num_lilies 6
  
  let num_tulips := 2
  let tulips_petal_count := num_petals num_tulips 3

  let num_roses := 2
  let roses_petal_count := num_petals num_roses 5
  
  let num_daisies := 1
  let daisies_petal_count := num_petals num_daisies 12
  
  lilies_petal_count + tulips_petal_count + roses_petal_count + daisies_petal_count = 52 := by
  sorry

end elena_bouquet_petals_l151_151835


namespace find_number_l151_151251

theorem find_number (x : ℤ) (h : 3 * x + 3 * 12 + 3 * 13 + 11 = 134) : x = 16 :=
by
  sorry

end find_number_l151_151251


namespace handshakes_exchanged_l151_151568

-- Let n be the number of couples
noncomputable def num_couples := 7

-- Total number of people at the gathering
noncomputable def total_people := num_couples * 2

-- Number of people each person shakes hands with
noncomputable def handshakes_per_person := total_people - 2

-- Total number of unique handshakes
noncomputable def total_handshakes := total_people * handshakes_per_person / 2

theorem handshakes_exchanged :
  total_handshakes = 77 :=
by
  sorry

end handshakes_exchanged_l151_151568


namespace min_value_fraction_expr_l151_151765

theorem min_value_fraction_expr : ∀ (x : ℝ), x > 0 → (4 + x) * (1 + x) / x ≥ 9 :=
by
  sorry

end min_value_fraction_expr_l151_151765


namespace average_minutes_per_day_l151_151290

theorem average_minutes_per_day
  (f : ℕ) -- Number of fifth graders
  (third_grade_minutes : ℕ := 10)
  (fourth_grade_minutes : ℕ := 18)
  (fifth_grade_minutes : ℕ := 12)
  (third_grade_students : ℕ := 3 * f)
  (fourth_grade_students : ℕ := (3 / 2) * f) -- Assumed to work with integer or rational numbers
  (fifth_grade_students : ℕ := f)
  (total_minutes_third_grade : ℕ := third_grade_minutes * third_grade_students)
  (total_minutes_fourth_grade : ℕ := fourth_grade_minutes * fourth_grade_students)
  (total_minutes_fifth_grade : ℕ := fifth_grade_minutes * fifth_grade_students)
  (total_minutes : ℕ := total_minutes_third_grade + total_minutes_fourth_grade + total_minutes_fifth_grade)
  (total_students : ℕ := third_grade_students + fourth_grade_students + fifth_grade_students) :
  (total_minutes / total_students : ℝ) = 12.55 :=
by
  sorry

end average_minutes_per_day_l151_151290


namespace value_of_a_plus_d_l151_151246

variable (a b c d : ℝ)

theorem value_of_a_plus_d (h1 : a + b = 4) (h2 : b + c = 7) (h3 : c + d = 5) : a + d = 4 :=
sorry

end value_of_a_plus_d_l151_151246


namespace quotient_of_sum_of_remainders_div_16_eq_0_l151_151828

-- Define the set of distinct remainders of squares modulo 16 for n in 1 to 15
def distinct_remainders_mod_16 : Finset ℕ :=
  {1, 4, 9, 0}

-- Define the sum of the distinct remainders
def sum_of_remainders : ℕ :=
  distinct_remainders_mod_16.sum id

-- Proposition to prove the quotient when sum_of_remainders is divided by 16 is 0
theorem quotient_of_sum_of_remainders_div_16_eq_0 :
  (sum_of_remainders / 16) = 0 :=
by
  sorry

end quotient_of_sum_of_remainders_div_16_eq_0_l151_151828


namespace negation_of_universal_proposition_l151_151243

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 > Real.log x)) ↔ (∃ x : ℝ, x^2 ≤ Real.log x) :=
by
  sorry

end negation_of_universal_proposition_l151_151243


namespace butterfly_development_time_l151_151186

theorem butterfly_development_time :
  ∀ (larva_time cocoon_time : ℕ), 
  (larva_time = 3 * cocoon_time) → 
  (cocoon_time = 30) → 
  (larva_time + cocoon_time = 120) :=
by 
  intros larva_time cocoon_time h1 h2
  sorry

end butterfly_development_time_l151_151186


namespace discarded_second_number_l151_151293

-- Define the conditions
def avg_original_50 : ℝ := 38
def total_sum_50_numbers : ℝ := 50 * avg_original_50
def discarded_first : ℝ := 45
def avg_remaining_48 : ℝ := 37.5
def total_sum_remaining_48 : ℝ := 48 * avg_remaining_48
def sum_discarded := total_sum_50_numbers - total_sum_remaining_48

-- Define the proof statement
theorem discarded_second_number (x : ℝ) (h : discarded_first + x = sum_discarded) : x = 55 :=
by
  sorry

end discarded_second_number_l151_151293


namespace flight_time_sum_l151_151166

theorem flight_time_sum (h m : ℕ)
  (Hdep : true)   -- Placeholder condition for the departure time being 3:45 PM
  (Hlay : 25 = 25)   -- Placeholder condition for the layover being 25 minutes
  (Harr : true)   -- Placeholder condition for the arrival time being 8:02 PM
  (HsameTZ : true)   -- Placeholder condition for the same time zone
  (H0m : 0 < m) 
  (Hm60 : m < 60)
  (Hfinal_time : (h, m) = (3, 52)) : 
  h + m = 55 := 
by {
  sorry
}

end flight_time_sum_l151_151166


namespace zoo_revenue_is_61_l151_151220

def children_monday := 7
def children_tuesday := 4
def adults_monday := 5
def adults_tuesday := 2
def child_ticket_cost := 3
def adult_ticket_cost := 4

def total_children := children_monday + children_tuesday
def total_adults := adults_monday + adults_tuesday

def total_children_cost := total_children * child_ticket_cost
def total_adult_cost := total_adults * adult_ticket_cost

def total_zoo_revenue := total_children_cost + total_adult_cost

theorem zoo_revenue_is_61 : total_zoo_revenue = 61 := by
  sorry

end zoo_revenue_is_61_l151_151220


namespace range_of_x_satisfies_conditions_l151_151806

theorem range_of_x_satisfies_conditions (x : ℝ) (h : x^2 - 4 < 0 ∨ |x| = 2) : -2 ≤ x ∧ x ≤ 2 := 
by
  sorry

end range_of_x_satisfies_conditions_l151_151806


namespace complement_fraction_irreducible_l151_151902

theorem complement_fraction_irreducible (a b : ℕ) (h : Nat.gcd a b = 1) : Nat.gcd (b - a) b = 1 :=
sorry

end complement_fraction_irreducible_l151_151902


namespace remainder_of_sum_is_five_l151_151445

theorem remainder_of_sum_is_five (a b c d : ℕ) (ha : a % 15 = 11) (hb : b % 15 = 12) (hc : c % 15 = 13) (hd : d % 15 = 14) :
  (a + b + c + d) % 15 = 5 :=
by
  sorry

end remainder_of_sum_is_five_l151_151445


namespace cost_of_each_green_hat_l151_151863

theorem cost_of_each_green_hat
  (total_hats : ℕ) (cost_blue_hat : ℕ) (total_price : ℕ) (green_hats : ℕ) (blue_hats : ℕ) (cost_green_hat : ℕ)
  (h1 : total_hats = 85) 
  (h2 : cost_blue_hat = 6) 
  (h3 : total_price = 550) 
  (h4 : green_hats = 40) 
  (h5 : blue_hats = 45) 
  (h6 : green_hats + blue_hats = total_hats) 
  (h7 : total_price = green_hats * cost_green_hat + blue_hats * cost_blue_hat) :
  cost_green_hat = 7 := 
sorry

end cost_of_each_green_hat_l151_151863


namespace annual_interest_rate_l151_151437

-- Define the conditions as given in the problem
def principal : ℝ := 5000
def maturity_amount : ℝ := 5080
def interest_tax_rate : ℝ := 0.2

-- Define the annual interest rate x
variable (x : ℝ)

-- Statement to be proved: the annual interest rate x is 0.02
theorem annual_interest_rate :
  principal + principal * x - interest_tax_rate * (principal * x) = maturity_amount → x = 0.02 :=
by
  sorry

end annual_interest_rate_l151_151437


namespace find_value_l151_151169

theorem find_value (x : ℝ) (h : x^2 - x - 1 = 0) : 2 * x^2 - 2 * x + 2021 = 2023 := 
by 
  sorry -- Proof needs to be provided

end find_value_l151_151169


namespace total_emails_675_l151_151776

-- Definitions based on conditions
def emails_per_day_before : ℕ := 20
def extra_emails_per_day_after : ℕ := 5
def halfway_days : ℕ := 15
def total_days : ℕ := 30

-- Define the total number of emails received by the end of April
def total_emails_received : ℕ :=
  let emails_before := emails_per_day_before * halfway_days
  let emails_after := (emails_per_day_before + extra_emails_per_day_after) * halfway_days
  emails_before + emails_after

-- Theorem stating that the total number of emails received by the end of April is 675
theorem total_emails_675 : total_emails_received = 675 := by
  sorry

end total_emails_675_l151_151776


namespace factorize_expression_l151_151242

theorem factorize_expression (x y : ℝ) : 
  6 * x^3 * y^2 + 3 * x^2 * y^2 = 3 * x^2 * y^2 * (2 * x + 1) := 
by 
  sorry

end factorize_expression_l151_151242


namespace first_machine_copies_per_minute_l151_151996

theorem first_machine_copies_per_minute
    (x : ℕ)
    (h1 : ∀ (x : ℕ), 30 * x + 30 * 55 = 2850) :
  x = 40 :=
by
  sorry

end first_machine_copies_per_minute_l151_151996


namespace min_value_of_sum_l151_151319

theorem min_value_of_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : (1 / (2 * a)) + (1 / b) = 1) :
  a + 2 * b = 9 / 2 :=
sorry

end min_value_of_sum_l151_151319


namespace stopped_babysitting_16_years_ago_l151_151311

-- Definitions of given conditions
def started_babysitting_age (Jane_age_start : ℕ) := Jane_age_start = 16
def age_half_constraint (Jane_age child_age : ℕ) := child_age ≤ Jane_age / 2
def current_age (Jane_age_now : ℕ) := Jane_age_now = 32
def oldest_babysat_age_now (child_age_now : ℕ) := child_age_now = 24

-- The proposition to be proved
theorem stopped_babysitting_16_years_ago 
  (Jane_age_start Jane_age_now child_age_now : ℕ)
  (h1 : started_babysitting_age Jane_age_start)
  (h2 : ∀ (Jane_age child_age : ℕ), age_half_constraint Jane_age child_age → Jane_age > Jane_age_start → child_age_now = 24 → Jane_age = 24)
  (h3 : current_age Jane_age_now)
  (h4 : oldest_babysat_age_now child_age_now) :
  Jane_age_now - Jane_age_start = 16 :=
by sorry

end stopped_babysitting_16_years_ago_l151_151311


namespace ryan_sandwiches_l151_151628

theorem ryan_sandwiches (sandwich_slices : ℕ) (total_slices : ℕ) (h1 : sandwich_slices = 3) (h2 : total_slices = 15) :
  total_slices / sandwich_slices = 5 :=
by
  sorry

end ryan_sandwiches_l151_151628


namespace find_b_l151_151316

theorem find_b (a b : ℕ) (h1 : a = 105) (h2 : a^3 = 21 * 25 * 45 * b) : b = 49 :=
by
  sorry

end find_b_l151_151316


namespace find_x_l151_151705

variable (x y : ℝ)

theorem find_x (h1 : 0 < x) (h2 : 0 < y) (h3 : 5 * x^2 + 10 * x * y = x^3 + 2 * x^2 * y) : x = 5 := by
  sorry

end find_x_l151_151705


namespace average_first_18_even_numbers_l151_151600

theorem average_first_18_even_numbers : 
  let first_even := 2
  let difference := 2
  let n := 18
  let last_even := first_even + (n - 1) * difference
  let sum := (n / 2) * (first_even + last_even)
  let average := sum / n
  average = 19 :=
by
  -- Definitions
  let first_even := 2
  let difference := 2
  let n := 18
  let last_even := first_even + (n - 1) * difference
  let sum := (n / 2) * (first_even + last_even)
  let average := sum / n
  -- The claim
  show average = 19
  sorry

end average_first_18_even_numbers_l151_151600


namespace solution_to_fractional_equation_l151_151180

theorem solution_to_fractional_equation (x : ℝ) (h₁ : 2 / (x - 3) = 1 / x) (h₂ : x ≠ 3) (h₃ : x ≠ 0) : x = -3 :=
sorry

end solution_to_fractional_equation_l151_151180


namespace cost_per_rose_l151_151931

theorem cost_per_rose (P : ℝ) (h1 : 5 * 12 = 60) (h2 : 0.8 * 60 * P = 288) : P = 6 :=
by
  -- Proof goes here
  sorry

end cost_per_rose_l151_151931


namespace P_of_7_l151_151831

noncomputable def P (x : ℝ) : ℝ := 12 * (x - 1) * (x - 2) * (x - 3) * (x - 4)^2 * (x - 5)^2 * (x - 6)

theorem P_of_7 : P 7 = 51840 :=
by
  sorry

end P_of_7_l151_151831


namespace sufficient_but_not_necessary_condition_l151_151717

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x ≥ 1 → |x + 1| + |x - 1| = 2 * |x|)
  ∧ (∃ y : ℝ, ¬ (y ≥ 1) ∧ |y + 1| + |y - 1| = 2 * |y|) :=
by
  sorry

end sufficient_but_not_necessary_condition_l151_151717


namespace max_product_areas_l151_151773

theorem max_product_areas (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (h : a + b + c + d = 1) :
  a * b * c * d ≤ 1 / 256 :=
sorry

end max_product_areas_l151_151773


namespace area_of_path_cost_of_constructing_path_l151_151965

-- Definitions for the problem
def original_length : ℕ := 75
def original_width : ℕ := 40
def path_width : ℕ := 25 / 10  -- 2.5 converted to a Lean-readable form

-- Conditions
def new_length := original_length + 2 * path_width
def new_width := original_width + 2 * path_width

def area_with_path := new_length * new_width
def area_without_path := original_length * original_width

-- Statements to prove
theorem area_of_path : area_with_path - area_without_path = 600 := sorry

def cost_per_sq_m : ℕ := 2
def total_cost := (area_with_path - area_without_path) * cost_per_sq_m

theorem cost_of_constructing_path : total_cost = 1200 := sorry

end area_of_path_cost_of_constructing_path_l151_151965


namespace find_p_l151_151967

theorem find_p (p : ℝ) (h : 0 < p ∧ p < 1) : 
  p + (1 - p) * p + (1 - p)^2 * p = 0.784 → p = 0.4 :=
by
  intros h_eq
  sorry

end find_p_l151_151967


namespace value_of_f_2_plus_g_3_l151_151074

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 - 1

theorem value_of_f_2_plus_g_3 : f (2 + g 3) = 26 :=
by
  sorry

end value_of_f_2_plus_g_3_l151_151074


namespace negation_proof_converse_proof_l151_151253

-- Define the proposition
def prop_last_digit_zero_or_five (n : ℤ) : Prop := (n % 10 = 0) ∨ (n % 10 = 5)
def divisible_by_five (n : ℤ) : Prop := ∃ k : ℤ, n = 5 * k

-- Negation of the proposition
def negation_prop : Prop :=
  ∃ n : ℤ, prop_last_digit_zero_or_five n ∧ ¬ divisible_by_five n

-- Converse of the proposition
def converse_prop : Prop :=
  ∀ n : ℤ, ¬ prop_last_digit_zero_or_five n → ¬ divisible_by_five n

theorem negation_proof : negation_prop :=
  sorry  -- to be proved

theorem converse_proof : converse_prop :=
  sorry  -- to be proved

end negation_proof_converse_proof_l151_151253


namespace number_of_bookshelves_l151_151278

-- Definitions based on the conditions
def books_per_shelf : ℕ := 2
def total_books : ℕ := 38

-- Statement to prove
theorem number_of_bookshelves (books_per_shelf total_books : ℕ) : total_books / books_per_shelf = 19 :=
by sorry

end number_of_bookshelves_l151_151278


namespace trajectory_of_B_l151_151666

-- Define the points and the line for the given conditions
def A : ℝ × ℝ := (3, -1)
def C : ℝ × ℝ := (2, -3)
def D_line (x : ℝ) (y : ℝ) : Prop := 3 * x - y + 1 = 0

-- Define the statement to be proved
theorem trajectory_of_B (x y : ℝ) :
  D_line x y → ∃ Bx By, (3 * Bx - By - 20 = 0) :=
sorry

end trajectory_of_B_l151_151666


namespace circle_tangent_x_axis_at_origin_l151_151841

theorem circle_tangent_x_axis_at_origin (D E F : ℝ) :
  (∀ (x y : ℝ), x^2 + y^2 + D*x + E*y + F = 0 → (∃ r : ℝ, r^2 = x^2 + y^2) ∧ y = 0) →
  D = 0 ∧ E ≠ 0 ∧ F ≠ 0 := 
sorry

end circle_tangent_x_axis_at_origin_l151_151841


namespace smallest_b_factors_l151_151538

theorem smallest_b_factors (b p q : ℤ) (hb : b = p + q) (hpq : p * q = 2052) : b = 132 :=
sorry

end smallest_b_factors_l151_151538


namespace find_salary_l151_151017

variable (S : ℝ)
variable (house_rent_percentage : ℝ) (education_percentage : ℝ) (clothes_percentage : ℝ)
variable (remaining_amount : ℝ)

theorem find_salary (h1 : house_rent_percentage = 0.20)
                    (h2 : education_percentage = 0.10)
                    (h3 : clothes_percentage = 0.10)
                    (h4 : remaining_amount = 1377)
                    (h5 : (1 - clothes_percentage) * (1 - education_percentage) * (1 - house_rent_percentage) * S = remaining_amount) :
                    S = 2125 := 
sorry

end find_salary_l151_151017


namespace vector_subtraction_l151_151129

variable (a b : ℝ × ℝ)

def vector_calc (a b : ℝ × ℝ) : ℝ × ℝ :=
  (2 * a.1 - b.1, 2 * a.2 - b.2)

theorem vector_subtraction :
  a = (2, 4) → b = (-1, 1) → vector_calc a b = (5, 7) := by
  intros ha hb
  simp [vector_calc]
  rw [ha, hb]
  simp
  sorry

end vector_subtraction_l151_151129


namespace equilateral_triangle_octagon_area_ratio_l151_151947

theorem equilateral_triangle_octagon_area_ratio
  (s_t s_o : ℝ)
  (h_triangle_area : (s_t^2 * Real.sqrt 3) / 4 = 2 * s_o^2 * (1 + Real.sqrt 2)) :
  s_t / s_o = Real.sqrt (8 * Real.sqrt 3 * (1 + Real.sqrt 2) / 3) :=
by
  sorry

end equilateral_triangle_octagon_area_ratio_l151_151947


namespace product_mod_5_l151_151720

theorem product_mod_5 : (3 * 13 * 23 * 33 * 43 * 53 * 63 * 73 * 83 * 93) % 5 = 4 := 
by 
  sorry

end product_mod_5_l151_151720


namespace cricket_player_avg_runs_l151_151465

theorem cricket_player_avg_runs (A : ℝ) :
  (13 * A + 92 = 14 * (A + 5)) → A = 22 :=
by
  intro h1
  have h2 : 13 * A + 92 = 14 * A + 70 := by sorry
  have h3 : 92 - 70 = 14 * A - 13 * A := by sorry
  sorry

end cricket_player_avg_runs_l151_151465


namespace value_of_b_l151_151815

theorem value_of_b (a c : ℝ) (b : ℝ) (h1 : a = 105) (h2 : c = 70) (h3 : a^4 = 21 * 25 * 15 * b * c^3) : b = 0.045 :=
by
  sorry

end value_of_b_l151_151815


namespace quadratic_inequality_solution_l151_151223

theorem quadratic_inequality_solution (x : ℝ) : 
  (x^2 - 6 * x + 5 > 0) ↔ (x < 1 ∨ x > 5) :=
by sorry

end quadratic_inequality_solution_l151_151223


namespace julie_same_hours_september_october_l151_151592

-- Define Julie's hourly rates and work hours
def rate_mowing : ℝ := 4
def rate_weeding : ℝ := 8
def september_mowing_hours : ℕ := 25
def september_weeding_hours : ℕ := 3
def total_earnings_september_october : ℤ := 248

-- Define Julie's earnings for each activity and total earnings for September
def september_earnings_mowing : ℝ := september_mowing_hours * rate_mowing
def september_earnings_weeding : ℝ := september_weeding_hours * rate_weeding
def september_total_earnings : ℝ := september_earnings_mowing + september_earnings_weeding

-- Define earnings in October
def october_earnings : ℝ := total_earnings_september_october - september_total_earnings

-- Define the theorem to prove Julie worked the same number of hours in October as in September
theorem julie_same_hours_september_october :
  october_earnings = september_total_earnings :=
by
  sorry

end julie_same_hours_september_october_l151_151592


namespace popsicle_total_l151_151807

def popsicle_count (g c b : Nat) : Nat :=
  g + c + b

theorem popsicle_total : 
  let g := 2
  let c := 13
  let b := 2
  popsicle_count g c b = 17 := by
  sorry

end popsicle_total_l151_151807


namespace geom_progression_common_ratio_l151_151914

theorem geom_progression_common_ratio (x y z r : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (h7 : ∃ a, a ≠ 0 ∧ x * (2 * y - z) = a ∧ y * (2 * z - x) = a * r ∧ z * (2 * x - y) = a * r^2) :
  r^2 + r + 1 = 0 :=
sorry

end geom_progression_common_ratio_l151_151914


namespace number_of_packages_needed_l151_151432

-- Define the problem constants and constraints
def students_per_class := 30
def number_of_classes := 4
def buns_per_student := 2
def buns_per_package := 8

-- Calculate the total number of students
def total_students := number_of_classes * students_per_class

-- Calculate the total number of buns needed
def total_buns := total_students * buns_per_student

-- Calculate the required number of packages
def required_packages := total_buns / buns_per_package

-- Prove that the required number of packages is 30
theorem number_of_packages_needed : required_packages = 30 := by
  -- The proof would be here, but for now we assume it is correct
  sorry

end number_of_packages_needed_l151_151432


namespace largest_non_prime_sum_l151_151725

theorem largest_non_prime_sum (a b n : ℕ) (h1 : a ≥ 1) (h2 : b < 47) (h3 : n = 47 * a + b) (h4 : ∀ b, b < 47 → ¬Nat.Prime b → b = 43) : 
  n = 90 :=
by
  sorry

end largest_non_prime_sum_l151_151725


namespace triangle_similar_l151_151236

variables {a b c m_a m_b m_c t : ℝ}

-- Define the triangle ABC and its properties
def triangle_ABC (a b c m_a m_b m_c t : ℝ) : Prop :=
  t = (1 / 2) * a * m_a ∧
  t = (1 / 2) * b * m_b ∧
  t = (1 / 2) * c * m_c ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  m_a > 0 ∧ m_b > 0 ∧ m_c > 0 ∧
  t > 0

-- Define the similarity condition for the triangles
def similitude_from_reciprocals (a b c m_a m_b m_c t : ℝ) : Prop :=
  (1 / m_a) / (1 / m_b) = a / b ∧
  (1 / m_b) / (1 / m_c) = b / c ∧
  (1 / m_a) / (1 / m_c) = a / c

theorem triangle_similar (a b c m_a m_b m_c t : ℝ) :
  triangle_ABC a b c m_a m_b m_c t →
  similitude_from_reciprocals a b c m_a m_b m_c t :=
by
  intro h
  sorry

end triangle_similar_l151_151236


namespace min_x_y_sum_l151_151128

theorem min_x_y_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/(x+1) + 1/y = 1/2) : x + y ≥ 7 := 
by 
  sorry

end min_x_y_sum_l151_151128


namespace square_of_fourth_power_of_fourth_smallest_prime_l151_151615

-- Define the fourth smallest prime number
def fourth_smallest_prime : ℕ := 7

-- Define the square of the fourth power of that number
def square_of_fourth_power (n : ℕ) : ℕ := (n^4)^2

-- Prove the main statement
theorem square_of_fourth_power_of_fourth_smallest_prime : square_of_fourth_power fourth_smallest_prime = 5764801 :=
by
  sorry

end square_of_fourth_power_of_fourth_smallest_prime_l151_151615


namespace arithmetic_geom_seq_S5_l151_151328

theorem arithmetic_geom_seq_S5 (a_n : ℕ → ℚ) (S_n : ℕ → ℚ)
  (h_arith_seq : ∀ n, a_n n = a_n 1 + (n - 1) * (1/2))
  (h_sum : ∀ n, S_n n = n * a_n 1 + (n * (n - 1) / 2) * (1/2))
  (h_geom_seq : (a_n 2) * (a_n 14) = (a_n 6) ^ 2) :
  S_n 5 = 25 / 2 :=
by
  sorry

end arithmetic_geom_seq_S5_l151_151328


namespace find_M_l151_151306

theorem find_M (M : ℤ) (h1 : 22 < M) (h2 : M < 24) : M = 23 := by
  sorry

end find_M_l151_151306


namespace fishing_boat_should_go_out_to_sea_l151_151109

def good_weather_profit : ℤ := 6000
def bad_weather_loss : ℤ := -8000
def stay_at_port_loss : ℤ := -1000

def prob_good_weather : ℚ := 0.6
def prob_bad_weather : ℚ := 0.4

def expected_profit_going : ℚ :=  prob_good_weather * good_weather_profit + prob_bad_weather * bad_weather_loss
def expected_profit_staying : ℚ := stay_at_port_loss

theorem fishing_boat_should_go_out_to_sea : 
  expected_profit_going > expected_profit_staying :=
  sorry

end fishing_boat_should_go_out_to_sea_l151_151109


namespace smallest_number_greater_than_300_divided_by_25_has_remainder_24_l151_151464

theorem smallest_number_greater_than_300_divided_by_25_has_remainder_24 :
  ∃ x : ℕ, (x > 300) ∧ (x % 25 = 24) ∧ (x = 324) := by
  sorry

end smallest_number_greater_than_300_divided_by_25_has_remainder_24_l151_151464


namespace problem_statement_l151_151340

theorem problem_statement (x : ℝ) (h : x - 1/x = 5) : x^4 - (1 / x)^4 = 527 :=
sorry

end problem_statement_l151_151340


namespace total_people_participated_l151_151338

theorem total_people_participated 
  (N f p : ℕ)
  (h1 : N = f * p)
  (h2 : N = (f - 10) * (p + 1))
  (h3 : N = (f - 25) * (p + 3)) : 
  N = 900 :=
by 
  sorry

end total_people_participated_l151_151338


namespace james_running_increase_l151_151595

theorem james_running_increase (initial_miles_per_week : ℕ) (percent_increase : ℝ) (total_days : ℕ) (days_in_week : ℕ) :
  initial_miles_per_week = 100 →
  percent_increase = 0.2 →
  total_days = 280 →
  days_in_week = 7 →
  ∃ miles_per_week_to_add : ℝ, miles_per_week_to_add = 3 :=
by
  intros
  sorry

end james_running_increase_l151_151595


namespace length_decreased_by_l151_151062

noncomputable def length_decrease_proof : Prop :=
  let length := 33.333333333333336
  let breadth := length / 2
  let new_length := length - 2.833333333333336
  let new_breadth := breadth + 4
  let original_area := length * breadth
  let new_area := new_length * new_breadth
  (new_area = original_area + 75) ↔ (new_length = length - 2.833333333333336)

theorem length_decreased_by : length_decrease_proof := sorry

end length_decreased_by_l151_151062


namespace square_difference_l151_151883

theorem square_difference : (601^2 - 599^2 = 2400) :=
by {
  -- Placeholder for the proof
  sorry
}

end square_difference_l151_151883


namespace river_width_l151_151785

theorem river_width
  (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) (flow_rate_m_per_min : ℝ)
  (H_depth : depth = 5)
  (H_flow_rate_kmph : flow_rate_kmph = 4)
  (H_volume_per_minute : volume_per_minute = 6333.333333333333)
  (H_flow_rate_m_per_min : flow_rate_m_per_min = 66.66666666666667) :
  volume_per_minute / (depth * flow_rate_m_per_min) = 19 :=
by
  -- proof goes here
  sorry

end river_width_l151_151785


namespace problem_solution_l151_151852

theorem problem_solution (s t : ℕ) (hpos_s : 0 < s) (hpos_t : 0 < t) (h_eq : s * (s - t) = 29) : s + t = 57 :=
by
  sorry

end problem_solution_l151_151852


namespace three_Z_five_l151_151956

def Z (a b : ℤ) : ℤ := b + 10 * a - 3 * a^2

theorem three_Z_five : Z 3 5 = 8 := sorry

end three_Z_five_l151_151956


namespace stratified_sampling_l151_151202

theorem stratified_sampling (total_students : ℕ) (ratio_grade1 ratio_grade2 ratio_grade3 : ℕ) (sample_size : ℕ) (h_ratio : ratio_grade1 = 3 ∧ ratio_grade2 = 3 ∧ ratio_grade3 = 4) (h_sample_size : sample_size = 50) : 
  (ratio_grade2 / (ratio_grade1 + ratio_grade2 + ratio_grade3) : ℚ) * sample_size = 15 := 
by
  sorry

end stratified_sampling_l151_151202


namespace six_digit_mod7_l151_151188

theorem six_digit_mod7 (a b c d e f : ℕ) (N : ℕ) (h : N = 100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) (h_div7 : N % 7 = 0) :
    (10^5 * f + 10^4 * a + 10^3 * b + 10^2 * c + 10 * d + e) % 7 = 0 :=
by
  sorry

end six_digit_mod7_l151_151188


namespace min_power_for_84_to_divide_336_l151_151161

theorem min_power_for_84_to_divide_336 : 
  ∃ n : ℕ, (∀ m : ℕ, 84^m % 336 = 0 → m ≥ n) ∧ n = 2 := 
sorry

end min_power_for_84_to_divide_336_l151_151161


namespace quadratic_roots_real_equal_l151_151214

theorem quadratic_roots_real_equal (m : ℝ) :
  (∃ a b c : ℝ, a ≠ 0 ∧ a = 3 ∧ b = 2 - m ∧ c = 6 ∧
    (b^2 - 4 * a * c = 0)) ↔ (m = 2 - 6 * Real.sqrt 2 ∨ m = 2 + 6 * Real.sqrt 2) :=
by
  sorry

end quadratic_roots_real_equal_l151_151214


namespace x_divisible_by_5_l151_151521

theorem x_divisible_by_5
  (x y : ℕ)
  (h_pos_x : 0 < x)
  (h_pos_y : 0 < y)
  (h_gt_1 : 1 < x)
  (h_eq : 2 * x^2 - 1 = y^15) : x % 5 = 0 :=
sorry

end x_divisible_by_5_l151_151521


namespace acid_percentage_in_original_mixture_l151_151370

theorem acid_percentage_in_original_mixture 
  {a w : ℕ} 
  (h1 : a / (a + w + 1) = 1 / 5) 
  (h2 : (a + 1) / (a + w + 2) = 1 / 3) : 
  a / (a + w) = 1 / 4 :=
sorry

end acid_percentage_in_original_mixture_l151_151370


namespace max_p_l151_151673

theorem max_p (p q r s t u v w : ℕ)
  (h1 : p + q + r + s = 35)
  (h2 : q + r + s + t = 35)
  (h3 : r + s + t + u = 35)
  (h4 : s + t + u + v = 35)
  (h5 : t + u + v + w = 35)
  (h6 : q + v = 14) :
  p ≤ 20 :=
sorry

end max_p_l151_151673


namespace vincent_earnings_l151_151759

def fantasy_book_cost : ℕ := 6
def literature_book_cost : ℕ := fantasy_book_cost / 2
def mystery_book_cost : ℕ := 4

def fantasy_books_sold_per_day : ℕ := 5
def literature_books_sold_per_day : ℕ := 8
def mystery_books_sold_per_day : ℕ := 3

def daily_earnings : ℕ :=
  (fantasy_books_sold_per_day * fantasy_book_cost) +
  (literature_books_sold_per_day * literature_book_cost) +
  (mystery_books_sold_per_day * mystery_book_cost)

def total_earnings_after_seven_days : ℕ :=
  daily_earnings * 7

theorem vincent_earnings : total_earnings_after_seven_days = 462 :=
by
  sorry

end vincent_earnings_l151_151759


namespace max_value_n_for_positive_an_l151_151356

-- Define the arithmetic sequence
noncomputable def arithmetic_seq (a d : ℤ) (n : ℤ) := a + (n - 1) * d

-- Define the sum of first n terms of an arithmetic sequence
noncomputable def sum_arith_seq (a d n : ℤ) := (n * (2 * a + (n - 1) * d)) / 2

-- Given conditions
axiom S15_pos (a d : ℤ) : sum_arith_seq a d 15 > 0
axiom S16_neg (a d : ℤ) : sum_arith_seq a d 16 < 0

-- Proof problem
theorem max_value_n_for_positive_an (a d : ℤ) :
  ∃ n : ℤ, n = 8 ∧ ∀ m : ℤ, (1 ≤ m ∧ m ≤ 8) → arithmetic_seq a d m > 0 :=
sorry

end max_value_n_for_positive_an_l151_151356


namespace min_distance_curveC1_curveC2_l151_151825

-- Definitions of the conditions
def curveC1 (P : ℝ × ℝ) : Prop :=
  ∃ θ : ℝ, P.1 = 3 + Real.cos θ ∧ P.2 = 4 + Real.sin θ

def curveC2 (P : ℝ × ℝ) : Prop :=
  P.1^2 + P.2^2 = 1

-- Proof statement
theorem min_distance_curveC1_curveC2 :
  (∀ A B : ℝ × ℝ,
    curveC1 A →
    curveC2 B →
    ∃ m : ℝ, m = 3 ∧ ∀ d : ℝ, (d = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) → d ≥ m) := 
  sorry

end min_distance_curveC1_curveC2_l151_151825


namespace imaginary_part_of_z_l151_151080

-- Define the complex number z
def z : ℂ :=
  3 - 2 * Complex.I

-- Lean theorem statement to prove the imaginary part of z is -2
theorem imaginary_part_of_z :
  Complex.im z = -2 :=
by
  sorry

end imaginary_part_of_z_l151_151080


namespace new_person_weight_l151_151320

variable {W : ℝ} -- Total weight of the original group of 15 people
variable {N : ℝ} -- Weight of the new person

theorem new_person_weight
  (avg_increase : (W - 90 + N) / 15 = (W - 90) / 14 + 3.7)
  : N = 55.5 :=
sorry

end new_person_weight_l151_151320


namespace not_possible_100_odd_sequence_l151_151115

def is_square_mod_8 (n : ℤ) : Prop :=
  n % 8 = 0 ∨ n % 8 = 1 ∨ n % 8 = 4

def sum_consecutive_is_square_mod_8 (seq : List ℤ) (k : ℕ) : Prop :=
  ∀ i : ℕ, i + k ≤ seq.length →
  is_square_mod_8 (seq.drop i |>.take k |>.sum)

def valid_odd_sequence (seq : List ℤ) : Prop :=
  seq.length = 100 ∧
  (∀ n ∈ seq, n % 2 = 1) ∧
  sum_consecutive_is_square_mod_8 seq 5 ∧
  sum_consecutive_is_square_mod_8 seq 9

theorem not_possible_100_odd_sequence :
  ¬∃ seq : List ℤ, valid_odd_sequence seq :=
by
  sorry

end not_possible_100_odd_sequence_l151_151115


namespace phone_price_increase_is_40_percent_l151_151419

-- Definitions based on the conditions
def initial_price_tv := 500
def increased_fraction_tv := 2 / 5
def initial_price_phone := 400
def total_amount_received := 1260

-- The price increase of the TV
def final_price_tv := initial_price_tv * (1 + increased_fraction_tv)

-- The final price of the phone
def final_price_phone := total_amount_received - final_price_tv

-- The percentage increase in the phone's price
def percentage_increase_phone := ((final_price_phone - initial_price_phone) / initial_price_phone) * 100

-- The theorem to prove
theorem phone_price_increase_is_40_percent :
  percentage_increase_phone = 40 := by
  sorry

end phone_price_increase_is_40_percent_l151_151419


namespace fraction_value_l151_151719

variable (a b : ℚ)  -- Variables a and b are rational numbers

theorem fraction_value (h : a / 4 = b / 3) : (a - b) / b = 1 / 3 := by
  sorry

end fraction_value_l151_151719


namespace man_rate_in_still_water_l151_151968

theorem man_rate_in_still_water (speed_with_stream speed_against_stream : ℝ)
  (h1 : speed_with_stream = 22) (h2 : speed_against_stream = 10) :
  (speed_with_stream + speed_against_stream) / 2 = 16 := by
  sorry

end man_rate_in_still_water_l151_151968


namespace expression_evaluation_l151_151888

-- Using the given conditions
def a : ℕ := 3
def b : ℕ := a^2 + 2 * a + 5
def c : ℕ := b^2 - 14 * b + 45

-- We need to assume that none of the denominators are zero.
lemma non_zero_denominators : (a + 1 ≠ 0) ∧ (b - 3 ≠ 0) ∧ (c + 7 ≠ 0) :=
  by {
    -- Proof goes here
  sorry }

theorem expression_evaluation :
  (a = 3) →
  ((a^2 + 2*a + 5) = b) →
  ((b^2 - 14*b + 45) = c) →
  (a + 1 ≠ 0) →
  (b - 3 ≠ 0) →
  (c + 7 ≠ 0) →
  (↑(a + 3) / ↑(a + 1) * ↑(b - 1) / ↑(b - 3) * ↑(c + 9) / ↑(c + 7) = 4923 / 2924) :=
  by {
    -- Proof goes here
  sorry }

end expression_evaluation_l151_151888


namespace determine_k_l151_151925

variable (x y z w : ℝ)

theorem determine_k
  (h₁ : 9 / (x + y + w) = k / (x + z + w))
  (h₂ : k / (x + z + w) = 12 / (z - y)) :
  k = 21 :=
sorry

end determine_k_l151_151925


namespace min_value_of_a_plus_b_l151_151454

theorem min_value_of_a_plus_b (a b c : ℝ) (C : ℝ) 
  (hC : C = 60) 
  (h : (a + b)^2 - c^2 = 4) : 
  a + b ≥ (4 * Real.sqrt 3) / 3 :=
by
  sorry

end min_value_of_a_plus_b_l151_151454


namespace vertex_angle_is_130_8_l151_151309

-- Define the given conditions
variables {a b h : ℝ}

def is_isosceles_triangle (a b h : ℝ) : Prop :=
  a^2 = b * 3 * h ∧ b = 2 * h

-- Define the obtuse condition on the vertex angle
def vertex_angle_obtuse (a b h : ℝ) : Prop :=
  ∃ θ : ℝ, 120 < θ ∧ θ < 180 ∧ θ = (130.8 : ℝ)

-- The formal proof statement using Lean 4
theorem vertex_angle_is_130_8 (a b h : ℝ) 
  (h1 : is_isosceles_triangle a b h)
  (h2 : vertex_angle_obtuse a b h) : 
  ∃ (φ : ℝ), φ = 130.8 :=
sorry

end vertex_angle_is_130_8_l151_151309


namespace cost_of_producing_one_component_l151_151501

-- Define the conditions as constants
def shipping_cost_per_unit : ℕ := 5
def fixed_monthly_cost : ℕ := 16500
def components_per_month : ℕ := 150
def selling_price_per_component : ℕ := 195

-- Define the cost of producing one component as a variable
variable (C : ℕ)

/-- Prove that C must be less than or equal to 80 given the conditions -/
theorem cost_of_producing_one_component : 
  150 * C + 150 * shipping_cost_per_unit + fixed_monthly_cost ≤ 150 * selling_price_per_component → C ≤ 80 :=
by
  sorry

end cost_of_producing_one_component_l151_151501


namespace interval_of_x_l151_151286

theorem interval_of_x (x : ℝ) :
  (2 < 4 * x ∧ 4 * x < 3) ∧ (2 < 5 * x ∧ 5 * x < 3) ↔ (1 / 2 < x ∧ x < 3 / 5) := by
  sorry

end interval_of_x_l151_151286


namespace fraction_in_between_l151_151484

variable {r u s v : ℤ}

/-- Assumes r, u, s, v be positive integers such that su - rv = 1 --/
theorem fraction_in_between (h1 : r > 0) (h2 : u > 0) (h3 : s > 0) (h4 : v > 0) (h5 : s * u - r * v = 1) :
  ∀ ⦃x num denom : ℤ⦄, r * denom = num * u → s * denom = (num + 1) * v → r * v ≤ num * denom - 1 / u * v * denom
   ∧ num * denom - 1 / u * v * denom ≤ s * v :=
sorry

end fraction_in_between_l151_151484


namespace polynomial_evaluation_l151_151457

theorem polynomial_evaluation (P : ℕ → ℝ) (n : ℕ) 
  (h_degree : ∀ k : ℕ, k ≤ n → P k = k / (k + 1)) 
  (h_poly : ∀ k : ℕ, ∃ a : ℝ, P k = a * k ^ n) : 
  P (n + 1) = (n + 1 + (-1) ^ (n + 1)) / (n + 2) :=
by 
  sorry

end polynomial_evaluation_l151_151457


namespace symmetric_point_is_correct_l151_151467

/-- A point in 2D Cartesian coordinates -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Defining the point P with given coordinates -/
def P : Point := {x := 2, y := 3}

/-- Defining the symmetry of a point with respect to the origin -/
def symmetric_origin (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- States that the symmetric point of P (2, 3) with respect to the origin is (-2, -3) -/
theorem symmetric_point_is_correct :
  symmetric_origin P = {x := -2, y := -3} :=
by
  sorry

end symmetric_point_is_correct_l151_151467


namespace domain_of_f2x_l151_151879

theorem domain_of_f2x (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → ∃ y, f y = f x) : 
  ∀ x, 0 ≤ x ∧ x ≤ 1 → ∃ y, f y = f (2 * x) :=
by
  sorry

end domain_of_f2x_l151_151879


namespace john_spent_l151_151977

-- Given definitions from the conditions.
def total_time_in_hours := 4
def additional_minutes := 35
def break_time_per_break := 10
def number_of_breaks := 5
def cost_per_5_minutes := 0.75
def playing_cost (total_time_in_hours additional_minutes break_time_per_break number_of_breaks : ℕ) 
  (cost_per_5_minutes : ℝ) : ℝ :=
  let total_minutes := total_time_in_hours * 60 + additional_minutes
  let break_time := number_of_breaks * break_time_per_break
  let actual_playing_time := total_minutes - break_time
  let number_of_intervals := actual_playing_time / 5
  number_of_intervals * cost_per_5_minutes

-- Statement to be proved.
theorem john_spent (total_time_in_hours := 4) (additional_minutes := 35) (break_time_per_break := 10) 
  (number_of_breaks := 5) (cost_per_5_minutes := 0.75) :
  playing_cost total_time_in_hours additional_minutes break_time_per_break number_of_breaks cost_per_5_minutes = 33.75 := 
by
  sorry

end john_spent_l151_151977


namespace percent_problem_l151_151010

theorem percent_problem (x : ℝ) (h : 0.20 * x = 60) : 0.80 * x = 240 :=
sorry

end percent_problem_l151_151010


namespace negation_example_l151_151178

theorem negation_example : ¬(∀ x : ℝ, x^2 + |x| ≥ 0) ↔ ∃ x : ℝ, x^2 + |x| < 0 :=
by
  sorry

end negation_example_l151_151178


namespace clock_ticks_12_times_l151_151995

theorem clock_ticks_12_times (t1 t2 : ℕ) (d1 d2 : ℕ) (h1 : t1 = 6) (h2 : d1 = 40) (h3 : d2 = 88) : t2 = 12 := by
  sorry

end clock_ticks_12_times_l151_151995


namespace total_texts_received_l151_151724

structure TextMessageScenario :=
  (textsBeforeNoon : Nat)
  (textsAtNoon : Nat)
  (textsAfterNoonDoubling : (Nat → Nat) → Nat)
  (textsAfter6pm : (Nat → Nat) → Nat)

def textsBeforeNoon := 21
def textsAtNoon := 2

-- Calculation for texts received from noon to 6 pm
def noonTo6pmTexts (textsAtNoon : Nat) : Nat :=
  let rec doubling (n : Nat) : Nat := match n with
    | 0 => textsAtNoon
    | n + 1 => 2 * (doubling n)
  (doubling 0) + (doubling 1) + (doubling 2) + (doubling 3) + (doubling 4) + (doubling 5)

def textsAfterNoonDoubling : (Nat → Nat) → Nat := λ doubling => noonTo6pmTexts 2

-- Calculation for texts received from 6 pm to midnight
def after6pmTexts (textsAt6pm : Nat) : Nat :=
  let rec decrease (n : Nat) : Nat := match n with
    | 0 => textsAt6pm
    | n + 1 => (decrease n) - 5
  (decrease 0) + (decrease 1) + (decrease 2) + (decrease 3) + (decrease 4) + (decrease 5) + (decrease 6)

def textsAfter6pm : (Nat → Nat) → Nat := λ decrease => after6pmTexts 64

theorem total_texts_received : textsBeforeNoon + (textsAfterNoonDoubling (λ x => x)) + (textsAfter6pm (λ x => x)) = 490 := by
  sorry
 
end total_texts_received_l151_151724


namespace intersection_M_N_eq_M_inter_N_l151_151264

def M : Set ℝ := { x | x^2 - 4 > 0 }
def N : Set ℝ := { x | x < 0 }
def M_inter_N : Set ℝ := { x | x < -2 }

theorem intersection_M_N_eq_M_inter_N : M ∩ N = M_inter_N := 
by
  sorry

end intersection_M_N_eq_M_inter_N_l151_151264


namespace page_number_added_twice_l151_151613

-- Define the sum of natural numbers from 1 to n
def sum_nat (n: ℕ): ℕ := n * (n + 1) / 2

-- Incorrect sum due to one page number being counted twice
def incorrect_sum (n p: ℕ): ℕ := sum_nat n + p

-- Declaring the known conditions as Lean definitions
def n : ℕ := 70
def incorrect_sum_val : ℕ := 2550

-- Lean theorem statement to be proven
theorem page_number_added_twice :
  ∃ p, incorrect_sum n p = incorrect_sum_val ∧ p = 65 := by
  sorry

end page_number_added_twice_l151_151613


namespace min_value_of_y_l151_151023

noncomputable def y (x : ℝ) : ℝ :=
  2 * Real.sin (Real.pi / 3 - x) - Real.cos (Real.pi / 6 + x)

theorem min_value_of_y : ∃ x : ℝ, y x = -1 := by
  sorry

end min_value_of_y_l151_151023


namespace lattice_points_on_sphere_at_distance_5_with_x_1_l151_151179

theorem lattice_points_on_sphere_at_distance_5_with_x_1 :
  let points := [(1, 0, 4), (1, 0, -4), (1, 4, 0), (1, -4, 0),
                 (1, 2, 4), (1, 2, -4), (1, -2, 4), (1, -2, -4),
                 (1, 4, 2), (1, 4, -2), (1, -4, 2), (1, -4, -2),
                 (1, 2, 2), (1, 2, -2), (1, -2, 2), (1, -2, -2)]
  (hs : ∀ y z, (1, y, z) ∈ points → 1^2 + y^2 + z^2 = 25) →
  24 = points.length :=
sorry

end lattice_points_on_sphere_at_distance_5_with_x_1_l151_151179


namespace rebus_solution_l151_151650

theorem rebus_solution :
  ∃ (A B C : ℕ), A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C ∧
    (A*100 + B*10 + A) + (A*100 + B*10 + C) + (A*100 + C*10 + C) = 1416 ∧ 
    A = 4 ∧ B = 7 ∧ C = 6 :=
by {
  sorry
}

end rebus_solution_l151_151650


namespace cone_base_diameter_l151_151908

theorem cone_base_diameter
  (h_cone : ℝ) (r_sphere : ℝ) (waste_percentage : ℝ) (d : ℝ) :
  h_cone = 9 → r_sphere = 9 → waste_percentage = 0.75 → 
  (V_cone = 1/3 * π * (d/2)^2 * h_cone) →
  (V_sphere = 4/3 * π * r_sphere^3) →
  (V_cone = (1 - waste_percentage) * V_sphere) →
  d = 9 :=
by
  intros h_cond r_cond waste_cond v_cone_eq v_sphere_eq v_cone_sphere_eq
  sorry

end cone_base_diameter_l151_151908


namespace day_before_yesterday_l151_151425

theorem day_before_yesterday (day_after_tomorrow_is_monday : String) : String :=
by
  have tomorrow := "Sunday"
  have today := "Saturday"
  exact today

end day_before_yesterday_l151_151425


namespace angle_through_point_l151_151012

theorem angle_through_point : 
  (∃ θ : ℝ, ∃ k : ℤ, θ = 2 * k * Real.pi + 5 * Real.pi / 6 ∧ 
                      ∃ x y : ℝ, x = -Real.sqrt 3 / 2 ∧ y = 1 / 2 ∧ 
                                    y / x = Real.tan θ) := 
sorry

end angle_through_point_l151_151012


namespace arlo_stationery_count_l151_151043

theorem arlo_stationery_count (books pens : ℕ) (ratio_books_pens : ℕ × ℕ) (total_books : ℕ)
  (h_ratio : ratio_books_pens = (7, 3)) (h_books : total_books = 280) :
  books + pens = 400 :=
by
  sorry

end arlo_stationery_count_l151_151043


namespace fraction_of_red_knights_magical_l151_151297

variable {knights : ℕ}
variable {red_knights : ℕ}
variable {blue_knights : ℕ}
variable {magical_knights : ℕ}
variable {magical_red_knights : ℕ}
variable {magical_blue_knights : ℕ}

axiom total_knights : knights > 0
axiom red_knights_fraction : red_knights = (3 * knights) / 8
axiom blue_knights_fraction : blue_knights = (5 * knights) / 8
axiom magical_knights_fraction : magical_knights = knights / 4
axiom magical_fraction_relation : 3 * magical_blue_knights = magical_red_knights

theorem fraction_of_red_knights_magical :
  (magical_red_knights : ℚ) / red_knights = 3 / 7 :=
by
  sorry

end fraction_of_red_knights_magical_l151_151297


namespace total_time_is_12_years_l151_151172

noncomputable def total_time_spent (shape_years climb_years_per_summit dive_months cave_years : ℕ) : ℕ :=
  shape_years + (2 * shape_years) + (7 * climb_years_per_summit) / 12 + ((7 * climb_years_per_summit) % 12) / 12 + (dive_months + 12) / 12 + cave_years

theorem total_time_is_12_years :
  total_time_spent 2 5 13 2 = 12 :=
by
  sorry

end total_time_is_12_years_l151_151172


namespace remainder_of_266_div_33_and_8_is_2_l151_151126

theorem remainder_of_266_div_33_and_8_is_2 :
  (266 % 33 = 2) ∧ (266 % 8 = 2) := by
  sorry

end remainder_of_266_div_33_and_8_is_2_l151_151126


namespace find_f1_and_f_prime1_l151_151970

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)

-- Conditions
axiom differentiable_f : Differentiable ℝ f
axiom f_def : ∀ x : ℝ, f x = 2 * x^2 - f' 1 * x - 3

-- Proof using conditions
theorem find_f1_and_f_prime1 : f 1 + (f' 1) = -1 :=
sorry

end find_f1_and_f_prime1_l151_151970


namespace largest_divisor_prime_cube_diff_l151_151426

theorem largest_divisor_prime_cube_diff (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge5 : p ≥ 5) : 
  ∃ k, k = 12 ∧ ∀ n, n ∣ (p^3 - p) ↔ n ∣ 12 :=
by
  sorry

end largest_divisor_prime_cube_diff_l151_151426


namespace total_jelly_beans_l151_151848

theorem total_jelly_beans (vanilla_jelly_beans : ℕ) (h1 : vanilla_jelly_beans = 120) 
                           (grape_jelly_beans : ℕ) (h2 : grape_jelly_beans = 5 * vanilla_jelly_beans + 50) :
                           vanilla_jelly_beans + grape_jelly_beans = 770 :=
by
  sorry

end total_jelly_beans_l151_151848


namespace sandra_beignets_l151_151821

theorem sandra_beignets (beignets_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) (h1 : beignets_per_day = 3) (h2 : days_per_week = 7) (h3: weeks = 16) : 
  (beignets_per_day * days_per_week * weeks) = 336 :=
by {
  -- the proof goes here
  sorry
}

end sandra_beignets_l151_151821


namespace factor_expression_l151_151706

theorem factor_expression (x : ℝ) : 16 * x ^ 2 + 8 * x = 8 * x * (2 * x + 1) :=
by
  -- Problem: Completely factor the expression
  -- Given Condition
  -- Conclusion
  sorry

end factor_expression_l151_151706


namespace find_m_l151_151604

theorem find_m 
  (m : ℤ) 
  (h1 : ∀ x y : ℤ, -3 * x + y = m → 2 * x + y = 28 → x = -6) : 
  m = 58 :=
by 
  sorry

end find_m_l151_151604


namespace inequality_ineq_l151_151200

variable (x y z : Real)

theorem inequality_ineq {x y z : Real} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x^2 + y^2 + z^2 = 3) :
  (1 / (x^5 - x^2 + 3)) + (1 / (y^5 - y^2 + 3)) + (1 / (z^5 - z^2 + 3)) ≤ 1 :=
by 
  sorry

end inequality_ineq_l151_151200


namespace gcd_lcm_product_l151_151960

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 30) (h2 : b = 75) :
  (Nat.gcd a b) * (Nat.lcm a b) = 2250 := by
  sorry

end gcd_lcm_product_l151_151960


namespace circle_symmetric_eq_l151_151463

theorem circle_symmetric_eq :
  ∀ (x y : ℝ), (x^2 + y^2 + 2 * x - 2 * y + 1 = 0) → (x - y + 3 = 0) → 
  (∃ (a b : ℝ), (a + 2)^2 + (b - 2)^2 = 1) :=
by
  intros x y hc hl
  sorry

end circle_symmetric_eq_l151_151463


namespace moses_income_l151_151176

theorem moses_income (investment : ℝ) (percentage : ℝ) (dividend_rate : ℝ) (income : ℝ)
  (h1 : investment = 3000) (h2 : percentage = 0.72) (h3 : dividend_rate = 0.0504) :
  income = 210 :=
sorry

end moses_income_l151_151176


namespace max_value_of_d_l151_151675

-- Define the conditions
variable (a b c d : ℝ) (h_sum : a + b + c + d = 10) 
          (h_prod_sum : ab + ac + ad + bc + bd + cd = 20)

-- Define the theorem statement
theorem max_value_of_d : 
  d ≤ (5 + Real.sqrt 105) / 2 :=
sorry

end max_value_of_d_l151_151675


namespace inequality_solution_l151_151781

theorem inequality_solution :
  {x : ℝ | -x^2 - |x| + 6 > 0} = {x : ℝ | -2 < x ∧ x < 2} :=
sorry

end inequality_solution_l151_151781


namespace flight_duration_l151_151561

theorem flight_duration (departure_time arrival_time : ℕ) (time_difference : ℕ) (h m : ℕ) (m_bound : 0 < m ∧ m < 60) 
  (h_val : h = 1) (m_val : m = 35)  : h + m = 36 := by
  sorry

end flight_duration_l151_151561


namespace find_special_numbers_l151_151102

theorem find_special_numbers (N : ℕ) (hN : 100 ≤ N ∧ N < 1000) :
  (∀ k : ℕ, N^k % 1000 = N % 1000) ↔ (N = 376 ∨ N = 625) :=
by
  sorry

end find_special_numbers_l151_151102


namespace find_smallest_nat_with_remainder_2_l151_151051

noncomputable def smallest_nat_with_remainder_2 : Nat :=
    let x := 26
    if x > 0 ∧ x ≡ 2 [MOD 3] 
                 ∧ x ≡ 2 [MOD 4] 
                 ∧ x ≡ 2 [MOD 6] 
                 ∧ x ≡ 2 [MOD 8] then x
    else 0

theorem find_smallest_nat_with_remainder_2 :
    ∃ x : Nat, x > 0 ∧ x ≡ 2 [MOD 3] 
                 ∧ x ≡ 2 [MOD 4] 
                 ∧ x ≡ 2 [MOD 6] 
                 ∧ x ≡ 2 [MOD 8] ∧ x = smallest_nat_with_remainder_2 :=
    sorry

end find_smallest_nat_with_remainder_2_l151_151051


namespace find_n_divides_polynomial_l151_151046

theorem find_n_divides_polynomial :
  ∀ (n : ℕ), 0 < n → (n + 2) ∣ (n^3 + 3 * n + 29) ↔ (n = 1 ∨ n = 3 ∨ n = 13) :=
by
  sorry

end find_n_divides_polynomial_l151_151046


namespace evaluate_f_x_l151_151339

def f (x : ℝ) : ℝ := x^5 + 3 * x^3 + 2 * x^2 + 4 * x

theorem evaluate_f_x : f 3 - f (-3) = 672 :=
by
  -- Proof omitted
  sorry

end evaluate_f_x_l151_151339


namespace part1_part2_l151_151789

open Real

noncomputable def condition1 (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a^2 + 3 * b^2 = 3

theorem part1 {a b : ℝ} (h : condition1 a b) : sqrt 5 * a + b ≤ 4 := 
sorry

theorem part2 {x a b : ℝ} (h₁ : condition1 a b) (h₂ : 2 * abs (x - 1) + abs x ≥ 4) : 
x ≤ -2/3 ∨ x ≥ 2 := 
sorry

end part1_part2_l151_151789


namespace find_number_l151_151812

theorem find_number (x : ℝ) (h : (2 / 5) * x = 10) : x = 25 :=
sorry

end find_number_l151_151812


namespace furthest_distance_l151_151097

-- Definitions of point distances as given conditions
def PQ : ℝ := 13
def QR : ℝ := 11
def RS : ℝ := 14
def SP : ℝ := 12

-- Statement of the problem in Lean
theorem furthest_distance :
  ∃ (P Q R S : ℝ),
    |P - Q| = PQ ∧
    |Q - R| = QR ∧
    |R - S| = RS ∧
    |S - P| = SP ∧
    ∀ (a b : ℝ), a ≠ b →
      |a - b| ≤ 25 :=
sorry

end furthest_distance_l151_151097


namespace probability_red_or_white_l151_151702

noncomputable def total_marbles : ℕ := 50
noncomputable def blue_marbles : ℕ := 5
noncomputable def red_marbles : ℕ := 9
noncomputable def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white : 
  (red_marbles + white_marbles) / total_marbles = 9 / 10 :=
by sorry

end probability_red_or_white_l151_151702


namespace option_one_cost_option_two_cost_cost_effectiveness_l151_151091

-- Definition of costs based on conditions
def price_of_suit : ℕ := 500
def price_of_tie : ℕ := 60
def discount_option_one (x : ℕ) : ℕ := 60 * x + 8800
def discount_option_two (x : ℕ) : ℕ := 54 * x + 9000

-- Theorem statements
theorem option_one_cost (x : ℕ) (hx : x > 20) : discount_option_one x = 60 * x + 8800 :=
by sorry

theorem option_two_cost (x : ℕ) (hx : x > 20) : discount_option_two x = 54 * x + 9000 :=
by sorry

theorem cost_effectiveness (x : ℕ) (hx : x = 30) : discount_option_one x < discount_option_two x :=
by sorry

end option_one_cost_option_two_cost_cost_effectiveness_l151_151091


namespace company_workers_l151_151771

theorem company_workers (W : ℕ) (H1 : (1/3 : ℚ) * W = ((1/3 : ℚ) * W)) 
  (H2 : 0.20 * ((1/3 : ℚ) * W) = ((1/15 : ℚ) * W)) 
  (H3 : 0.40 * ((2/3 : ℚ) * W) = ((4/15 : ℚ) * W)) 
  (H4 : (4/15 : ℚ) * W + (4/15 : ℚ) * W = 160)
  : (W - 160 = 140) :=
by
  sorry

end company_workers_l151_151771


namespace value_of_a_l151_151644

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem value_of_a (a : ℝ) : 
  (∀ x, deriv (f a) x = 6 * x + 3 * a * x^2) →
  deriv (f a) (-1) = 6 → a = 4 :=
by
  -- Proof will be filled in here
  sorry

end value_of_a_l151_151644


namespace find_50th_term_arithmetic_sequence_l151_151285

theorem find_50th_term_arithmetic_sequence :
  let a₁ := 3
  let d := 7
  let a₅₀ := a₁ + (50 - 1) * d
  a₅₀ = 346 :=
by
  let a₁ := 3
  let d := 7
  let a₅₀ := a₁ + (50 - 1) * d
  show a₅₀ = 346
  sorry

end find_50th_term_arithmetic_sequence_l151_151285


namespace polar_equation_of_circle_slope_of_line_l151_151227

-- Part 1: Polar equation of circle C
theorem polar_equation_of_circle (x y : ℝ) :
  (x - 2) ^ 2 + y ^ 2 = 9 -> ∃ (ρ θ : ℝ), ρ^2 - 4*ρ*Real.cos θ - 5 = 0 := 
sorry

-- Part 2: Slope of line L intersecting C at points A and B
theorem slope_of_line (α : ℝ) (L : ℝ → ℝ × ℝ) (A B : ℝ × ℝ) :
  (∀ t, L t = (t * Real.cos α, t * Real.sin α)) ∧ dist A B = 2 * Real.sqrt 7 ∧ 
  (∃ x y, (x - 2) ^ 2 + y ^ 2 = 9 ∧ L (Real.sqrt ((x - 2) ^ 2 + y ^ 2)) = (x, y))
  -> Real.tan α = 1 ∨ Real.tan α = -1 :=
sorry

end polar_equation_of_circle_slope_of_line_l151_151227


namespace p_implies_q_not_q_implies_p_l151_151893

def p (a : ℝ) := a = Real.sqrt 2

def q (a : ℝ) := ∀ x y : ℝ, y = -(x : ℝ) → (x^2 + (y - a)^2 = 1)

theorem p_implies_q_not_q_implies_p (a : ℝ) : (p a → q a) ∧ (¬(q a → p a)) := 
    sorry

end p_implies_q_not_q_implies_p_l151_151893


namespace correct_avg_weight_l151_151528

theorem correct_avg_weight (initial_avg_weight : ℚ) (num_boys : ℕ) (misread_weight : ℚ) (correct_weight : ℚ) :
  initial_avg_weight = 58.4 → num_boys = 20 → misread_weight = 56 → correct_weight = 60 →
  (initial_avg_weight * num_boys + (correct_weight - misread_weight)) / num_boys = 58.6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Plugging in the values makes the calculation straightforward, resulting in: 
  -- (58.4 * 20 + (60 - 56)) / 20 = 58.6 
  -- thus this verification step is:
  sorry

end correct_avg_weight_l151_151528


namespace sum_of_coefficients_of_parabolas_kite_formed_l151_151190

theorem sum_of_coefficients_of_parabolas_kite_formed (a b : ℝ) 
  (h1 : ∃ (x : ℝ), y = ax^2 - 4)
  (h2 : ∃ (y : ℝ), y = 6 - bx^2)
  (h3 : (a > 0) ∧ (b > 0) ∧ (ax^2 - 4 = 0) ∧ (6 - bx^2 = 0))
  (h4 : kite_area = 18) :
  a + b = 125/36 := 
by sorry

end sum_of_coefficients_of_parabolas_kite_formed_l151_151190


namespace solution_set_eq_l151_151617

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def decreasing_condition (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < 0 → x2 < 0 → x1 ≠ x2 → (x1 * f (x1) - x2 * f (x2)) / (x1 - x2) < 0

variable (f : ℝ → ℝ)
variable (h_odd : odd_function f)
variable (h_minus_2_zero : f (-2) = 0)
variable (h_decreasing : decreasing_condition f)

theorem solution_set_eq :
  {x : ℝ | f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 2 < x} :=
sorry

end solution_set_eq_l151_151617


namespace addition_of_two_negatives_l151_151385

theorem addition_of_two_negatives (a b : ℤ) (ha : a < 0) (hb : b < 0) : a + b < a ∧ a + b < b :=
by
  sorry

end addition_of_two_negatives_l151_151385


namespace range_of_a_l151_151147

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x → x^2 + 2 * a * x + 1 ≥ 0) ↔ a ≥ -1 := 
by
  sorry

end range_of_a_l151_151147


namespace product_of_primes_sum_101_l151_151635

theorem product_of_primes_sum_101 (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_sum : p + q = 101) : p * q = 194 := by
  sorry

end product_of_primes_sum_101_l151_151635


namespace solve_for_x_l151_151549

theorem solve_for_x (x : ℝ) (h : 0.05 * x + 0.12 * (30 + x) = 15.6) : x = 1200 / 17 :=
by
  sorry

end solve_for_x_l151_151549


namespace system_of_linear_eq_with_two_variables_l151_151279

-- Definitions of individual equations
def eqA (x : ℝ) : Prop := 3 * x - 2 = 5
def eqB (x : ℝ) : Prop := 6 * x^2 - 2 = 0
def eqC (x y : ℝ) : Prop := 1 / x + y = 3
def eqD (x y : ℝ) : Prop := 5 * x + y = 2

-- The main theorem to prove that D is a system of linear equations with two variables
theorem system_of_linear_eq_with_two_variables :
    (∃ x y : ℝ, eqD x y) ∧ (¬∃ x : ℝ, eqA x) ∧ (¬∃ x : ℝ, eqB x) ∧ (¬∃ x y : ℝ, eqC x y) :=
by
  sorry

end system_of_linear_eq_with_two_variables_l151_151279


namespace impossible_digit_placement_l151_151124

-- Define the main variables and assumptions
variable (A B C : ℕ)
variable (h_sum : A + B = 45)
variable (h_segmentSum : 3 * A + B = 6 * C)

-- Define the impossible placement problem
theorem impossible_digit_placement :
  ¬(∃ A B C, A + B = 45 ∧ 3 * A + B = 6 * C ∧ 2 * A = 6 * C - 45) :=
by
  sorry

end impossible_digit_placement_l151_151124


namespace ratio_of_distance_l151_151593

noncomputable def initial_distance : ℝ := 30 * 20

noncomputable def total_distance : ℝ := 2 * initial_distance

noncomputable def distance_after_storm : ℝ := initial_distance - 200

theorem ratio_of_distance (initial_distance : ℝ) (total_distance : ℝ) (distance_after_storm : ℝ) : 
  distance_after_storm / total_distance = 1 / 3 :=
by
  -- Given conditions
  have h1 : initial_distance = 30 * 20 := by sorry
  have h2 : total_distance = 2 * initial_distance := by sorry
  have h3 : distance_after_storm = initial_distance - 200 := by sorry
  -- Prove the ratio is 1 / 3
  sorry

end ratio_of_distance_l151_151593


namespace sum_of_number_and_reverse_l151_151438

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) (h3 : 0 ≤ b) (h4 : b ≤ 9) 
  (h5 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) : 
  (10 * a + b) + (10 * b + a) = 99 := 
sorry

end sum_of_number_and_reverse_l151_151438


namespace ratio_of_girls_l151_151797

theorem ratio_of_girls (total_julian_friends : ℕ) (percent_julian_girls : ℚ)
  (percent_julian_boys : ℚ) (total_boyd_friends : ℕ) (percent_boyd_boys : ℚ) :
  total_julian_friends = 80 →
  percent_julian_girls = 0.40 →
  percent_julian_boys = 0.60 →
  total_boyd_friends = 100 →
  percent_boyd_boys = 0.36 →
  (0.64 * total_boyd_friends : ℚ) / (0.40 * total_julian_friends : ℚ) = 2 :=
by
  sorry

end ratio_of_girls_l151_151797


namespace perimeter_of_square_with_area_36_l151_151895

theorem perimeter_of_square_with_area_36 : 
  ∀ (A : ℝ), A = 36 → (∃ P : ℝ, P = 24 ∧ (∃ s : ℝ, s^2 = A ∧ P = 4 * s)) :=
by
  sorry

end perimeter_of_square_with_area_36_l151_151895


namespace sector_angle_l151_151267

theorem sector_angle (r l : ℝ) (h₁ : 2 * r + l = 4) (h₂ : 1/2 * l * r = 1) : l / r = 2 :=
by
  sorry

end sector_angle_l151_151267


namespace JessieScore_l151_151740

-- Define the conditions as hypotheses
variables (correct_answers : ℕ) (incorrect_answers : ℕ) (unanswered_questions : ℕ)
variables (points_per_correct : ℕ) (points_deducted_per_incorrect : ℤ)

-- Define the values for the specific problem instance
def JessieCondition := correct_answers = 16 ∧ incorrect_answers = 4 ∧ unanswered_questions = 10 ∧
                       points_per_correct = 2 ∧ points_deducted_per_incorrect = -1 / 2

-- Define the statement that Jessie's score is 30 given the conditions
theorem JessieScore (h : JessieCondition correct_answers incorrect_answers unanswered_questions points_per_correct points_deducted_per_incorrect) :
  (correct_answers * points_per_correct : ℤ) + (incorrect_answers * points_deducted_per_incorrect) = 30 :=
by
  sorry

end JessieScore_l151_151740


namespace mary_has_34_lambs_l151_151696

def mary_lambs (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) (traded_lambs : ℕ) (found_lambs : ℕ): ℕ :=
  initial_lambs + (lambs_with_babies * babies_per_lamb) - traded_lambs + found_lambs

theorem mary_has_34_lambs :
  mary_lambs 12 4 3 5 15 = 34 :=
by
  -- This line is in place of the actual proof.
  sorry

end mary_has_34_lambs_l151_151696


namespace negation_statement_l151_151850

variable {α : Type} 
variable (student prepared : α → Prop)

theorem negation_statement :
  (¬ ∀ x, student x → prepared x) ↔ (∃ x, student x ∧ ¬ prepared x) :=
by 
  -- proof will be provided here
  sorry

end negation_statement_l151_151850


namespace mass_percentage_ba_in_bao_l151_151314

-- Define the constants needed in the problem
def molarMassBa : ℝ := 137.33
def molarMassO : ℝ := 16.00

-- Calculate the molar mass of BaO
def molarMassBaO : ℝ := molarMassBa + molarMassO

-- Express the problem as a Lean theorem for proof
theorem mass_percentage_ba_in_bao : 
  (molarMassBa / molarMassBaO) * 100 = 89.55 := by
  sorry

end mass_percentage_ba_in_bao_l151_151314


namespace problem_1_problem_2_l151_151485

open Set -- to work with sets conveniently

noncomputable section -- to allow the use of real numbers and other non-constructive elements

-- Define U as the set of all real numbers
def U : Set ℝ := univ

-- Define M as the set of all x such that y = sqrt(x - 2)
def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 2) }

-- Define N as the set of all x such that x < 1 or x > 3
def N : Set ℝ := {x : ℝ | x < 1 ∨ x > 3}

-- Statement to prove (1)
theorem problem_1 : M ∪ N = {x : ℝ | x < 1 ∨ x ≥ 2} := sorry

-- Statement to prove (2)
theorem problem_2 : M ∩ (compl N) = {x : ℝ | 2 ≤ x ∧ x ≤ 3} := sorry

end problem_1_problem_2_l151_151485


namespace problem_I2_1_problem_I2_2_problem_I2_3_problem_I2_4_l151_151435

-- Problem I2.1
theorem problem_I2_1 (a : ℕ) (h₁ : a > 0) (h₂ : a^2 - 1 = 123 * 125) : a = 124 :=
by {
  -- This proof needs to be filled in
  sorry
}

-- Problem I2.2
theorem problem_I2_2 (b : ℕ) (h₁ : b = (2^3 - 16*2^2 - 9*2 + 124)) : b = 50 :=
by {
  -- This proof needs to be filled in
  sorry
}

-- Problem I2.3
theorem problem_I2_3 (n : ℕ) (h₁ : (n * (n - 3)) / 2 = 54) : n = 12 :=
by {
  -- This proof needs to be filled in
  sorry
}

-- Problem I2_4
theorem problem_I2_4 (d : ℤ) (n : ℤ) (h₁ : n = 12) 
  (h₂ : (d - 1) * 2 = (1 - n) * 2) : d = -10 :=
by {
  -- This proof needs to be filled in
  sorry
}

end problem_I2_1_problem_I2_2_problem_I2_3_problem_I2_4_l151_151435


namespace arithmetic_sequence_count_l151_151446

theorem arithmetic_sequence_count :
  ∃ n : ℕ, 2 + (n-1) * 5 = 2507 ∧ n = 502 :=
by
  sorry

end arithmetic_sequence_count_l151_151446


namespace sum_of_roots_l151_151111

theorem sum_of_roots (m n : ℝ) (h₁ : m ≠ 0) (h₂ : n ≠ 0) (h₃ : ∀ x : ℝ, x^2 + m * x + n = 0 → (x = m ∨ x = n)) :
  m + n = -1 :=
sorry

end sum_of_roots_l151_151111


namespace multiple_of_1984_exists_l151_151732

theorem multiple_of_1984_exists (a : Fin 97 → ℕ) (h_distinct: Function.Injective a) :
  ∃ i j k l : Fin 97, i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧ 
  1984 ∣ (a i - a j) * (a k - a l) :=
by
  sorry

end multiple_of_1984_exists_l151_151732


namespace unloading_time_relationship_l151_151764

-- Conditions
def loading_speed : ℝ := 30
def loading_time : ℝ := 8
def total_tonnage : ℝ := loading_speed * loading_time
def unloading_speed (x : ℝ) : ℝ := x

-- Proof statement
theorem unloading_time_relationship (x : ℝ) (hx : x ≠ 0) : 
  ∀ y : ℝ, y = 240 / x :=
by 
  sorry

end unloading_time_relationship_l151_151764


namespace even_integers_count_l151_151938

theorem even_integers_count (n : ℤ) (m : ℤ) (total_even : ℤ) 
  (h1 : m = 45) (h2 : total_even = 10) (h3 : m % 2 = 1) :
  (∃ k : ℤ, ∀ x : ℤ, 0 ≤ x ∧ x < total_even → k = n + 2 * x) ∧ (n = 26) :=
by
  sorry

end even_integers_count_l151_151938


namespace arithmetic_sequence_l151_151308

theorem arithmetic_sequence {a b : ℤ} :
  (-1 < a ∧ a < b ∧ b < 8) ∧
  (8 - (-1) = 9) ∧
  (a + b = 7) →
  (a = 2 ∧ b = 5) :=
by
  sorry

end arithmetic_sequence_l151_151308


namespace value_of_k_l151_151351

theorem value_of_k (k : ℝ) :
  ∃ (k : ℝ), k ≠ 1 ∧ (k-1) * (0 : ℝ)^2 + 6 * (0 : ℝ) + k^2 - 1 = 0 ∧ k = -1 :=
by
  sorry

end value_of_k_l151_151351


namespace slope_at_A_is_7_l151_151430

def curve (x : ℝ) : ℝ := x^2 + 3 * x

def point_A : ℝ × ℝ := (2, 10)

theorem slope_at_A_is_7 : (deriv curve 2) = 7 := 
by
  sorry

end slope_at_A_is_7_l151_151430


namespace solution_set_l151_151458

noncomputable def f : ℝ → ℝ
| x => if x < 2 then 2 * Real.exp (x - 1) else Real.log (x^2 - 1) / Real.log 3

theorem solution_set (x : ℝ) : 
  ((x > 1 ∧ x < 2 ∨ x > Real.sqrt 10)) ↔ f x > 2 :=
sorry

end solution_set_l151_151458


namespace austin_more_apples_than_dallas_l151_151774

-- Conditions as definitions
def dallas_apples : ℕ := 14
def dallas_pears : ℕ := 9
def austin_pears : ℕ := dallas_pears - 5
def austin_total_fruit : ℕ := 24

-- The theorem statement
theorem austin_more_apples_than_dallas 
  (austin_apples : ℕ) (h1 : austin_apples + austin_pears = austin_total_fruit) :
  austin_apples - dallas_apples = 6 :=
sorry

end austin_more_apples_than_dallas_l151_151774


namespace remainder_of_n_l151_151052

theorem remainder_of_n {n : ℕ} (h1 : n^2 ≡ 4 [MOD 7]) (h2 : n^3 ≡ 6 [MOD 7]): 
  n ≡ 5 [MOD 7] :=
sorry

end remainder_of_n_l151_151052


namespace gamma_max_success_ratio_l151_151814

theorem gamma_max_success_ratio :
  ∀ (x y z w : ℕ),
    x > 0 → z > 0 →
    (5 * x < 3 * y) →
    (5 * z < 3 * w) →
    (y + w = 600) →
    (x + z ≤ 359) :=
by
  intros x y z w hx hz hxy hzw hyw
  sorry

end gamma_max_success_ratio_l151_151814


namespace sum_difference_of_consecutive_integers_l151_151047

theorem sum_difference_of_consecutive_integers (n : ℤ) :
  let set1 := [(n-3), (n-2), (n-1), n, (n+1), (n+2), (n+3)]
  let set2 := [(n+1), (n+2), (n+3), (n+4), (n+5), (n+6), (n+7)]
  let S1 := set1.sum
  let S2 := set2.sum
  S2 - S1 = 28 :=
by
  let set1 := [(n-3), (n-2), (n-1), n, (n+1), (n+2), (n+3)]
  let set2 := [(n+1), (n+2), (n+3), (n+4), (n+5), (n+6), (n+7)]
  let S1 := set1.sum
  let S2 := set2.sum
  have hS1 : S1 = (n-3) + (n-2) + (n-1) + n + (n+1) + (n+2) + (n+3) := by sorry
  have hS2 : S2 = (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) := by sorry
  have h_diff : S2 - S1 = 28 := by sorry
  exact h_diff

end sum_difference_of_consecutive_integers_l151_151047


namespace seafood_regular_price_l151_151434

theorem seafood_regular_price (y : ℝ) (h : y / 4 = 4) : 2 * y = 32 := by
  sorry

end seafood_regular_price_l151_151434


namespace union_complement_correctness_l151_151793

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem union_complement_correctness : 
  U = {1, 2, 3, 4, 5} →
  A = {1, 2, 3} →
  B = {2, 4} →
  A ∪ (U \ B) = {1, 2, 3, 5} :=
by
  intro hU hA hB
  sorry

end union_complement_correctness_l151_151793


namespace average_of_solutions_l151_151486

-- Define the quadratic equation condition
def quadratic_eq : Prop := ∃ x : ℂ, 3*x^2 - 4*x + 1 = 0

-- State the theorem
theorem average_of_solutions : quadratic_eq → (∃ avg : ℂ, avg = 2 / 3) :=
by
  sorry

end average_of_solutions_l151_151486


namespace min_value_x_squared_plus_6x_l151_151355

theorem min_value_x_squared_plus_6x : ∀ x : ℝ, ∃ y : ℝ, y ≤ x^2 + 6*x ∧ y = -9 := 
sorry

end min_value_x_squared_plus_6x_l151_151355


namespace fill_in_blanks_problem1_fill_in_blanks_problem2_fill_in_blanks_problem3_l151_151099

def problem1_seq : List ℕ := [102, 101, 100, 99, 98, 97, 96]
def problem2_seq : List ℕ := [190, 180, 170, 160, 150, 140, 130, 120, 110, 100]
def problem3_seq : List ℕ := [5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500]

theorem fill_in_blanks_problem1 :
  ∃ (a b c d : ℕ), [102, a, 100, b, c, 97, d] = [102, 101, 100, 99, 98, 97, 96] :=
by
  exact ⟨101, 99, 98, 96, rfl⟩ -- Proof omitted with exact values

theorem fill_in_blanks_problem2 :
  ∃ (a b c d e f g : ℕ), [190, a, b, 160, c, d, e, 120, f, g] = [190, 180, 170, 160, 150, 140, 130, 120, 110, 100] :=
by
  exact ⟨180, 170, 150, 140, 130, 110, 100, rfl⟩ -- Proof omitted with exact values

theorem fill_in_blanks_problem3 :
  ∃ (a b c d e f : ℕ), [5000, a, 6000, b, 7000, c, d, e, f, 9500] = [5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500] :=
by
  exact ⟨5500, 6500, 7500, 8000, 8500, 9000, rfl⟩ -- Proof omitted with exact values

end fill_in_blanks_problem1_fill_in_blanks_problem2_fill_in_blanks_problem3_l151_151099


namespace correct_calculation_l151_151697

theorem correct_calculation :
  (∀ a : ℝ, (a^2)^3 = a^6) ∧
  ¬(∀ a : ℝ, a * a^3 = a^3) ∧
  ¬(∀ a : ℝ, a + 2 * a^2 = 3 * a^3) ∧
  ¬(∀ (a b : ℝ), (-2 * a^2 * b)^2 = -4 * a^4 * b^2) :=
by
  sorry

end correct_calculation_l151_151697


namespace projection_of_vec_c_onto_vec_b_l151_151259

def vec (x y : ℝ) : Prod ℝ ℝ := (x, y)

noncomputable def projection_of_c_onto_b := 
  let a := vec 2 3
  let b := vec (-4) 7
  let c := vec (-2) (-3)
  let dot_product_c_b := (-2) * (-4) + (-3) * 7
  let magnitude_b := Real.sqrt ((-4)^2 + 7^2)
  dot_product_c_b / magnitude_b
  
theorem projection_of_vec_c_onto_vec_b : 
  let a := vec 2 3
  let b := vec (-4) 7
  let c := vec (-2) (-3)
  let projection := projection_of_c_onto_b
  a + c = vec 0 0 ->
  projection = - Real.sqrt 65 / 5 := by
    sorry

end projection_of_vec_c_onto_vec_b_l151_151259


namespace time_to_cross_tree_l151_151197

variable (length_train : ℕ) (time_platform : ℕ) (length_platform : ℕ)

theorem time_to_cross_tree (h1 : length_train = 1200) (h2 : time_platform = 190) (h3 : length_platform = 700) :
  let distance_platform := length_train + length_platform
  let speed_train := distance_platform / time_platform
  let time_to_cross_tree := length_train / speed_train
  time_to_cross_tree = 120 :=
by
  -- Using the conditions to prove the goal
  sorry

end time_to_cross_tree_l151_151197


namespace classroom_students_count_l151_151557

-- Definitions from the conditions
def students (C S Sh : ℕ) : Prop :=
  S = 2 * C ∧
  S = Sh + 8 ∧
  Sh = C + 19

-- Proof statement
theorem classroom_students_count (C S Sh : ℕ) 
  (h : students C S Sh) : 3 * C = 81 :=
by
  sorry

end classroom_students_count_l151_151557


namespace fewest_apples_l151_151763

-- Definitions based on the conditions
def Yoongi_apples : Nat := 4
def Jungkook_initial_apples : Nat := 6
def Jungkook_additional_apples : Nat := 3
def Jungkook_apples : Nat := Jungkook_initial_apples + Jungkook_additional_apples
def Yuna_apples : Nat := 5

-- Main theorem based on the question and the correct answer
theorem fewest_apples : Yoongi_apples < Jungkook_apples ∧ Yoongi_apples < Yuna_apples :=
by
  sorry

end fewest_apples_l151_151763


namespace sum_of_ages_is_22_l151_151498

noncomputable def Ashley_Age := 8
def Mary_Age (M : ℕ) := 7 * Ashley_Age = 4 * M

theorem sum_of_ages_is_22 (M : ℕ) (h : Mary_Age M):
  Ashley_Age + M = 22 :=
by
  -- skipping proof details
  sorry

end sum_of_ages_is_22_l151_151498


namespace largest_n_unique_k_l151_151490

theorem largest_n_unique_k (n : ℕ) (h : ∃ k : ℕ, (9 / 17 : ℚ) < n / (n + k) ∧ n / (n + k) < (8 / 15 : ℚ) ∧ ∀ k' : ℕ, ((9 / 17 : ℚ) < n / (n + k') ∧ n / (n + k') < (8 / 15 : ℚ)) → k' = k) : n = 72 :=
sorry

end largest_n_unique_k_l151_151490


namespace annie_overtakes_bonnie_l151_151492

-- Define the conditions
def track_circumference : ℝ := 300
def bonnie_speed (v : ℝ) : ℝ := v
def annie_speed (v : ℝ) : ℝ := 1.5 * v

-- Define the statement for proving the number of laps completed by Annie when she first overtakes Bonnie
theorem annie_overtakes_bonnie (v t : ℝ) : 
  bonnie_speed v * t = track_circumference * 2 → 
  annie_speed v * t = track_circumference * 3 :=
by
  sorry

end annie_overtakes_bonnie_l151_151492


namespace A_takes_4_hours_l151_151037

variables (A B C : ℝ)

-- Given conditions
axiom h1 : 1 / B + 1 / C = 1 / 2
axiom h2 : 1 / A + 1 / C = 1 / 2
axiom h3 : B = 4

-- What we need to prove: A = 4
theorem A_takes_4_hours :
  A = 4 := by
  sorry

end A_takes_4_hours_l151_151037


namespace maximum_M_k_l151_151249

-- Define the problem
def J (k : ℕ) : ℕ := 10^(k + 2) + 128

-- Define M(k) as the number of factors of 2 in the prime factorization of J(k)
def M (k : ℕ) : ℕ :=
  -- implementation details omitted
  sorry

-- The core theorem to prove
theorem maximum_M_k : ∃ k > 0, M k = 8 :=
by sorry

end maximum_M_k_l151_151249


namespace find_projection_l151_151461

noncomputable def a : ℝ × ℝ := (-3, 2)
noncomputable def b : ℝ × ℝ := (5, -1)
noncomputable def p : ℝ × ℝ := (21/73, 56/73)
noncomputable def d : ℝ × ℝ := (8, -3)

theorem find_projection :
  ∃ t : ℝ, (t * d.1 - a.1, t * d.2 + a.2) = p ∧
          (p.1 - a.1) * d.1 + (p.2 - a.2) * d.2 = 0 :=
by
  sorry

end find_projection_l151_151461


namespace delta_max_success_ratio_l151_151594

theorem delta_max_success_ratio :
  ∃ a b c d : ℕ, 
    0 < a ∧ a < b ∧ (40 * a) < (21 * b) ∧
    0 < c ∧ c < d ∧ (4 * c) < (3 * d) ∧
    b + d = 600 ∧
    (a + c) / 600 = 349 / 600 :=
by
  sorry

end delta_max_success_ratio_l151_151594


namespace evaluate_expression_l151_151910

theorem evaluate_expression : (1 - (1 / 4)) / (1 - (1 / 5)) = 15 / 16 :=
by 
  sorry

end evaluate_expression_l151_151910


namespace sum_areas_of_eight_disks_l151_151940

noncomputable def eight_disks_sum_areas (C_radius disk_count : ℝ) 
  (cover_C : ℝ) (no_overlap : ℝ) (tangent_neighbors : ℝ) : ℕ :=
  let r := (2 - Real.sqrt 2)
  let area_one_disk := Real.pi * r^2
  let total_area := disk_count * area_one_disk
  let a := 48
  let b := 32
  let c := 2
  a + b + c

theorem sum_areas_of_eight_disks : eight_disks_sum_areas 1 8 1 1 1 = 82 :=
  by
  -- sorry is used to skip the proof
  sorry

end sum_areas_of_eight_disks_l151_151940


namespace construct_inaccessible_angle_bisector_l151_151366

-- Definitions for problem context
structure Point :=
  (x y : ℝ)

structure Line :=
  (p1 p2 : Point)

structure Angle := 
  (vertex : Point)
  (ray1 ray2 : Line)

-- Predicate to determine if a line bisects an angle
def IsAngleBisector (L : Line) (A : Angle) : Prop := sorry

-- The inaccessible vertex angle we are considering
-- Let's assume the vertex is defined but we cannot access it physically in constructions
noncomputable def inaccessible_angle : Angle := sorry

-- Statement to prove: Construct a line that bisects the inaccessible angle
theorem construct_inaccessible_angle_bisector :
  ∃ L : Line, IsAngleBisector L inaccessible_angle :=
sorry

end construct_inaccessible_angle_bisector_l151_151366


namespace find_C_l151_151570

theorem find_C (C : ℤ) (h : 4 * C + 3 = 31) : C = 7 := by
  sorry

end find_C_l151_151570


namespace product_of_roots_l151_151405

theorem product_of_roots (r1 r2 r3 : ℝ) : 
  (∀ x : ℝ, 2 * x^3 - 24 * x^2 + 96 * x + 56 = 0 → x = r1 ∨ x = r2 ∨ x = r3) →
  r1 * r2 * r3 = -28 :=
by
  sorry

end product_of_roots_l151_151405


namespace convert_cylindrical_to_rectangular_l151_151859

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem convert_cylindrical_to_rectangular :
  cylindrical_to_rectangular 7 (Real.pi / 4) 8 = (7 * Real.sqrt 2 / 2, 7 * Real.sqrt 2 / 2, 8) :=
by
  sorry

end convert_cylindrical_to_rectangular_l151_151859


namespace quadrilateral_area_is_114_5_l151_151723

noncomputable def area_of_quadrilateral_114_5 
  (AB BC CD AD : ℝ) (angle_ABC : ℝ)
  (h1 : AB = 5) (h2 : BC = 12) (h3 : CD = 13) (h4 : AD = 13) (h5 : angle_ABC = 90) : ℝ :=
  114.5

theorem quadrilateral_area_is_114_5
  (AB BC CD AD : ℝ) (angle_ABC : ℝ)
  (h1 : AB = 5) (h2 : BC = 12) (h3 : CD = 13) (h4 : AD = 13) (h5 : angle_ABC = 90) :
  area_of_quadrilateral_114_5 AB BC CD AD angle_ABC h1 h2 h3 h4 h5 = 114.5 :=
sorry

end quadrilateral_area_is_114_5_l151_151723


namespace point_in_quadrant_l151_151636

theorem point_in_quadrant (a b : ℝ) (h1 : a - b > 0) (h2 : a * b < 0) : 
  (a > 0 ∧ b < 0) ∧ ¬(a > 0 ∧ b > 0) ∧ ¬(a < 0 ∧ b > 0) ∧ ¬(a < 0 ∧ b < 0) := 
by 
  sorry

end point_in_quadrant_l151_151636


namespace part1_part2_l151_151518

noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then -3 * x + (1/2)^x - 1 else sorry -- Placeholder: function definition incomplete for x ≤ 0

def odd (f : ℝ → ℝ) :=
∀ x, f (-x) = - f x

def monotonic_decreasing (f : ℝ → ℝ) :=
∀ x y, x < y → f x > f y

axiom f_conditions :
  monotonic_decreasing f ∧
  odd f ∧
  (∀ x, x > 0 → f x = -3 * x + (1/2)^x - 1)

theorem part1 : f (-1) = 3.5 :=
by
  sorry

theorem part2 (t : ℝ) (k : ℝ) :
  (∀ t, f (t^2 - 2 * t) + f (2 * t^2 - k) < 0) ↔ k < -1/3 :=
by
  sorry

end part1_part2_l151_151518


namespace a_three_equals_35_l151_151620

-- Define the mathematical sequences and functions
def S (n : ℕ) : ℕ := 5 * n^2 + 10 * n

def a (n : ℕ) : ℕ := S (n + 1) - S n

-- The proposition we want to prove
theorem a_three_equals_35 : a 2 = 35 := by 
  sorry

end a_three_equals_35_l151_151620


namespace rationalize_denominator_l151_151934

theorem rationalize_denominator :
  (3 : ℝ) / Real.sqrt 48 = Real.sqrt 3 / 4 :=
by
  sorry

end rationalize_denominator_l151_151934


namespace min_value_fraction_l151_151961

theorem min_value_fraction (m n : ℝ) (h₀ : m > 0) (h₁ : n > 0) (h₂ : m + 2 * n = 1) : 
  (1 / m + 1 / n) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end min_value_fraction_l151_151961


namespace circle_radius_l151_151298

theorem circle_radius (x y : ℝ) : x^2 - 10 * x + y^2 + 4 * y + 13 = 0 → (x - 5)^2 + (y + 2)^2 = 4^2 :=
by
  sorry

end circle_radius_l151_151298


namespace find_x_squared_plus_y_squared_l151_151233

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 6) : x^2 + y^2 = 13 :=
by
  sorry

end find_x_squared_plus_y_squared_l151_151233


namespace extreme_values_l151_151000

def f (x : ℝ) : ℝ := x^3 - 4 * x^2 + 5 * x - 4

theorem extreme_values :
  (∃ (x1 x2 : ℝ), x1 = 1 ∧ x2 = 5 / 3 ∧ f x1 = -2 ∧ f x2 = -58 / 27) ∧ 
  (∃ (a b : ℝ), a = 2 ∧ b = f 2 ∧ (∀ (x : ℝ), (a, b) = (x, f x) → (∀ y : ℝ, y = x - 4))) :=
by
  sorry

end extreme_values_l151_151000


namespace root_product_identity_l151_151532

theorem root_product_identity (a b c : ℝ) (h1 : a * b * c = -8) (h2 : a * b + b * c + c * a = 20) (h3 : a + b + c = 15) :
    (1 + a) * (1 + b) * (1 + c) = 28 :=
by
  sorry

end root_product_identity_l151_151532


namespace fraction_of_married_men_l151_151140

theorem fraction_of_married_men (total_women married_women : ℕ) 
    (h1 : total_women = 7)
    (h2 : married_women = 4)
    (single_women_probability : ℚ)
    (h3 : single_women_probability = 3 / 7) : 
    (4 / 11 : ℚ) = (married_women / (total_women + married_women)) := 
sorry

end fraction_of_married_men_l151_151140


namespace cost_of_weed_eater_string_l151_151517

-- Definitions
def num_blades := 4
def cost_per_blade := 8
def total_spent := 39
def total_cost_of_blades := num_blades * cost_per_blade
def cost_of_string := total_spent - total_cost_of_blades

-- The theorem statement
theorem cost_of_weed_eater_string : cost_of_string = 7 :=
by {
  -- The proof would go here
  sorry
}

end cost_of_weed_eater_string_l151_151517


namespace point_on_line_and_in_first_quadrant_l151_151824

theorem point_on_line_and_in_first_quadrant (x y : ℝ) (hline : y = -2 * x + 3) (hfirst_quadrant : x > 0 ∧ y > 0) :
    (x, y) = (1, 1) :=
by
  sorry

end point_on_line_and_in_first_quadrant_l151_151824


namespace scientific_notation_of_population_l151_151107

theorem scientific_notation_of_population (population : Real) (h_pop : population = 6.8e6) :
    ∃ a n, (1 ≤ |a| ∧ |a| < 10) ∧ (population = a * 10^n) ∧ (a = 6.8) ∧ (n = 6) :=
by
  sorry

end scientific_notation_of_population_l151_151107


namespace train_crossing_time_l151_151962

theorem train_crossing_time
  (length_train : ℕ)
  (speed_train_kmph : ℕ)
  (total_length : ℕ)
  (htotal_length : total_length = 225)
  (hlength_train : length_train = 150)
  (hspeed_train_kmph : speed_train_kmph = 45) : 
  (total_length / (speed_train_kmph * 1000 / 3600)) = 18 := by 
  sorry

end train_crossing_time_l151_151962


namespace selection_problem_l151_151232

def group_size : ℕ := 10
def selected_group_size : ℕ := 3
def total_ways_without_C := Nat.choose 9 3
def ways_without_A_B_C := Nat.choose 7 3
def correct_answer := total_ways_without_C - ways_without_A_B_C

theorem selection_problem:
  (∃ (A B C : ℕ), total_ways_without_C - ways_without_A_B_C = 49) :=
by
  sorry

end selection_problem_l151_151232


namespace part1_case1_part1_case2_part1_case3_part2_l151_151325

def f (m x : ℝ) : ℝ := (m+1)*x^2 - (m-1)*x + (m-1)

theorem part1_case1 (m x : ℝ) (h : m = -1) : 
  f m x ≥ (m+1)*x → x ≥ 1 := sorry

theorem part1_case2 (m x : ℝ) (h : m > -1) :
  f m x ≥ (m+1)*x →
  (x ≤ (m-1)/(m+1) ∨ x ≥ 1) := sorry

theorem part1_case3 (m x : ℝ) (h : m < -1) : 
  f m x ≥ (m+1)*x →
  (1 ≤ x ∧ x ≤ (m-1)/(m+1)) := sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f m x ≥ 0) →
  m ≥ 1 := sorry

end part1_case1_part1_case2_part1_case3_part2_l151_151325


namespace Antoinette_weight_l151_151761

-- Define weights for Antoinette and Rupert
variables (A R : ℕ)

-- Define the given conditions
def condition1 := A = 2 * R - 7
def condition2 := A + R = 98

-- The theorem to prove under the given conditions
theorem Antoinette_weight : condition1 A R → condition2 A R → A = 63 := 
by {
  -- The proof is omitted
  sorry
}

end Antoinette_weight_l151_151761


namespace sqrt_cosine_identity_l151_151413

theorem sqrt_cosine_identity :
  Real.sqrt ((3 - Real.cos (Real.pi / 8)^2) * (3 - Real.cos (3 * Real.pi / 8)^2)) = (3 * Real.sqrt 5) / 4 :=
by
  sorry

end sqrt_cosine_identity_l151_151413


namespace num_non_divisible_by_3_divisors_l151_151637

theorem num_non_divisible_by_3_divisors (a b c : ℕ) (h1: 0 ≤ a ∧ a ≤ 2) (h2: 0 ≤ b ∧ b ≤ 2) (h3: 0 ≤ c ∧ c ≤ 1) :
  (3 * 2 = 6) :=
by sorry

end num_non_divisible_by_3_divisors_l151_151637


namespace sum_of_arithmetic_sequence_l151_151548

variable {α : Type*} [LinearOrderedField α]

noncomputable def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
∀ n, a (n + 1) - a n = d

noncomputable def sum_of_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
n * (a 1 + a n) / 2

theorem sum_of_arithmetic_sequence {a : ℕ → α} {d : α}
  (h3 : a 3 * a 7 = -16)
  (h4 : a 4 + a 6 = 0)
  (ha : is_arithmetic_sequence a d) :
  ∃ (s : α), s = n * (n - 9) ∨ s = -n * (n - 9) :=
sorry

end sum_of_arithmetic_sequence_l151_151548


namespace greatest_x_l151_151400

-- Define x as a positive multiple of 4.
def is_positive_multiple_of_four (x : ℕ) : Prop :=
  x > 0 ∧ ∃ k : ℕ, x = 4 * k

-- Statement of the equivalent proof problem
theorem greatest_x (x : ℕ) (h1: is_positive_multiple_of_four x) (h2: x^3 < 4096) : x ≤ 12 :=
by {
  sorry
}

end greatest_x_l151_151400


namespace total_toys_l151_151753

theorem total_toys (toys_kamari : ℕ) (toys_anais : ℕ) (h1 : toys_kamari = 65) (h2 : toys_anais = toys_kamari + 30) :
  toys_kamari + toys_anais = 160 :=
by 
  sorry

end total_toys_l151_151753


namespace value_of_a_l151_151993

def star (a b : ℤ) : ℤ := 3 * a - b^3

theorem value_of_a : ∃ a : ℤ, star a 3 = 63 ∧ a = 30 := by
  sorry

end value_of_a_l151_151993


namespace committee_selection_count_l151_151493

-- Definition of the problem condition: Club of 12 people, one specific person must always be on the committee.
def club_size : ℕ := 12
def committee_size : ℕ := 4
def specific_person_included : ℕ := 1

-- Number of ways to choose 3 members from the other 11 people
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem committee_selection_count : choose 11 3 = 165 := 
  sorry

end committee_selection_count_l151_151493


namespace most_likely_number_of_red_balls_l151_151289

-- Define the conditions
def total_balls : ℕ := 20
def red_ball_frequency : ℝ := 0.8

-- Define the statement we want to prove
theorem most_likely_number_of_red_balls : red_ball_frequency * total_balls = 16 :=
by sorry

end most_likely_number_of_red_balls_l151_151289


namespace find_siblings_l151_151199

-- Define the characteristics of each child
structure Child where
  name : String
  eyeColor : String
  hairColor : String
  age : Nat

-- List of children
def Olivia : Child := { name := "Olivia", eyeColor := "Green", hairColor := "Red", age := 12 }
def Henry  : Child := { name := "Henry", eyeColor := "Gray", hairColor := "Brown", age := 12 }
def Lucas  : Child := { name := "Lucas", eyeColor := "Green", hairColor := "Red", age := 10 }
def Emma   : Child := { name := "Emma", eyeColor := "Green", hairColor := "Brown", age := 12 }
def Mia    : Child := { name := "Mia", eyeColor := "Gray", hairColor := "Red", age := 10 }
def Noah   : Child := { name := "Noah", eyeColor := "Gray", hairColor := "Brown", age := 12 }

-- Define a family as a set of children who share at least one characteristic
def isFamily (c1 c2 c3 : Child) : Prop :=
  (c1.eyeColor = c2.eyeColor ∨ c1.eyeColor = c3.eyeColor ∨ c2.eyeColor = c3.eyeColor) ∨
  (c1.hairColor = c2.hairColor ∨ c1.hairColor = c3.hairColor ∨ c2.hairColor = c3.hairColor) ∨
  (c1.age = c2.age ∨ c1.age = c3.age ∨ c2.age = c3.age)

-- The main theorem
theorem find_siblings : isFamily Olivia Lucas Emma :=
by
  sorry

end find_siblings_l151_151199


namespace rate_mangoes_correct_l151_151587

-- Define the conditions
def weight_apples : ℕ := 8
def rate_apples : ℕ := 70
def cost_apples := weight_apples * rate_apples

def total_payment : ℕ := 1145
def weight_mangoes : ℕ := 9
def cost_mangoes := total_payment - cost_apples

-- Define the rate per kg of mangoes
def rate_mangoes := cost_mangoes / weight_mangoes

-- Prove the rate per kg for mangoes
theorem rate_mangoes_correct : rate_mangoes = 65 := by
  -- all conditions and intermediate calculations already stated
  sorry

end rate_mangoes_correct_l151_151587


namespace A_wins_match_prob_correct_l151_151926

def probA_wins_game : ℝ := 0.6
def probB_wins_game : ℝ := 0.4

def probA_wins_match : ℝ :=
  let probA_wins_first_two := probA_wins_game * probA_wins_game
  let probA_wins_first_and_third := probA_wins_game * probB_wins_game * probA_wins_game
  let probA_wins_last_two := probB_wins_game * probA_wins_game * probA_wins_game
  probA_wins_first_two + probA_wins_first_and_third + probA_wins_last_two

theorem A_wins_match_prob_correct : probA_wins_match = 0.648 := by
  sorry

end A_wins_match_prob_correct_l151_151926


namespace sum_sequence_S_n_l151_151978

variable {S : ℕ+ → ℚ}
noncomputable def S₁ : ℚ := 1 / 2
noncomputable def S₂ : ℚ := 5 / 6
noncomputable def S₃ : ℚ := 49 / 72
noncomputable def S₄ : ℚ := 205 / 288

theorem sum_sequence_S_n (n : ℕ+) :
  (S 1 = S₁) ∧ (S 2 = S₂) ∧ (S 3 = S₃) ∧ (S 4 = S₄) ∧ (∀ n : ℕ+, S n = n / (n + 1)) :=
by
  sorry

end sum_sequence_S_n_l151_151978


namespace sum_of_geometric_ratios_l151_151183

theorem sum_of_geometric_ratios (k p r : ℝ) (a2 a3 b2 b3 : ℝ)
  (hk : k ≠ 0) (hp : p ≠ r)
  (ha2 : a2 = k * p) (ha3 : a3 = k * p * p)
  (hb2 : b2 = k * r) (hb3 : b3 = k * r * r)
  (h : a3 - b3 = 3 * (a2 - b2)) :
  p + r = 3 :=
by sorry

end sum_of_geometric_ratios_l151_151183


namespace distance_to_SFL_is_81_l151_151711

variable (Speed : ℝ)
variable (Time : ℝ)

def distance_to_SFL (Speed : ℝ) (Time : ℝ) := Speed * Time

theorem distance_to_SFL_is_81 : distance_to_SFL 27 3 = 81 :=
by
  sorry

end distance_to_SFL_is_81_l151_151711


namespace cubic_roots_sum_of_cubes_l151_151210

theorem cubic_roots_sum_of_cubes (r s t a b c : ℚ) 
  (h1 : r + s + t = a) 
  (h2 : r * s + r * t + s * t = b)
  (h3 : r * s * t = c) 
  (h_poly : ∀ x : ℚ, x^3 - a*x^2 + b*x - c = 0 ↔ (x = r ∨ x = s ∨ x = t)) :
  r^3 + s^3 + t^3 = a^3 - 3 * a * b + 3 * c :=
sorry

end cubic_roots_sum_of_cubes_l151_151210


namespace Sara_has_3194_quarters_in_the_end_l151_151117

theorem Sara_has_3194_quarters_in_the_end
  (initial_quarters : ℕ)
  (borrowed_quarters : ℕ)
  (initial_quarters_eq : initial_quarters = 4937)
  (borrowed_quarters_eq : borrowed_quarters = 1743) :
  initial_quarters - borrowed_quarters = 3194 := by
  sorry

end Sara_has_3194_quarters_in_the_end_l151_151117


namespace pythagorean_triple_square_l151_151442

theorem pythagorean_triple_square (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pythagorean : a^2 + b^2 = c^2) : ∃ k : ℤ, k^2 = (c - a) * (c - b) / 2 := 
sorry

end pythagorean_triple_square_l151_151442


namespace find_y_l151_151157

theorem find_y (x : ℝ) (h1 : x = 1.3333333333333333) (h2 : (x * y) / 3 = x^2) : y = 4 :=
by 
  sorry

end find_y_l151_151157


namespace average_shifted_samples_l151_151539

variables (x1 x2 x3 x4 : ℝ)

theorem average_shifted_samples (h : (x1 + x2 + x3 + x4) / 4 = 2) :
  ((x1 + 3) + (x2 + 3) + (x3 + 3) + (x4 + 3)) / 4 = 5 :=
by
  sorry

end average_shifted_samples_l151_151539


namespace angle_in_third_quadrant_l151_151177

theorem angle_in_third_quadrant (θ : ℝ) (hθ : θ = 2014) : 180 < θ % 360 ∧ θ % 360 < 270 :=
by
  sorry

end angle_in_third_quadrant_l151_151177


namespace coordinates_of_N_l151_151626

-- Define the given conditions
def M : ℝ × ℝ := (5, -6)
def a : ℝ × ℝ := (1, -2)
def minusThreeA : ℝ × ℝ := (-3, 6)
def vectorMN (N : ℝ × ℝ) : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)

-- Define the required goal
theorem coordinates_of_N (N : ℝ × ℝ) : vectorMN N = minusThreeA → N = (2, 0) :=
by
  sorry

end coordinates_of_N_l151_151626


namespace find_x_for_set_6_l151_151782

theorem find_x_for_set_6 (x : ℝ) (h : 6 ∈ ({2, 4, x^2 - x} : Set ℝ)) : x = 3 ∨ x = -2 := 
by 
  sorry

end find_x_for_set_6_l151_151782


namespace flowerbed_width_l151_151238

theorem flowerbed_width (w : ℝ) (h₁ : 22 = 2 * (2 * w - 1) + 2 * w) : w = 4 :=
sorry

end flowerbed_width_l151_151238


namespace age_impossibility_l151_151350

/-
Problem statement:
Ann is 5 years older than Kristine.
Their current ages sum up to 24.
Prove that it's impossible for both their ages to be whole numbers.
-/

theorem age_impossibility 
  (K A : ℕ) -- Kristine's and Ann's ages are natural numbers
  (h1 : A = K + 5) -- Ann is 5 years older than Kristine
  (h2 : K + A = 24) -- their combined age is 24
  : false := sorry

end age_impossibility_l151_151350


namespace largest_valid_four_digit_number_l151_151900

-- Definition of the problem conditions
def is_valid_number (a b c d : ℕ) : Prop :=
  c = a + b ∧ d = b + c

-- Proposition that we need to prove
theorem largest_valid_four_digit_number : ∃ (a b c d : ℕ),
  is_valid_number a b c d ∧ a * 1000 + b * 100 + c * 10 + d = 9099 :=
by
  sorry

end largest_valid_four_digit_number_l151_151900


namespace career_preference_angles_l151_151820

theorem career_preference_angles (m f : ℕ) (total_degrees : ℕ) (one_fourth_males one_half_females : ℚ) (male_ratio female_ratio : ℚ) :
  total_degrees = 360 → male_ratio = 2/3 → female_ratio = 3/3 →
  m = 2 * f / 3 → one_fourth_males = 1/4 * m → one_half_females = 1/2 * f →
  (one_fourth_males + one_half_females) / (m + f) * total_degrees = 144 :=
by
  sorry

end career_preference_angles_l151_151820


namespace regular_polygon_sides_l151_151343

-- Define the problem conditions based on the given problem.
variables (n : ℕ)
def sum_of_interior_angles (n : ℕ) : ℤ := 180 * (n - 2)
def interior_angle (n : ℕ) : ℤ := 160
def total_interior_angle (n : ℕ) : ℤ := 160 * n

-- State the problem and expected result.
theorem regular_polygon_sides (h : 180 * (n - 2) = 160 * n) : n = 18 := 
by {
  -- This is to only state the theorem for now, proof will be crafted separately.
  sorry
}

end regular_polygon_sides_l151_151343


namespace score_order_l151_151760

variable (A B C D : ℕ)

theorem score_order (h1 : A + B = C + D) (h2 : C + A > B + D) (h3 : C > A + B) :
  (C > A ∧ A > B ∧ B > D) :=
by
  sorry

end score_order_l151_151760


namespace number_of_white_balls_l151_151700

-- Definitions based on the problem conditions
def total_balls : Nat := 120
def red_freq : ℝ := 0.15
def black_freq : ℝ := 0.45

-- Result to prove
theorem number_of_white_balls :
  let red_balls := total_balls * red_freq
  let black_balls := total_balls * black_freq
  total_balls - red_balls - black_balls = 48 :=
by
  sorry

end number_of_white_balls_l151_151700


namespace sale_in_third_month_l151_151816

def sales_in_months (m1 m2 m3 m4 m5 m6 : Int) : Prop :=
  m1 = 5124 ∧
  m2 = 5366 ∧
  m4 = 6124 ∧
  m6 = 4579 ∧
  (m1 + m2 + m3 + m4 + m5 + m6) / 6 = 5400

theorem sale_in_third_month (m5 : Int) :
  (∃ m3 : Int, sales_in_months 5124 5366 m3 6124 m5 4579 → m3 = 11207) :=
sorry

end sale_in_third_month_l151_151816


namespace find_other_number_l151_151525

def smallest_multiple_of_711 (n : ℕ) : ℕ := Nat.lcm n 711

theorem find_other_number (n : ℕ) : smallest_multiple_of_711 n = 3555 → n = 5 := by
  sorry

end find_other_number_l151_151525


namespace abs_neg_five_l151_151992

theorem abs_neg_five : abs (-5) = 5 :=
by
  sorry

end abs_neg_five_l151_151992


namespace triangular_square_l151_151168

def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_square (m n : ℕ) (h1 : 1 ≤ m) (h2 : 1 ≤ n) (h3 : 2 * triangular m = triangular n) :
  ∃ k : ℕ, triangular (2 * m - n) = k * k :=
by
  sorry

end triangular_square_l151_151168


namespace investment_c_is_correct_l151_151132

-- Define the investments of a and b
def investment_a : ℕ := 45000
def investment_b : ℕ := 63000
def profit_c : ℕ := 24000
def total_profit : ℕ := 60000

-- Define the equation to find the investment of c
def proportional_share (x y total : ℕ) : Prop :=
  2 * (x + y + total) = 5 * total

-- The theorem to prove c's investment given the conditions
theorem investment_c_is_correct (c : ℕ) (h_proportional: proportional_share investment_a investment_b c) :
  c = 72000 :=
by
  sorry

end investment_c_is_correct_l151_151132


namespace simplify_sqrt_l151_151402

theorem simplify_sqrt (a : ℝ) (h : a < 2) : Real.sqrt ((a - 2)^2) = 2 - a :=
by
  sorry

end simplify_sqrt_l151_151402


namespace evaluate_expression_l151_151605

theorem evaluate_expression : 3 + 5 * 2^3 - 4 / 2 + 7 * 3 = 62 := 
  by sorry

end evaluate_expression_l151_151605


namespace river_length_l151_151076

theorem river_length :
  let still_water_speed := 10 -- Karen's paddling speed on still water in miles per hour
  let current_speed      := 4  -- River's current speed in miles per hour
  let time               := 2  -- Time it takes Karen to paddle up the river in hours
  let effective_speed    := still_water_speed - current_speed -- Karen's effective speed against the current
  effective_speed * time = 12 -- Length of the river in miles
:= by
  sorry

end river_length_l151_151076


namespace larger_solution_quadratic_l151_151512

theorem larger_solution_quadratic :
  ∃ x : ℝ, x^2 - 13 * x + 30 = 0 ∧ (∀ y : ℝ, y^2 - 13 * y + 30 = 0 → y ≤ x) ∧ x = 10 := 
by
  sorry

end larger_solution_quadratic_l151_151512


namespace area_of_shaded_region_l151_151693

theorem area_of_shaded_region 
    (large_side : ℝ) (small_side : ℝ)
    (h_large : large_side = 10) 
    (h_small : small_side = 4) : 
    (large_side^2 - small_side^2) / 4 = 21 :=
by
  -- All proof steps are to be completed and checked,
  -- and sorry is used as placeholder for the final proof.
  sorry

end area_of_shaded_region_l151_151693


namespace people_with_uncool_parents_l151_151120

theorem people_with_uncool_parents :
  ∀ (total cool_dads cool_moms cool_both : ℕ),
    total = 50 →
    cool_dads = 25 →
    cool_moms = 30 →
    cool_both = 15 →
    (total - (cool_dads + cool_moms - cool_both)) = 10 := 
by
  intros total cool_dads cool_moms cool_both h1 h2 h3 h4
  sorry

end people_with_uncool_parents_l151_151120


namespace intersection_eq_l151_151829

def setA : Set ℝ := { y | ∃ x : ℝ, y = 2^x }
def setB : Set ℝ := { y | ∃ x : ℝ, y = x^2 }
def expectedIntersection : Set ℝ := { y | 0 < y }

theorem intersection_eq :
  setA ∩ setB = expectedIntersection :=
sorry

end intersection_eq_l151_151829


namespace bad_carrots_l151_151294

-- Conditions
def carrots_picked_by_vanessa := 17
def carrots_picked_by_mom := 14
def good_carrots := 24
def total_carrots := carrots_picked_by_vanessa + carrots_picked_by_mom

-- Question and Proof
theorem bad_carrots :
  total_carrots - good_carrots = 7 :=
by
  -- Placeholder for proof
  sorry

end bad_carrots_l151_151294


namespace neg_one_quadratic_residue_iff_l151_151616

theorem neg_one_quadratic_residue_iff (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1) : 
  (∃ x : ℤ, x^2 ≡ -1 [ZMOD p]) ↔ p % 4 = 1 :=
sorry

end neg_one_quadratic_residue_iff_l151_151616


namespace AC_amount_l151_151407

variable (A B C : ℝ)

theorem AC_amount
  (h1 : A + B + C = 400)
  (h2 : B + C = 150)
  (h3 : C = 50) :
  A + C = 300 := by
  sorry

end AC_amount_l151_151407


namespace red_peaches_count_l151_151418

/-- Math problem statement:
There are some red peaches and 16 green peaches in the basket.
There is 1 more red peach than green peaches in the basket.
Prove that the number of red peaches in the basket is 17.
--/

-- Let G be the number of green peaches and R be the number of red peaches.
def G : ℕ := 16
def R : ℕ := G + 1

theorem red_peaches_count : R = 17 := by
  sorry

end red_peaches_count_l151_151418


namespace lucky_point_m2_is_lucky_point_A33_point_M_quadrant_l151_151189

noncomputable def lucky_point (m n : ℝ) : Prop := 2 * m = 4 + n ∧ ∃ (x y : ℝ), (x = m - 1) ∧ (y = (n + 2) / 2)

theorem lucky_point_m2 :
  lucky_point 2 0 := sorry

theorem is_lucky_point_A33 :
  lucky_point 4 4 := sorry

theorem point_M_quadrant (a : ℝ) :
  lucky_point (a + 1) (2 * (2 * a - 1) - 2) → (a = 1) := sorry

end lucky_point_m2_is_lucky_point_A33_point_M_quadrant_l151_151189


namespace total_time_l151_151516

/-- Define the different time periods in years --/
def getting_in_shape : ℕ := 2
def learning_to_climb : ℕ := 2 * getting_in_shape
def months_climbing : ℕ := 7 * 5
def climbing : ℚ := months_climbing / 12
def break_after_climbing : ℚ := 13 / 12
def diving : ℕ := 2

/-- Prove that the total time taken to achieve all goals is 12 years --/
theorem total_time : getting_in_shape + learning_to_climb + climbing + break_after_climbing + diving = 12 := by
  sorry

end total_time_l151_151516


namespace smallest_three_digit_multiple_of_eleven_l151_151865

theorem smallest_three_digit_multiple_of_eleven : ∃ n, n = 110 ∧ 100 ≤ n ∧ n < 1000 ∧ 11 ∣ n := by
  sorry

end smallest_three_digit_multiple_of_eleven_l151_151865


namespace angle_C_in_triangle_l151_151794

theorem angle_C_in_triangle (A B C : ℝ) (h1 : A + B = 90) (h2 : A + B + C = 180) : C = 90 :=
sorry

end angle_C_in_triangle_l151_151794


namespace jill_trips_to_fill_tank_l151_151744

-- Definitions as per the conditions specified
def tank_capacity : ℕ := 600
def bucket_capacity : ℕ := 5
def jack_buckets_per_trip : ℕ := 2
def jill_buckets_per_trip : ℕ := 1
def jack_trips_ratio : ℕ := 3
def jill_trips_ratio : ℕ := 2
def leak_per_trip : ℕ := 2

-- Prove that the number of trips Jill will make = 20 given the above conditions
theorem jill_trips_to_fill_tank : 
  (jack_buckets_per_trip * bucket_capacity + jill_buckets_per_trip * bucket_capacity - leak_per_trip) * (tank_capacity / ((jack_trips_ratio + jill_trips_ratio) * (jack_buckets_per_trip * bucket_capacity + jill_buckets_per_trip * bucket_capacity - leak_per_trip) / (jack_trips_ratio + jill_trips_ratio)))  = 20 := 
sorry

end jill_trips_to_fill_tank_l151_151744


namespace beef_stew_duration_l151_151795

noncomputable def original_portions : ℕ := 14
noncomputable def your_portion : ℕ := 1
noncomputable def roommate_portion : ℕ := 3
noncomputable def guest_portion : ℕ := 4
noncomputable def total_daily_consumption : ℕ := your_portion + roommate_portion + guest_portion
noncomputable def days_stew_lasts : ℕ := original_portions / total_daily_consumption

theorem beef_stew_duration : days_stew_lasts = 2 :=
by
  sorry

end beef_stew_duration_l151_151795


namespace cone_height_ratio_l151_151854

theorem cone_height_ratio (circumference : ℝ) (orig_height : ℝ) (short_volume : ℝ)
  (h_circumference : circumference = 20 * Real.pi)
  (h_orig_height : orig_height = 40)
  (h_short_volume : short_volume = 400 * Real.pi) :
  let r := circumference / (2 * Real.pi)
  let h_short := (3 * short_volume) / (Real.pi * r^2)
  (h_short / orig_height) = 3 / 10 :=
by {
  sorry
}

end cone_height_ratio_l151_151854


namespace quadratic_eq_roots_quadratic_eq_positive_integer_roots_l151_151716

theorem quadratic_eq_roots (m : ℝ) (hm : m ≠ 0 ∧ m ≤ 9 / 8) :
  ∃ x1 x2 : ℝ, (m * x1^2 + 3 * x1 + 2 = 0) ∧ (m * x2^2 + 3 * x2 + 2 = 0) :=
sorry

theorem quadratic_eq_positive_integer_roots (m : ℕ) (hm : m = 1) :
  ∃ x1 x2 : ℝ, (x1 = -1) ∧ (x2 = -2) ∧ (m * x1^2 + 3 * x1 + 2 = 0) ∧ (m * x2^2 + 3 * x2 + 2 = 0) :=
sorry

end quadratic_eq_roots_quadratic_eq_positive_integer_roots_l151_151716


namespace area_of_triangle_PQR_l151_151436

theorem area_of_triangle_PQR 
  (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 8)
  (P_is_center : ∃ P : ℝ, True) -- Simplified assumption that P exists
  (bases_on_same_line : True) -- Assumed true, as touching condition implies it
  : ∃ area : ℝ, area = 20 := 
by
  sorry

end area_of_triangle_PQR_l151_151436


namespace tiles_needed_l151_151845

-- Definitions of the given conditions
def side_length_smaller_tile : ℝ := 0.3
def number_smaller_tiles : ℕ := 500
def side_length_larger_tile : ℝ := 0.5

-- Statement to prove the required number of larger tiles
theorem tiles_needed (x : ℕ) :
  side_length_larger_tile * side_length_larger_tile * x =
  side_length_smaller_tile * side_length_smaller_tile * number_smaller_tiles →
  x = 180 :=
by
  sorry

end tiles_needed_l151_151845


namespace explicit_formula_l151_151310

def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem explicit_formula (x1 x2 : ℝ) (h1 : x1 ∈ Set.Icc (-1 : ℝ) 1) (h2 : x2 ∈ Set.Icc (-1 : ℝ) 1) :
  f x = x^3 - 3 * x ∧ |f x1 - f x2| ≤ 4 :=
by
  sorry

end explicit_formula_l151_151310


namespace opposite_of_2023_l151_151783

/-- The opposite of a number n is defined as the number that, when added to n, results in zero. -/
def opposite (n : ℤ) : ℤ := -n

theorem opposite_of_2023 : opposite 2023 = -2023 :=
by 
  sorry

end opposite_of_2023_l151_151783


namespace find_number_of_numbers_l151_151543

theorem find_number_of_numbers (S : ℝ) (n : ℝ) (h1 : S - 30 = 16 * n) (h2 : S = 19 * n) : n = 10 :=
by
  sorry

end find_number_of_numbers_l151_151543


namespace JacobNeed_l151_151936

-- Definitions of the conditions
def jobEarningsBeforeTax : ℝ := 25 * 15
def taxAmount : ℝ := 0.10 * jobEarningsBeforeTax
def jobEarningsAfterTax : ℝ := jobEarningsBeforeTax - taxAmount

def cookieEarnings : ℝ := 5 * 30

def tutoringEarnings : ℝ := 100 * 4

def lotteryWinnings : ℝ := 700 - 20
def friendShare : ℝ := 0.30 * lotteryWinnings
def netLotteryWinnings : ℝ := lotteryWinnings - friendShare

def giftFromSisters : ℝ := 700 * 2

def totalEarnings : ℝ := jobEarningsAfterTax + cookieEarnings + tutoringEarnings + netLotteryWinnings + giftFromSisters

def travelGearExpenses : ℝ := 3 + 47

def netSavings : ℝ := totalEarnings - travelGearExpenses

def tripCost : ℝ := 8000

-- Statement to be proven
theorem JacobNeed (jobEarningsBeforeTax taxAmount jobEarningsAfterTax cookieEarnings tutoringEarnings 
netLotteryWinnings giftFromSisters totalEarnings travelGearExpenses netSavings tripCost : ℝ) : 
  (jobEarningsAfterTax == (25 * 15) - (0.10 * (25 * 15))) → 
  (cookieEarnings == 5 * 30) →
  (tutoringEarnings == 100 * 4) →
  (netLotteryWinnings == (700 - 20) - (0.30 * (700 - 20))) →
  (giftFromSisters == 700 * 2) →
  (totalEarnings == jobEarningsAfterTax + cookieEarnings + tutoringEarnings + netLotteryWinnings + giftFromSisters) →
  (travelGearExpenses == 3 + 47) →
  (netSavings == totalEarnings - travelGearExpenses) →
  (tripCost == 8000) →
  (tripCost - netSavings = 5286.50) :=
by
  intros
  sorry

end JacobNeed_l151_151936


namespace part1_part2_l151_151881

noncomputable def f (x m : ℝ) := |x + 1| + |m - x|

theorem part1 (x : ℝ) : (f x 3) ≥ 6 ↔ (x ≤ -2 ∨ x ≥ 4) :=
by sorry

theorem part2 (m : ℝ) : (∀ x, f x m ≥ 8) ↔ (m ≥ 7 ∨ m ≤ -9) :=
by sorry

end part1_part2_l151_151881


namespace simplify_expression_l151_151146

theorem simplify_expression (x : ℝ) :
  ( ( ((x + 1) ^ 3 * (x ^ 2 - x + 1) ^ 3) / (x ^ 3 + 1) ^ 3 ) ^ 2 *
    ( ((x - 1) ^ 3 * (x ^ 2 + x + 1) ^ 3) / (x ^ 3 - 1) ^ 3 ) ^ 2 ) = 1 :=
by
  sorry

end simplify_expression_l151_151146


namespace minimum_perimeter_l151_151258

/-
Given:
1. (a: ℤ), (b: ℤ), (c: ℤ)
2. (a ≠ b)
3. 2 * a + 10 * c = 2 * b + 8 * c
4. 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2
5. 10 * c / 8 * c = 5 / 4

Prove:
The minimum perimeter is 1180 
-/

theorem minimum_perimeter (a b c : ℤ) 
(h1 : a ≠ b)
(h2 : 2 * a + 10 * c = 2 * b + 8 * c)
(h3 : 25 * a^2 - 625 * c^2 = 16 * b^2 - 256 * c^2)
(h4 : 10 * c / 8 * c = 5 / 4) :
2 * a + 10 * c = 1180 ∨ 2 * b + 8 * c = 1180 :=
sorry

end minimum_perimeter_l151_151258


namespace divisor_exists_l151_151500

theorem divisor_exists (n : ℕ) : (∃ k, 10 ≤ k ∧ k ≤ 50 ∧ n ∣ k) →
                                (∃ k, 10 ≤ k ∧ k ≤ 50 ∧ n ∣ k) ∧
                                (n = 3) :=
by
  sorry

end divisor_exists_l151_151500


namespace length_AB_l151_151330

theorem length_AB 
  (P : ℝ × ℝ) 
  (hP : 3 * P.1 + 4 * P.2 + 8 = 0)
  (C : ℝ × ℝ := (1, 1))
  (A B : ℝ × ℝ)
  (hA : (A.1 - 1)^2 + (A.2 - 1)^2 = 1 ∧ (3 * A.1 + 4 * A.2 + 8 ≠ 0))
  (hB : (B.1 - 1)^2 + (B.2 - 1)^2 = 1 ∧ (3 * B.1 + 4 * B.2 + 8 ≠ 0)) :
  dist A B = 4 * Real.sqrt 2 / 3 := sorry

end length_AB_l151_151330


namespace non_juniors_play_instrument_l151_151123

theorem non_juniors_play_instrument (total_students juniors non_juniors play_instrument_juniors play_instrument_non_juniors total_do_not_play : ℝ) :
  total_students = 600 →
  play_instrument_juniors = 0.3 * juniors →
  play_instrument_non_juniors = 0.65 * non_juniors →
  total_do_not_play = 0.4 * total_students →
  0.7 * juniors + 0.35 * non_juniors = total_do_not_play →
  juniors + non_juniors = total_students →
  non_juniors * 0.65 = 334 :=
by
  sorry

end non_juniors_play_instrument_l151_151123


namespace probability_of_roots_l151_151529

theorem probability_of_roots (k : ℝ) (h1 : 8 ≤ k) (h2 : k ≤ 13) :
  let a := k^2 - 2 * k - 35
  let b := 3 * k - 9
  let c := 2
  let discriminant := b^2 - 4 * a * c
  discriminant ≥ 0 → 
  (∃ x1 x2 : ℝ, 
    a * x1^2 + b * x1 + c = 0 ∧ 
    a * x2^2 + b * x2 + c = 0 ∧
    x1 ≤ 2 * x2) ↔ 
  ∃ p : ℝ, p = 0.6 := 
sorry

end probability_of_roots_l151_151529


namespace arithmetic_sequence_tenth_term_l151_151184

theorem arithmetic_sequence_tenth_term (a d : ℤ) (h₁ : a + 3 * d = 23) (h₂ : a + 8 * d = 38) : a + 9 * d = 41 := by
  sorry

end arithmetic_sequence_tenth_term_l151_151184


namespace geometric_sequence_b_value_l151_151121

theorem geometric_sequence_b_value (a b c : ℝ) (h : 1 * a = a * b ∧ a * b = b * c ∧ b * c = c * 5) : b = Real.sqrt 5 :=
sorry

end geometric_sequence_b_value_l151_151121


namespace smallest_m_n_sum_l151_151138

noncomputable def f (m n : ℕ) (x : ℝ) : ℝ := Real.arcsin (Real.log (n * x) / Real.log m)

theorem smallest_m_n_sum 
  (m n : ℕ) 
  (h_m1 : 1 < m) 
  (h_mn_closure : ∀ x, -1 ≤ Real.log (n * x) / Real.log m ∧ Real.log (n * x) / Real.log m ≤ 1) 
  (h_length : (m ^ 2 - 1) / (m * n) = 1 / 2021) : 
  m + n = 86259 := by
sorry

end smallest_m_n_sum_l151_151138


namespace hypotenuse_length_l151_151280

-- Definition of the right triangle with the given leg lengths
structure RightTriangle :=
  (BC AC AB : ℕ)
  (right : BC^2 + AC^2 = AB^2)

-- The theorem we need to prove
theorem hypotenuse_length (T : RightTriangle) (h1 : T.BC = 5) (h2 : T.AC = 12) :
  T.AB = 13 :=
by
  sorry

end hypotenuse_length_l151_151280


namespace log_400_cannot_be_computed_l151_151299

theorem log_400_cannot_be_computed :
  let log_8 : ℝ := 0.9031
  let log_9 : ℝ := 0.9542
  let log_7 : ℝ := 0.8451
  (∀ (log_2 log_3 log_5 : ℝ), log_2 = 1 / 3 * log_8 → log_3 = 1 / 2 * log_9 → log_5 = 1 → 
    (∀ (log_val : ℝ), 
      (log_val = log_21 → log_21 = log_3 + log_7 → log_val = (1 / 2) * log_9 + log_7)
      ∧ (log_val = log_9_over_8 → log_9_over_8 = log_9 - log_8)
      ∧ (log_val = log_126 → log_126 = log_2 + log_7 + log_9 → log_val = (1 / 3) * log_8 + log_7 + log_9)
      ∧ (log_val = log_0_875 → log_0_875 = log_7 - log_8)
      ∧ (log_val = log_400 → log_400 = log_8 + 1 + log_5) 
      → False))
:= 
sorry

end log_400_cannot_be_computed_l151_151299


namespace consecutive_odds_base_eqn_l151_151125

-- Given conditions
def isOdd (n : ℕ) : Prop := n % 2 = 1

variables {C D : ℕ}

theorem consecutive_odds_base_eqn (C_odd : isOdd C) (D_odd : isOdd D) (consec : D = C + 2)
    (base_eqn : 2 * C^2 + 4 * C + 3 + 6 * D + 5 = 10 * (C + D) + 7) :
    C + D = 16 :=
sorry

end consecutive_odds_base_eqn_l151_151125


namespace sum_of_common_ratios_eq_three_l151_151954

theorem sum_of_common_ratios_eq_three
  (k a2 a3 b2 b3 : ℕ)
  (p r : ℕ)
  (h_nonconst1 : k ≠ 0)
  (h_nonconst2 : p ≠ r)
  (h_seq1 : a3 = k * p ^ 2)
  (h_seq2 : b3 = k * r ^ 2)
  (h_seq3 : a2 = k * p)
  (h_seq4 : b2 = k * r)
  (h_eq : a3 - b3 = 3 * (a2 - b2)) :
  p + r = 3 := 
sorry

end sum_of_common_ratios_eq_three_l151_151954


namespace total_stamps_l151_151130

def c : ℕ := 578833
def bw : ℕ := 523776
def total : ℕ := 1102609

theorem total_stamps : c + bw = total := 
by 
  sorry

end total_stamps_l151_151130


namespace simplify_fraction_result_l151_151506

theorem simplify_fraction_result :
  (144: ℝ) / 1296 * 72 = 8 :=
by
  sorry

end simplify_fraction_result_l151_151506


namespace find_four_digit_number_l151_151241

variable {N : ℕ} {a x y : ℕ}

theorem find_four_digit_number :
  (∃ a x y : ℕ, y < 10 ∧ 10 + a = x * y ∧ x = 9 + a ∧ N = 1000 + a + 10 * b + 100 * b ∧
  (N = 1014 ∨ N = 1035 ∨ N = 1512)) :=
by
  sorry

end find_four_digit_number_l151_151241


namespace max_odd_integers_l151_151029

theorem max_odd_integers (a1 a2 a3 a4 a5 a6 a7 : ℕ) (hpos : ∀ i, i ∈ [a1, a2, a3, a4, a5, a6, a7] → i > 0) 
  (hprod : a1 * a2 * a3 * a4 * a5 * a6 * a7 % 2 = 0) : 
  ∃ l : List ℕ, l.length = 6 ∧ (∀ i, i ∈ l → i % 2 = 1) ∧ ∃ e : ℕ, e % 2 = 0 ∧ e ∈ [a1, a2, a3, a4, a5, a6, a7] :=
by
  sorry

end max_odd_integers_l151_151029


namespace speed_difference_l151_151826

noncomputable def park_distance : ℝ := 10
noncomputable def kevin_time_hours : ℝ := 1 / 4
noncomputable def joel_time_hours : ℝ := 2

theorem speed_difference : (10 / kevin_time_hours) - (10 / joel_time_hours) = 35 := by
  sorry

end speed_difference_l151_151826


namespace product_of_roots_l151_151086

theorem product_of_roots (p q r : ℝ)
  (h1 : ∀ x : ℝ, (3 * x^3 - 9 * x^2 + 5 * x - 15 = 0) → (x = p ∨ x = q ∨ x = r)) :
  p * q * r = 5 := by
  sorry

end product_of_roots_l151_151086


namespace lloyd_normal_hours_l151_151665

-- Definitions based on the conditions
def regular_rate : ℝ := 3.50
def overtime_rate : ℝ := 1.5 * regular_rate
def total_hours_worked : ℝ := 10.5
def total_earnings : ℝ := 42
def normal_hours_worked (h : ℝ) : Prop := 
  h * regular_rate + (total_hours_worked - h) * overtime_rate = total_earnings

-- The theorem to prove
theorem lloyd_normal_hours : ∃ h : ℝ, normal_hours_worked h ∧ h = 7.5 := sorry

end lloyd_normal_hours_l151_151665


namespace student_B_most_stable_l151_151175

variable (S_A S_B S_C : ℝ)
variables (hA : S_A^2 = 2.6) (hB : S_B^2 = 1.7) (hC : S_C^2 = 3.5)

/-- Student B has the most stable performance among students A, B, and C based on their variances.
    Given the conditions:
    - S_A^2 = 2.6
    - S_B^2 = 1.7
    - S_C^2 = 3.5
    we prove that student B has the most stable performance.
-/
theorem student_B_most_stable : S_B^2 < S_A^2 ∧ S_B^2 < S_C^2 :=
by
  -- Proof goes here
  sorry

end student_B_most_stable_l151_151175


namespace Cherry_weekly_earnings_l151_151988

theorem Cherry_weekly_earnings :
  let cost_3_5 := 2.50
  let cost_6_8 := 4.00
  let cost_9_12 := 6.00
  let cost_13_15 := 8.00
  let num_5kg := 4
  let num_8kg := 2
  let num_10kg := 3
  let num_14kg := 1
  let daily_earnings :=
    (num_5kg * cost_3_5) + (num_8kg * cost_6_8) + (num_10kg * cost_9_12) + (num_14kg * cost_13_15)
  let weekly_earnings := daily_earnings * 7
  weekly_earnings = 308 := by
  sorry

end Cherry_weekly_earnings_l151_151988


namespace proof_of_problem_l151_151787

noncomputable def proof_problem : Prop :=
  ∀ x : ℝ, (x + 2) ^ (x + 3) = 1 ↔ (x = -1 ∨ x = -3)

theorem proof_of_problem : proof_problem :=
by
  sorry

end proof_of_problem_l151_151787


namespace find_a_and_b_min_value_expression_l151_151611

universe u

-- Part (1): Prove the values of a and b
theorem find_a_and_b :
    (∀ x : ℝ, a * x^2 - 3 * x + 2 > 0 ↔ x < 1 ∨ x > b) →
    a = 1 ∧ b = 2 :=
sorry

-- Part (2): Given a = 1 and b = 2 prove the minimum value of 2x + y + 3
theorem min_value_expression :
    (1 / (x + 1) + 2 / (y + 1) = 1) →
    (x > 0) →
    (y > 0) →
    ∀ x y : ℝ, 2 * x + y + 3 ≥ 8 :=
sorry

end find_a_and_b_min_value_expression_l151_151611


namespace percentage_of_apples_sold_l151_151234

variables (A P : ℝ) 

theorem percentage_of_apples_sold :
  (A = 700) →
  (A * (1 - P / 100) = 420) →
  (P = 40) :=
by
  intros h1 h2
  sorry

end percentage_of_apples_sold_l151_151234


namespace min_value_expression_l151_151582

theorem min_value_expression :
  ∀ (x y z w : ℝ), x > 0 → y > 0 → z > 0 → w > 0 → x = y → x + y + z + w = 1 →
  (x + y + z) / (x * y * z * w) ≥ 1024 :=
by
  intros x y z w hx hy hz hw hxy hsum
  sorry

end min_value_expression_l151_151582


namespace sum_of_squares_l151_151323

theorem sum_of_squares (x y : ℝ) (h₁ : x + y = 16) (h₂ : x * y = 28) : x^2 + y^2 = 200 :=
by
  sorry

end sum_of_squares_l151_151323


namespace model_to_statue_ratio_l151_151288

theorem model_to_statue_ratio (h_statue : ℝ) (h_model : ℝ) (h_statue_eq : h_statue = 60) (h_model_eq : h_model = 4) :
  (h_statue / h_model) = 15 := by
  sorry

end model_to_statue_ratio_l151_151288


namespace vertical_distance_rotated_square_l151_151397

-- Lean 4 statement for the mathematically equivalent proof problem
theorem vertical_distance_rotated_square
  (side_length : ℝ)
  (n : ℕ)
  (rot_angle : ℝ)
  (orig_line_height before_rotation : ℝ)
  (diagonal_length : ℝ)
  (lowered_distance : ℝ)
  (highest_point_drop : ℝ)
  : side_length = 2 →
    n = 4 →
    rot_angle = 45 →
    orig_line_height = 1 →
    diagonal_length = side_length * (2:ℝ)^(1/2) →
    lowered_distance = (diagonal_length / 2) - orig_line_height →
    highest_point_drop = lowered_distance →
    2 = 2 :=
    sorry

end vertical_distance_rotated_square_l151_151397


namespace point_not_on_line_l151_151009

theorem point_not_on_line (m b : ℝ) (h1 : m > 2) (h2 : m * b > 0) : ¬ (b = -2023) :=
by
  sorry

end point_not_on_line_l151_151009


namespace find_value_l151_151846

-- Define the mean, standard deviation, and the number of standard deviations
def mean : ℝ := 17.5
def std_dev : ℝ := 2.5
def num_std_dev : ℝ := 2.7

-- The theorem to prove that the value is exactly 10.75
theorem find_value : mean - (num_std_dev * std_dev) = 10.75 := 
by
  sorry

end find_value_l151_151846


namespace solve_fractional_equation_l151_151686

theorem solve_fractional_equation (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 4) : (1 / (x - 1) = 2 / (1 - x) + 1) → x = 4 :=
by
  sorry

end solve_fractional_equation_l151_151686


namespace decode_plaintext_l151_151849

theorem decode_plaintext (a x y : ℕ) (h1 : y = a^x - 2) (h2 : 6 = a^3 - 2) (h3 : y = 14) : x = 4 := by
  sorry

end decode_plaintext_l151_151849


namespace first_offset_length_l151_151364

theorem first_offset_length (diagonal : ℝ) (offset2 : ℝ) (area : ℝ) (h_diagonal : diagonal = 50) (h_offset2 : offset2 = 8) (h_area : area = 450) :
  ∃ offset1 : ℝ, offset1 = 10 :=
by
  sorry

end first_offset_length_l151_151364


namespace find_side_c_l151_151901

theorem find_side_c (a C S : ℝ) (ha : a = 3) (hC : C = 120) (hS : S = (15 * Real.sqrt 3) / 4) : 
  ∃ (c : ℝ), c = 7 :=
by
  sorry

end find_side_c_l151_151901


namespace expression_change_l151_151262

theorem expression_change (a b c : ℝ) : 
  a - (2 * b - 3 * c) = a + (-2 * b + 3 * c) := 
by sorry

end expression_change_l151_151262


namespace midpoint_of_AB_l151_151780

theorem midpoint_of_AB (xA xB : ℝ) (p : ℝ) (h_parabola : ∀ y, y^2 = 4 * xA → y^2 = 4 * xB)
  (h_focus : (2 : ℝ) = p)
  (h_length_AB : (abs (xB - xA)) = 5) :
  (xA + xB) / 2 = 3 / 2 :=
sorry

end midpoint_of_AB_l151_151780


namespace calculate_star_operation_l151_151149

def operation (a b : ℚ) : ℚ := 2 * a - b + 1

theorem calculate_star_operation :
  operation 1 (operation 3 (-2)) = -6 :=
by
  sorry

end calculate_star_operation_l151_151149


namespace diamonds_in_G_20_equals_840_l151_151155

def diamonds_in_G (n : ℕ) : ℕ :=
  if n < 3 then 1 else 2 * n * (n + 1)

theorem diamonds_in_G_20_equals_840 : diamonds_in_G 20 = 840 :=
by
  sorry

end diamonds_in_G_20_equals_840_l151_151155


namespace remaining_budget_l151_151093

def charge_cost : ℝ := 3.5
def num_charges : ℝ := 4
def total_budget : ℝ := 20

theorem remaining_budget : total_budget - (num_charges * charge_cost) = 6 := 
by 
  sorry

end remaining_budget_l151_151093


namespace base_r_5555_square_palindrome_l151_151329

theorem base_r_5555_square_palindrome (r : ℕ) (a b c d : ℕ) 
  (h1 : r % 2 = 0) 
  (h2 : r >= 18) 
  (h3 : d - c = 2)
  (h4 : ∀ x, (x = 5 * r^3 + 5 * r^2 + 5 * r + 5) → 
    (x^2 = a * r^7 + b * r^6 + c * r^5 + d * r^4 + d * r^3 + c * r^2 + b * r + a)) : 
  r = 24 := 
sorry

end base_r_5555_square_palindrome_l151_151329


namespace incorrect_statement_D_l151_151389

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 + x
else -(x^2 + x)

theorem incorrect_statement_D : ¬(∀ x : ℝ, x ≤ 0 → f x = x^2 + x) :=
by
  sorry

end incorrect_statement_D_l151_151389


namespace wheel_radius_increase_l151_151403

theorem wheel_radius_increase 
  (d₁ d₂ : ℝ) -- distances according to the odometer (600 and 580 miles)
  (r₀ : ℝ)   -- original radius (17 inches)
  (C₁: d₁ = 600)
  (C₂: d₂ = 580)
  (C₃: r₀ = 17) :
  ∃ Δr : ℝ, Δr = 0.57 :=
by
  sorry

end wheel_radius_increase_l151_151403


namespace find_m_value_l151_151999

noncomputable def is_solution (p q m : ℝ) : Prop :=
  ∃ x : ℝ, (x^2 + p*x + q = 0) ∧ (x^2 - m*x + m^2 - 19 = 0)

theorem find_m_value :
  let A := { x : ℝ | x^2 + 2 * x - 8 = 0 }
  let B := { x : ℝ | x^2 - 5 * x + 6 = 0 }
  ∀ (C : ℝ → Prop), 
  (∃ x, B x ∧ C x) ∧ (¬ ∃ x, A x ∧ C x) → 
  (∃ m, C = { x : ℝ | x^2 - m * x + m^2 - 19 = 0 } ∧ m = -2) :=
by
  sorry

end find_m_value_l151_151999


namespace quadratic_roots_l151_151013

theorem quadratic_roots (a : ℝ) (k c : ℝ) : 
    (∀ x : ℝ, 2 * x^2 + k * x + c = 0 ↔ (x = 7 ∨ x = a)) →
    k = -2 * a - 14 ∧ c = 14 * a :=
by
  sorry

end quadratic_roots_l151_151013


namespace weight_of_5_diamonds_l151_151324

-- Define the weight of one diamond and one jade
variables (D J : ℝ)

-- Conditions:
-- 1. Total weight of 4 diamonds and 2 jades
def condition1 : Prop := 4 * D + 2 * J = 140
-- 2. A jade is 10 g heavier than a diamond
def condition2 : Prop := J = D + 10

-- Total weight of 5 diamonds
def total_weight_of_5_diamonds : ℝ := 5 * D

-- Theorem: Prove that the total weight of 5 diamonds is 100 g
theorem weight_of_5_diamonds (h1 : condition1 D J) (h2 : condition2 D J) : total_weight_of_5_diamonds D = 100 :=
by {
  sorry
}

end weight_of_5_diamonds_l151_151324


namespace tom_gave_2_seashells_to_jessica_l151_151730

-- Conditions
def original_seashells : Nat := 5
def current_seashells : Nat := 3

-- Question as a proposition
def seashells_given (x : Nat) : Prop :=
  original_seashells - current_seashells = x

-- The proof problem
theorem tom_gave_2_seashells_to_jessica : seashells_given 2 :=
by 
  sorry

end tom_gave_2_seashells_to_jessica_l151_151730


namespace score_calculation_l151_151228

theorem score_calculation (N : ℕ) (C : ℕ) (hN: 1 ≤ N ∧ N ≤ 20) (hC: 1 ≤ C) : 
  ∃ (score: ℕ), score = Nat.floor (N / C) :=
by sorry

end score_calculation_l151_151228


namespace quadratic_expression_positive_l151_151156

theorem quadratic_expression_positive
  (a b c : ℝ) (x : ℝ)
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  b^2 * x^2 + (b^2 + c^2 - a^2) * x + c^2 > 0 :=
sorry

end quadratic_expression_positive_l151_151156


namespace plus_minus_pairs_l151_151358

theorem plus_minus_pairs (a b p q : ℕ) (h_plus_pairs : p = a) (h_minus_pairs : q = b) : 
  a - b = p - q := 
by 
  sorry

end plus_minus_pairs_l151_151358


namespace equation_of_the_line_l151_151676

theorem equation_of_the_line (a b : ℝ) :
    ((a - b = 5) ∧ (9 / a + 4 / b = 1)) → 
    ( (2 * 9 + 3 * 4 - 30 = 0) ∨ (2 * 9 - 3 * 4 - 6 = 0) ∨ (9 - 4 - 5 = 0)) :=
  by
    sorry

end equation_of_the_line_l151_151676


namespace bacon_strips_needed_l151_151607

theorem bacon_strips_needed (plates : ℕ) (eggs_per_plate : ℕ) (bacon_per_plate : ℕ) (customers : ℕ) :
  eggs_per_plate = 2 →
  bacon_per_plate = 2 * eggs_per_plate →
  customers = 14 →
  plates = customers →
  plates * bacon_per_plate = 56 := by
  sorry

end bacon_strips_needed_l151_151607


namespace no_real_solution_l151_151312

theorem no_real_solution (x : ℝ) : x + 64 / (x + 3) ≠ -13 :=
by {
  -- Proof is not required, so we mark it as sorry.
  sorry
}

end no_real_solution_l151_151312


namespace malcolm_red_lights_bought_l151_151483

-- Define the problem's parameters and conditions
variable (R : ℕ) (B : ℕ := 3 * R) (G : ℕ := 6)
variable (initial_white_lights : ℕ := 59) (remaining_colored_lights : ℕ := 5)

-- The total number of colored lights that he still needs to replace the white lights
def total_colored_lights_needed : ℕ := initial_white_lights - remaining_colored_lights

-- Total colored lights bought so far
def total_colored_lights_bought : ℕ := R + B + G

-- The main theorem to prove that Malcolm bought 12 red lights
theorem malcolm_red_lights_bought (h : total_colored_lights_bought = total_colored_lights_needed) :
  R = 12 := by
  sorry

end malcolm_red_lights_bought_l151_151483


namespace problem_l151_151921

noncomputable def x : ℝ := Real.sqrt 3 + Real.sqrt 2
noncomputable def y : ℝ := Real.sqrt 3 - Real.sqrt 2

theorem problem (x y : ℝ) (hx : x = Real.sqrt 3 + Real.sqrt 2) (hy : y = Real.sqrt 3 - Real.sqrt 2) :
  x * y^2 - x^2 * y = -2 * Real.sqrt 2 :=
by
  rw [hx, hy]
  sorry

end problem_l151_151921


namespace product_has_correct_sign_and_units_digit_l151_151439

noncomputable def product_negative_integers_divisible_by_3_less_than_198 : ℤ :=
  sorry

theorem product_has_correct_sign_and_units_digit :
  product_negative_integers_divisible_by_3_less_than_198 < 0 ∧
  product_negative_integers_divisible_by_3_less_than_198 % 10 = 6 :=
by
  sorry

end product_has_correct_sign_and_units_digit_l151_151439


namespace least_positive_integer_condition_l151_151369

theorem least_positive_integer_condition :
  ∃ n > 1, (∀ k ∈ [3, 4, 5, 6, 7, 8, 9, 10], n % k = 1) → n = 25201 := by
  sorry

end least_positive_integer_condition_l151_151369


namespace speed_ratio_bus_meets_Vasya_first_back_trip_time_l151_151658

namespace TransportProblem

variable (d : ℝ) -- distance from point A to B
variable (v_bus : ℝ) -- bus speed
variable (v_Vasya : ℝ) -- Vasya's speed
variable (v_Petya : ℝ) -- Petya's speed

-- Conditions
axiom bus_speed : v_bus * 3 = d
axiom bus_meet_Vasya_second_trip : 7.5 * v_Vasya = 0.5 * d
axiom bus_meet_Petya_at_B : 9 * v_Petya = d
axiom bus_start_time : d / v_bus = 3

theorem speed_ratio: (v_Vasya / v_Petya) = (3 / 5) :=
  sorry

theorem bus_meets_Vasya_first_back_trip_time: ∃ (x: ℕ), x = 11 :=
  sorry

end TransportProblem

end speed_ratio_bus_meets_Vasya_first_back_trip_time_l151_151658


namespace polynomial_not_factorable_l151_151804

theorem polynomial_not_factorable (b c d : Int) (h₁ : (b * d + c * d) % 2 = 1) : 
  ¬ ∃ p q r : Int, (x + p) * (x^2 + q * x + r) = x^3 + b * x^2 + c * x + d :=
by 
  sorry

end polynomial_not_factorable_l151_151804


namespace find_f3_l151_151712

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 6

theorem find_f3 (a b c : ℝ) (h : f a b c (-3) = -12) : f a b c 3 = 24 :=
by
  sorry

end find_f3_l151_151712


namespace smallest_n_for_congruence_l151_151599

theorem smallest_n_for_congruence : ∃ n : ℕ, 0 < n ∧ 7^n % 5 = n^4 % 5 ∧ (∀ m : ℕ, 0 < m ∧ 7^m % 5 = m^4 % 5 → n ≤ m) ∧ n = 4 :=
by
  sorry

end smallest_n_for_congruence_l151_151599


namespace high_temp_three_years_same_l151_151105

theorem high_temp_three_years_same
  (T : ℝ)                               -- The high temperature for the three years with the same temperature
  (temp2017 : ℝ := 79)                   -- The high temperature for 2017
  (temp2016 : ℝ := 71)                   -- The high temperature for 2016
  (average_temp : ℝ := 84)               -- The average high temperature for 5 years
  (num_years : ℕ := 5)                   -- The number of years to consider
  (years_with_same_temp : ℕ := 3)        -- The number of years with the same high temperature
  (total_temp : ℝ := average_temp * num_years) -- The sum of the high temperatures for the 5 years
  (total_known_temp : ℝ := temp2017 + temp2016) -- The known high temperatures for 2016 and 2017
  (total_for_three_years : ℝ := total_temp - total_known_temp) -- Total high temperatures for the three years
  (high_temp_per_year : ℝ := total_for_three_years / years_with_same_temp) -- High temperature per year for three years
  :
  T = 90 :=
sorry

end high_temp_three_years_same_l151_151105


namespace garden_length_l151_151505

theorem garden_length (P B : ℕ) (h₁ : P = 600) (h₂ : B = 95) : (∃ L : ℕ, 2 * (L + B) = P ∧ L = 205) :=
by
  sorry

end garden_length_l151_151505


namespace least_positive_integer_exists_l151_151953

theorem least_positive_integer_exists 
  (exists_k : ∃ k, (1 ≤ k ∧ k ≤ 2 * 5) ∧ (5^2 - 5 + k) % k = 0)
  (not_all_k : ¬(∀ k, (1 ≤ k ∧ k ≤ 2 * 5) → (5^2 - 5 + k) % k = 0)) :
  5 = 5 := 
by
  trivial

end least_positive_integer_exists_l151_151953


namespace kim_driving_speed_l151_151103

open Nat
open Real

noncomputable def driving_speed (distance there distance_back time_spent traveling_time total_time: ℝ) : ℝ :=
  (distance + distance_back) / traveling_time

theorem kim_driving_speed:
  ∀ (distance there distance_back time_spent traveling_time total_time: ℝ),
  distance = 30 →
  distance_back = 30 * 1.20 →
  total_time = 2 →
  time_spent = 0.5 →
  traveling_time = total_time - time_spent →
  driving_speed distance there distance_back time_spent traveling_time total_time = 44 :=
by
  intros
  simp only [driving_speed]
  sorry

end kim_driving_speed_l151_151103


namespace solve_quadratic_equation_l151_151077

noncomputable def f (x : ℝ) := 
  5 / (Real.sqrt (x - 9) - 8) - 
  2 / (Real.sqrt (x - 9) - 5) + 
  6 / (Real.sqrt (x - 9) + 5) - 
  9 / (Real.sqrt (x - 9) + 8)

theorem solve_quadratic_equation :
  ∀ (x : ℝ), x ≥ 9 → f x = 0 → 
  x = 19.2917 ∨ x = 8.9167 :=
by sorry

end solve_quadratic_equation_l151_151077


namespace johns_total_due_l151_151349

noncomputable def total_amount_due (initial_amount : ℝ) (first_charge_rate : ℝ) 
  (second_charge_rate : ℝ) (third_charge_rate : ℝ) : ℝ := 
  let after_first_charge := initial_amount * first_charge_rate
  let after_second_charge := after_first_charge * second_charge_rate
  let after_third_charge := after_second_charge * third_charge_rate
  after_third_charge

theorem johns_total_due : total_amount_due 500 1.02 1.03 1.025 = 538.43 := 
  by
    -- The proof would go here.
    sorry

end johns_total_due_l151_151349


namespace number_of_blocks_l151_151527

theorem number_of_blocks (children_per_block : ℕ) (total_children : ℕ) (h1: children_per_block = 6) (h2: total_children = 54) : (total_children / children_per_block) = 9 :=
by {
  sorry
}

end number_of_blocks_l151_151527


namespace jenny_best_neighborhood_earnings_l151_151502

theorem jenny_best_neighborhood_earnings :
  let cost_per_box := 2
  let neighborhood_a_homes := 10
  let neighborhood_a_boxes_per_home := 2
  let neighborhood_b_homes := 5
  let neighborhood_b_boxes_per_home := 5
  let earnings_a := neighborhood_a_homes * neighborhood_a_boxes_per_home * cost_per_box
  let earnings_b := neighborhood_b_homes * neighborhood_b_boxes_per_home * cost_per_box
  max earnings_a earnings_b = 50
:= by
  sorry

end jenny_best_neighborhood_earnings_l151_151502


namespace magnitude_v_l151_151609

open Complex

theorem magnitude_v (u v : ℂ) (h1 : u * v = 20 - 15 * Complex.I) (h2 : Complex.abs u = 5) :
  Complex.abs v = 5 := by
  sorry

end magnitude_v_l151_151609


namespace range_of_m_l151_151372

theorem range_of_m (m : ℝ) : (∀ x : ℝ, ∀ y : ℝ, (2 ≤ x ∧ x ≤ 3) → (3 ≤ y ∧ y ≤ 6) → m * x^2 - x * y + y^2 ≥ 0) ↔ (m ≥ 0) :=
by
  sorry

end range_of_m_l151_151372


namespace intersection_empty_l151_151334

def setA : Set ℝ := { x | x^2 - 2 * x > 0 }
def setB : Set ℝ := { x | |x + 1| < 0 }

theorem intersection_empty : setA ∩ setB = ∅ :=
by
  sorry

end intersection_empty_l151_151334


namespace percent_of_x_is_z_l151_151206

theorem percent_of_x_is_z (x y z : ℝ) (h1 : 0.45 * z = 1.2 * y) (h2 : y = 0.75 * x) : z = 2 * x :=
by
  sorry

end percent_of_x_is_z_l151_151206


namespace find_f_neg_two_l151_151746

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_equation (x : ℝ) (hx : x ≠ 0) : 3 * f (1 / x) + (2 * f x) / x = x ^ 2

theorem find_f_neg_two : f (-2) = 67 / 20 :=
by
  sorry

end find_f_neg_two_l151_151746


namespace alpha_values_perpendicular_l151_151622

theorem alpha_values_perpendicular
  (α : ℝ)
  (h1 : α ∈ Set.Ico 0 (2 * Real.pi))
  (h2 : ∀ (x y : ℝ), x * Real.cos α - y - 1 = 0 → x + y * Real.sin α + 1 = 0 → false):
  α = Real.pi / 4 ∨ α = 5 * Real.pi / 4 :=
by
  sorry

end alpha_values_perpendicular_l151_151622


namespace gcd_117_182_evaluate_polynomial_l151_151647

-- Problem 1: Prove that GCD of 117 and 182 is 13
theorem gcd_117_182 : Int.gcd 117 182 = 13 := 
by
  sorry

-- Problem 2: Prove that evaluating the polynomial at x = -1 results in 12
noncomputable def f : ℤ → ℤ := λ x => 1 - 9 * x + 8 * x^2 - 4 * x^4 + 5 * x^5 + 3 * x^6

theorem evaluate_polynomial : f (-1) = 12 := 
by
  sorry

end gcd_117_182_evaluate_polynomial_l151_151647


namespace cost_equal_at_60_l151_151065

variable (x : ℝ)

def PlanA_cost (x : ℝ) : ℝ := 0.25 * x + 9
def PlanB_cost (x : ℝ) : ℝ := 0.40 * x

theorem cost_equal_at_60 : PlanA_cost x = PlanB_cost x → x = 60 :=
by
  intro h
  sorry

end cost_equal_at_60_l151_151065


namespace remainders_equal_if_difference_divisible_l151_151016

theorem remainders_equal_if_difference_divisible (a b k : ℤ) (h : k ∣ (a - b)) : 
  a % k = b % k :=
sorry

end remainders_equal_if_difference_divisible_l151_151016


namespace probability_red_white_red_l151_151929

-- Definitions and assumptions
def total_marbles := 10
def red_marbles := 4
def white_marbles := 6

def P_first_red : ℚ := red_marbles / total_marbles
def P_second_white_given_first_red : ℚ := white_marbles / (total_marbles - 1)
def P_third_red_given_first_red_and_second_white : ℚ := (red_marbles - 1) / (total_marbles - 2)

-- The target probability hypothesized
theorem probability_red_white_red :
  P_first_red * P_second_white_given_first_red * P_third_red_given_first_red_and_second_white = 1 / 10 :=
by
  sorry

end probability_red_white_red_l151_151929


namespace cost_of_each_new_shirt_l151_151678

theorem cost_of_each_new_shirt (pants_cost shorts_cost shirts_cost : ℕ)
  (pants_sold shorts_sold shirts_sold : ℕ) (money_left : ℕ) (new_shirts : ℕ)
  (h₁ : pants_cost = 5) (h₂ : shorts_cost = 3) (h₃ : shirts_cost = 4)
  (h₄ : pants_sold = 3) (h₅ : shorts_sold = 5) (h₆ : shirts_sold = 5)
  (h₇ : money_left = 30) (h₈ : new_shirts = 2) :
  (pants_cost * pants_sold + shorts_cost * shorts_sold + shirts_cost * shirts_sold - money_left) / new_shirts = 10 :=
by sorry

end cost_of_each_new_shirt_l151_151678


namespace impossible_network_of_triangles_l151_151810

-- Define the conditions of the problem, here we could define vertices and properties of the network
structure Vertex :=
(triangles_meeting : Nat)

def five_triangles_meeting (v : Vertex) : Prop :=
v.triangles_meeting = 5

-- The main theorem statement - it's impossible to cover the entire plane with such a network
theorem impossible_network_of_triangles :
  ¬ (∀ v : Vertex, five_triangles_meeting v) :=
sorry

end impossible_network_of_triangles_l151_151810


namespace problem_statement_l151_151907

theorem problem_statement (a : ℤ)
  (h : (2006 - a) * (2004 - a) = 2005) :
  (2006 - a) ^ 2 + (2004 - a) ^ 2 = 4014 :=
sorry

end problem_statement_l151_151907


namespace correct_calc_value_l151_151001

theorem correct_calc_value (x : ℕ) (h : 2 * (3 * x + 14) = 946) : 2 * (x / 3 + 14) = 130 := 
by
  sorry

end correct_calc_value_l151_151001


namespace one_third_of_7_times_9_l151_151118

theorem one_third_of_7_times_9 : (1 / 3) * (7 * 9) = 21 := by
  sorry

end one_third_of_7_times_9_l151_151118


namespace intersection_of_complement_l151_151476

open Set

variable (U : Set ℤ) (A B : Set ℤ)

def complement (U A : Set ℤ) : Set ℤ := U \ A

theorem intersection_of_complement (hU : U = {-1, 0, 1, 2, 3, 4})
  (hA : A = {1, 2, 3, 4}) (hB : B = {0, 2}) :
  (complement U A) ∩ B = {0} :=
by
  sorry

end intersection_of_complement_l151_151476


namespace scientific_notation_of_819000_l151_151255

theorem scientific_notation_of_819000 : (819000 : ℝ) = 8.19 * 10^5 :=
by
  sorry

end scientific_notation_of_819000_l151_151255


namespace division_problem_l151_151588

theorem division_problem : 250 / (15 + 13 * 3 - 4) = 5 := by
  sorry

end division_problem_l151_151588


namespace inequality_correct_transformation_l151_151818

-- Definitions of the conditions
variables (a b : ℝ)

-- The equivalent proof problem
theorem inequality_correct_transformation (h : a > b) : -a < -b :=
by sorry

end inequality_correct_transformation_l151_151818


namespace compare_abc_case1_compare_abc_case2_compare_abc_case3_l151_151670

variable (a : ℝ)
variable (b : ℝ := (1 / 2) * (a + 3 / a))
variable (c : ℝ := (1 / 2) * (b + 3 / b))

-- First condition: if \(a > \sqrt{3}\), then \(a > b > c\)
theorem compare_abc_case1 (h1 : a > 0) (h2 : a > Real.sqrt 3) : a > b ∧ b > c := sorry

-- Second condition: if \(a = \sqrt{3}\), then \(a = b = c\)
theorem compare_abc_case2 (h1 : a > 0) (h2 : a = Real.sqrt 3) : a = b ∧ b = c := sorry

-- Third condition: if \(0 < a < \sqrt{3}\), then \(a < c < b\)
theorem compare_abc_case3 (h1 : a > 0) (h2 : a < Real.sqrt 3) : a < c ∧ c < b := sorry

end compare_abc_case1_compare_abc_case2_compare_abc_case3_l151_151670


namespace problem_statement_l151_151006

theorem problem_statement (x y z : ℝ) :
    2 * x > y^2 + z^2 →
    2 * y > x^2 + z^2 →
    2 * z > y^2 + x^2 →
    x * y * z < 1 := by
  sorry

end problem_statement_l151_151006


namespace circle_through_two_points_on_y_axis_l151_151495

theorem circle_through_two_points_on_y_axis :
  ∃ (b : ℝ), (∀ (x y : ℝ), (x + 1)^2 + (y - 4)^2 = (x - 3)^2 + (y - 2)^2 → b = 1) ∧ 
  (∀ (x y : ℝ), (x - 0)^2 + (y - b)^2 = 10) := 
sorry

end circle_through_two_points_on_y_axis_l151_151495


namespace find_width_of_river_l151_151135

theorem find_width_of_river
    (total_distance : ℕ)
    (river_width : ℕ)
    (prob_find_item : ℚ)
    (h1 : total_distance = 500)
    (h2 : prob_find_item = 4/5)
    : river_width = 100 :=
by
    sorry

end find_width_of_river_l151_151135


namespace find_constant_c_l151_151110

theorem find_constant_c : ∃ (c : ℝ), (∀ n : ℤ, c * (n:ℝ)^2 ≤ 3600) ∧ (∀ n : ℤ, n ≤ 5) ∧ (c = 144) :=
by
  sorry

end find_constant_c_l151_151110


namespace expenses_of_5_yuan_l151_151059

-- Define the given condition: income of 5 yuan is +5 yuan
def income (x : Int) : Int := x

-- Define the opposite relationship between income and expenses
def expenses (x : Int) : Int := -income x

-- Proof statement to show that expenses of 5 yuan are -5 yuan, given the above definitions
theorem expenses_of_5_yuan : expenses 5 = -5 := by
  -- The proof is not provided here, so we use sorry to indicate its place
  sorry

end expenses_of_5_yuan_l151_151059


namespace solve_inequality_l151_151963

variable {x : ℝ}

theorem solve_inequality :
  (x - 8) / (x^2 - 4 * x + 13) ≥ 0 ↔ x ≥ 8 :=
by
  sorry

end solve_inequality_l151_151963


namespace equal_roots_condition_l151_151558

theorem equal_roots_condition (m : ℝ) :
  (m = 2 ∨ m = (9 + Real.sqrt 57) / 8 ∨ m = (9 - Real.sqrt 57) / 8) →
  ∃ a b c : ℝ, 
  (∀ x : ℝ, (a * x ^ 2 + b * x + c = 0) ↔
  (x * (x - 3) - (m + 2)) / ((x - 3) * (m - 2)) = x / m) ∧
  (b^2 - 4 * a * c = 0) :=
sorry

end equal_roots_condition_l151_151558


namespace nat_n_divisibility_cond_l151_151563

theorem nat_n_divisibility_cond (n : ℕ) : (n * 2^n + 1) % 3 = 0 ↔ (n % 3 = 1 ∨ n % 3 = 2) :=
by sorry

end nat_n_divisibility_cond_l151_151563


namespace max_sundays_in_84_days_l151_151796

-- Define constants
def days_in_week : ℕ := 7
def total_days : ℕ := 84

-- Theorem statement
theorem max_sundays_in_84_days : (total_days / days_in_week) = 12 :=
by sorry

end max_sundays_in_84_days_l151_151796


namespace robyn_packs_l151_151513

-- Define the problem conditions
def total_packs : ℕ := 76
def lucy_packs : ℕ := 29

-- Define the goal to be proven
theorem robyn_packs : total_packs - lucy_packs = 47 := 
by
  sorry

end robyn_packs_l151_151513


namespace find_initial_mean_l151_151116

/-- 
  The mean of 50 observations is M.
  One observation was wrongly taken as 23 but should have been 30.
  The corrected mean is 36.5.
  Prove that the initial mean M was 36.36.
-/
theorem find_initial_mean (M : ℝ) (h : 50 * 36.36 + 7 = 50 * 36.5) : 
  (500 * 36.36 - 7) = 1818 :=
sorry

end find_initial_mean_l151_151116


namespace difference_of_two_numbers_l151_151574

-- Definitions as per conditions
def L : ℕ := 1656
def S : ℕ := 273
def quotient : ℕ := 6
def remainder : ℕ := 15

-- Statement of the proof problem
theorem difference_of_two_numbers (h1 : L = 6 * S + 15) : L - S = 1383 :=
by sorry

end difference_of_two_numbers_l151_151574


namespace zeros_of_f_on_interval_l151_151578

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem zeros_of_f_on_interval : ∃ (S : Set ℝ), S ⊆ (Set.Ioo 0 1) ∧ S.Infinite ∧ ∀ x ∈ S, f x = 0 := by
  sorry

end zeros_of_f_on_interval_l151_151578


namespace max_marks_exam_l151_151412

theorem max_marks_exam (M : ℝ) 
  (h1 : 0.80 * M = 400) :
  M = 500 := 
by
  sorry

end max_marks_exam_l151_151412


namespace volume_of_prism_is_429_l151_151654

theorem volume_of_prism_is_429 (x y z : ℝ) (h1 : x * y = 56) (h2 : y * z = 57) (h3 : z * x = 58) : 
  x * y * z = 429 :=
by
  sorry

end volume_of_prism_is_429_l151_151654


namespace sum_first_n_terms_arithmetic_sequence_l151_151170

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n m : ℕ, a (m + 1) - a m = d

theorem sum_first_n_terms_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (a12 : a 12 = -8) (S9 : S 9 = -9) (h_arith : is_arithmetic_sequence a) :
  S 16 = -72 :=
sorry

end sum_first_n_terms_arithmetic_sequence_l151_151170


namespace circle_reflection_l151_151767

variable (x₀ y₀ : ℝ)

def reflect_over_line_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem circle_reflection 
  (h₁ : x₀ = 8)
  (h₂ : y₀ = -3) :
  reflect_over_line_y_eq_neg_x (x₀, y₀) = (3, -8) := by
  sorry

end circle_reflection_l151_151767


namespace union_set_A_set_B_l151_151844

def set_A : Set ℝ := { x | x^2 - 5 * x - 6 < 0 }
def set_B : Set ℝ := { x | -3 < x ∧ x < 2 }
def set_union (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∨ x ∈ B }

theorem union_set_A_set_B : set_union set_A set_B = { x | -3 < x ∧ x < 6 } := 
by sorry

end union_set_A_set_B_l151_151844


namespace smallest_integer_n_l151_151230

theorem smallest_integer_n (n : ℕ) (h₁ : 50 ∣ n^2) (h₂ : 294 ∣ n^3) : n = 210 :=
sorry

end smallest_integer_n_l151_151230


namespace find_a2014_l151_151025

open Nat

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 0 ∧
  (∀ n, a (n + 1) = (a n - 2) / (5 * a n / 4 - 2))

theorem find_a2014 (a : ℕ → ℚ) (h : seq a) : a 2014 = 1 :=
by
  sorry

end find_a2014_l151_151025


namespace general_formula_sum_first_n_terms_l151_151159

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

axiom a_initial : a 1 = 1
axiom a_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = 3 * a n * (1 + 1 / n)

theorem general_formula : ∀ n : ℕ, n > 0 → a n = n * 3^(n - 1) :=
by
  sorry

theorem sum_first_n_terms : ∀ n : ℕ, S n = (2 * n - 1) * 3^n + 1 / 4 :=
by
  sorry

end general_formula_sum_first_n_terms_l151_151159


namespace show_spiders_l151_151669

noncomputable def spiders_found (ants : ℕ) (ladybugs_initial : ℕ) (ladybugs_fly_away : ℕ) (total_insects_remaining : ℕ) : ℕ :=
  let ladybugs_remaining := ladybugs_initial - ladybugs_fly_away
  let insects_observed := ants + ladybugs_remaining
  total_insects_remaining - insects_observed

theorem show_spiders
  (ants : ℕ := 12)
  (ladybugs_initial : ℕ := 8)
  (ladybugs_fly_away : ℕ := 2)
  (total_insects_remaining : ℕ := 21) :
  spiders_found ants ladybugs_initial ladybugs_fly_away total_insects_remaining = 3 := by
  sorry

end show_spiders_l151_151669


namespace interval_length_l151_151455

theorem interval_length (x : ℝ) :
  (1/x > 1/2) ∧ (Real.sin x > 1/2) → (2 - Real.pi / 6 = 1.48) :=
by
  sorry

end interval_length_l151_151455


namespace max_load_truck_l151_151957

theorem max_load_truck (bag_weight : ℕ) (num_bags : ℕ) (remaining_load : ℕ) 
  (h1 : bag_weight = 8) (h2 : num_bags = 100) (h3 : remaining_load = 100) : 
  bag_weight * num_bags + remaining_load = 900 :=
by
  -- We leave the proof step intentionally, as per instructions.
  sorry

end max_load_truck_l151_151957


namespace white_bread_served_l151_151414

theorem white_bread_served (total_bread : ℝ) (wheat_bread : ℝ) (white_bread : ℝ) 
  (h1 : total_bread = 0.9) (h2 : wheat_bread = 0.5) : white_bread = 0.4 :=
by
  sorry

end white_bread_served_l151_151414


namespace douglas_votes_in_county_y_l151_151837

variable (V : ℝ) -- Number of voters in County Y
variable (A B : ℝ) -- Votes won by Douglas in County X and County Y respectively

-- Conditions
axiom h1 : A = 0.74 * 2 * V
axiom h2 : A + B = 0.66 * 3 * V
axiom ratio : (2 * V) / V = 2

-- Proof Statement
theorem douglas_votes_in_county_y :
  (B / V) * 100 = 50 := by
sorry

end douglas_votes_in_county_y_l151_151837


namespace count_lines_in_2008_cube_l151_151657

def num_lines_through_centers_of_unit_cubes (n : ℕ) : ℕ :=
  n * n * 3 + n * 2 * 3 + 4

theorem count_lines_in_2008_cube :
  num_lines_through_centers_of_unit_cubes 2008 = 12115300 :=
by
  -- The actual proof would go here
  sorry

end count_lines_in_2008_cube_l151_151657


namespace roots_of_equation_l151_151020

theorem roots_of_equation (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 2 * x1 + 2 * |x1 + 1| = a) ∧ (x2^2 + 2 * x2 + 2 * |x2 + 1| = a)) ↔ a > -1 := 
by
  sorry

end roots_of_equation_l151_151020


namespace swim_time_CBA_l151_151661

theorem swim_time_CBA (d t_down t_still t_upstream: ℝ) 
  (h1 : d = 1) 
  (h2 : t_down = 1 / (6 / 5))
  (h3 : t_still = 1)
  (h4 : t_upstream = (4 / 5) / 2)
  (total_time_down : (t_down + t_still) = 1)
  (total_time_up : (t_still + t_down) = 2) :
  (t_upstream * (d - (d / 5))) / 2 = 5 / 2 :=
by sorry

end swim_time_CBA_l151_151661


namespace eccentricity_of_ellipse_l151_151707

theorem eccentricity_of_ellipse 
  (a b c m n : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0) 
  (h3 : m > 0) 
  (h4 : n > 0) 
  (ellipse_eq : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 -> (m^2 + n^2 > x^2 + y^2))
  (hyperbola_eq : ∀ x y : ℝ, x^2 / m^2 - y^2 / n^2 = 1 -> (m^2 + n^2 > x^2 - y^2))
  (same_foci: ∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 <= 1 → x^2 / m^2 - y^2 / n^2 = 1)
  (geometric_mean : c^2 = a * m)
  (arithmetic_mean : 2 * n^2 = 2 * m^2 + c^2) : 
  (c / a = 1 / 2) :=
sorry

end eccentricity_of_ellipse_l151_151707


namespace set_representation_equiv_l151_151041

open Nat

theorem set_representation_equiv :
  {x : ℕ | (0 < x) ∧ (x - 3 < 2)} = {1, 2, 3, 4} :=
by
  sorry

end set_representation_equiv_l151_151041


namespace inequality_proof_l151_151896

theorem inequality_proof (x y : ℝ) (h : |x - 2 * y| = 5) : x^2 + y^2 ≥ 5 := 
  sorry

end inequality_proof_l151_151896


namespace worker_efficiency_l151_151205

theorem worker_efficiency (W_p W_q : ℚ) 
  (h1 : W_p = 1 / 24) 
  (h2 : W_p + W_q = 1 / 14) :
  (W_p - W_q) / W_q * 100 = 40 :=
by
  sorry

end worker_efficiency_l151_151205


namespace product_of_symmetric_complex_numbers_l151_151798

def z1 : ℂ := 1 + 2 * Complex.I

def z2 : ℂ := -1 + 2 * Complex.I

theorem product_of_symmetric_complex_numbers :
  z1 * z2 = -5 :=
by 
  sorry

end product_of_symmetric_complex_numbers_l151_151798


namespace quadratic_two_distinct_real_roots_l151_151847

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - x₁ - k^2 = 0) ∧ (x₂^2 - x₂ - k^2 = 0) :=
by
  -- The proof is omitted as requested.
  sorry

end quadratic_two_distinct_real_roots_l151_151847


namespace trig_identity_l151_151215

-- Define the given condition
def tan_half (α : ℝ) : Prop := Real.tan (α / 2) = 2

-- The main statement we need to prove
theorem trig_identity (α : ℝ) (h : tan_half α) : (1 + Real.cos α) / (Real.sin α) = 1 / 2 :=
  by
  sorry

end trig_identity_l151_151215


namespace find_fraction_l151_151260

theorem find_fraction (a b : ℝ) (h : a ≠ b) (h_eq : a / b + (a + 20 * b) / (b + 20 * a) = 3) : a / b = 0.33 :=
sorry

end find_fraction_l151_151260


namespace cash_still_missing_l151_151487

theorem cash_still_missing (c : ℝ) (h : c > 0) :
  (1 : ℝ) - (8 / 9) = (1 / 9 : ℝ) :=
by
  sorry

end cash_still_missing_l151_151487


namespace remainder_example_l151_151682

def P (x : ℝ) := 8 * x^3 - 20 * x^2 + 28 * x - 26
def D (x : ℝ) := 4 * x - 8

theorem remainder_example : P 2 = 14 :=
by
  sorry

end remainder_example_l151_151682


namespace savings_from_discount_l151_151971

-- Define the initial price
def initial_price : ℝ := 475.00

-- Define the discounted price
def discounted_price : ℝ := 199.00

-- The theorem to prove the savings amount
theorem savings_from_discount : initial_price - discounted_price = 276.00 :=
by 
  -- This is where the actual proof would go
  sorry

end savings_from_discount_l151_151971


namespace incorrect_option_D_l151_151092

theorem incorrect_option_D (x y : ℝ) : y = (x - 2) ^ 2 + 1 → ¬ (∀ (x : ℝ), x < 2 → y < (x - 1) ^ 2 + 1) :=
by
  intro h
  sorry

end incorrect_option_D_l151_151092


namespace relationship_between_length_and_width_l151_151211

theorem relationship_between_length_and_width 
  (x y : ℝ) (h : 2 * (x + y) = 20) : y = 10 - x := 
by
  sorry

end relationship_between_length_and_width_l151_151211


namespace attendants_both_tools_l151_151603

theorem attendants_both_tools (pencil_users pen_users only_one_type total_attendants both_types : ℕ)
  (h1 : pencil_users = 25) 
  (h2 : pen_users = 15) 
  (h3 : only_one_type = 20) 
  (h4 : total_attendants = only_one_type + both_types) 
  (h5 : total_attendants = pencil_users + pen_users - both_types) 
  : both_types = 10 :=
by
  -- Fill in the proof sub-steps here if needed
  sorry

end attendants_both_tools_l151_151603


namespace find_T_shirts_l151_151596

variable (T S : ℕ)

-- Given conditions
def condition1 : S = 2 * T := sorry
def condition2 : T + S - (T + 3) = 15 := sorry

-- Prove that number of T-shirts T Norma left in the washer is 9
theorem find_T_shirts (h1 : S = 2 * T) (h2 : T + S - (T + 3) = 15) : T = 9 :=
  by
    sorry

end find_T_shirts_l151_151596


namespace columbia_distinct_arrangements_l151_151281

theorem columbia_distinct_arrangements : 
  let total_letters := 8
  let repeat_I := 2
  let repeat_U := 2
  Nat.factorial total_letters / (Nat.factorial repeat_I * Nat.factorial repeat_U) = 90720 := by
  sorry

end columbia_distinct_arrangements_l151_151281


namespace true_false_question_count_l151_151082

theorem true_false_question_count (n : ℕ) (h : (1 / 3) * (1 / 2)^n = 1 / 12) : n = 2 := by
  sorry

end true_false_question_count_l151_151082


namespace find_y_in_terms_of_x_l151_151747

variable (x y : ℝ)

theorem find_y_in_terms_of_x (hx : x = 5) (hy : y = -4) (hp : ∃ k, y = k * (x - 3)) :
  y = -2 * x + 6 := by
sorry

end find_y_in_terms_of_x_l151_151747


namespace tan_alpha_eq_neg_one_third_l151_151884

open Real

theorem tan_alpha_eq_neg_one_third
  (h : cos (π / 4 - α) / cos (π / 4 + α) = 1 / 2) :
  tan α = -1 / 3 :=
sorry

end tan_alpha_eq_neg_one_third_l151_151884


namespace joe_collected_cards_l151_151153

theorem joe_collected_cards (boxes : ℕ) (cards_per_box : ℕ) (filled_boxes : boxes = 11) (max_cards_per_box : cards_per_box = 8) : boxes * cards_per_box = 88 := by
  sorry

end joe_collected_cards_l151_151153


namespace quadratic_root_range_specific_m_value_l151_151917

theorem quadratic_root_range (m : ℝ) : 
  ∃ x1 x2 : ℝ, x1^2 - 2 * (1 - m) * x1 + m^2 = 0 ∧ x2^2 - 2 * (1 - m) * x2 + m^2 = 0 ↔ m ≤ 1/2 :=
by
  sorry

theorem specific_m_value (m : ℝ) (x1 x2 : ℝ) (h1 : x1^2 - 2 * (1 - m) * x1 + m^2 = 0)
  (h2 : x2^2 - 2 * (1 - m) * x2 + m^2 = 0) (h3 : x1^2 + 12 * m + x2^2 = 10) : 
  m = -3 :=
by
  sorry

end quadratic_root_range_specific_m_value_l151_151917


namespace no_valid_weights_l151_151584

theorem no_valid_weights (w_1 w_2 w_3 w_4 : ℝ) : 
  w_1 + w_2 + w_3 = 100 → w_1 + w_2 + w_4 = 101 → w_2 + w_3 + w_4 = 102 → 
  w_1 < 90 → w_2 < 90 → w_3 < 90 → w_4 < 90 → False :=
by 
  intros h1 h2 h3 hl1 hl2 hl3 hl4
  sorry

end no_valid_weights_l151_151584


namespace complement_union_M_N_l151_151984

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {3, 4}
def complement_U (s : Set ℕ) : Set ℕ := U \ s

theorem complement_union_M_N : complement_U (M ∪ N) = {5} := by
  sorry

end complement_union_M_N_l151_151984


namespace fraction_calculation_l151_151951

theorem fraction_calculation : 
  ( (1 / 5 + 1 / 7) / (3 / 8 + 2 / 9) ) = (864 / 1505) := 
by
  sorry

end fraction_calculation_l151_151951


namespace gcd_840_1764_gcd_459_357_l151_151802

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := sorry

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := sorry

end gcd_840_1764_gcd_459_357_l151_151802


namespace sum_series_equals_half_l151_151090

theorem sum_series_equals_half :
  ∑' n, 1 / (n * (n+1) * (n+2)) = 1 / 2 :=
sorry

end sum_series_equals_half_l151_151090


namespace odd_function_value_at_neg_two_l151_151005

noncomputable def f : ℝ → ℝ :=
  λ x => if x > 0 then 2 * x - 3 else - (2 * (-x) - 3)

theorem odd_function_value_at_neg_two :
  (∀ x, f (-x) = -f x) → f (-2) = -1 :=
by
  intro odd_f
  sorry

end odd_function_value_at_neg_two_l151_151005


namespace complex_sum_cubics_eq_zero_l151_151475

-- Define the hypothesis: omega is a nonreal root of x^3 = 1
def is_nonreal_root_of_cubic (ω : ℂ) : Prop :=
  ω^3 = 1 ∧ ω ≠ 1

-- Now state the theorem to prove the expression evaluates to 0
theorem complex_sum_cubics_eq_zero (ω : ℂ) (h : is_nonreal_root_of_cubic ω) :
  (2 - 2*ω + 2*ω^2)^3 + (2 + 2*ω - 2*ω^2)^3 = 0 :=
by
  -- This is where the proof would go. 
  sorry

end complex_sum_cubics_eq_zero_l151_151475


namespace mod_abc_eq_zero_l151_151194

open Nat

theorem mod_abc_eq_zero
    (a b c : ℕ)
    (h1 : (a + 2 * b + 3 * c) % 9 = 1)
    (h2 : (2 * a + 3 * b + c) % 9 = 2)
    (h3 : (3 * a + b + 2 * c) % 9 = 3) :
    (a * b * c) % 9 = 0 := by
  sorry

end mod_abc_eq_zero_l151_151194


namespace exponential_comparison_l151_151032

theorem exponential_comparison
  (a : ℕ := 3^55)
  (b : ℕ := 4^44)
  (c : ℕ := 5^33) :
  c < a ∧ a < b :=
by
  sorry

end exponential_comparison_l151_151032


namespace total_bill_l151_151808

theorem total_bill (n : ℝ) (h : 9 * (n / 10 + 3) = n) : n = 270 := 
sorry

end total_bill_l151_151808


namespace prime_quadruples_l151_151911

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_quadruples {p₁ p₂ p₃ p₄ : ℕ} (prime_p₁ : is_prime p₁) (prime_p₂ : is_prime p₂) (prime_p₃ : is_prime p₃) (prime_p₄ : is_prime p₄)
  (h1 : p₁ < p₂) (h2 : p₂ < p₃) (h3 : p₃ < p₄) (eq_condition : p₁ * p₂ + p₂ * p₃ + p₃ * p₄ + p₄ * p₁ = 882) :
  (p₁ = 2 ∧ p₂ = 5 ∧ p₃ = 19 ∧ p₄ = 37) ∨
  (p₁ = 2 ∧ p₂ = 11 ∧ p₃ = 19 ∧ p₄ = 31) ∨
  (p₁ = 2 ∧ p₂ = 13 ∧ p₃ = 19 ∧ p₄ = 29) :=
sorry

end prime_quadruples_l151_151911


namespace largest_divisor_of_n_l151_151391

theorem largest_divisor_of_n 
  (n : ℕ) (h_pos : n > 0) (h_div : 72 ∣ n^2) : 
  ∃ v : ℕ, v = 12 ∧ v ∣ n :=
by
  use 12
  sorry

end largest_divisor_of_n_l151_151391


namespace color_films_count_l151_151519

variables (x y C : ℕ)
variables (h1 : 0.9615384615384615 = (C : ℝ) / ((2 * (y : ℝ) / 5) + (C : ℝ)))

theorem color_films_count (x y : ℕ) (C : ℕ) (h1 : 0.9615384615384615 = (C : ℝ) / ((2 * (y : ℝ) / 5) + (C : ℝ))) :
  C = 10 * y :=
sorry

end color_films_count_l151_151519


namespace david_money_left_l151_151566

theorem david_money_left (S : ℤ) (h1 : S - 800 = 1800 - S) : 1800 - S = 500 :=
by
  sorry

end david_money_left_l151_151566


namespace football_games_this_year_l151_151918

theorem football_games_this_year 
  (total_games : ℕ) 
  (games_last_year : ℕ) 
  (games_this_year : ℕ) 
  (h1 : total_games = 9) 
  (h2 : games_last_year = 5) 
  (h3 : total_games = games_last_year + games_this_year) : 
  games_this_year = 4 := 
sorry

end football_games_this_year_l151_151918


namespace identify_quadratic_equation_l151_151586

theorem identify_quadratic_equation :
  (∀ b c d : Prop, ∀ (f : ℕ → Prop), f 0 → ¬ f 1 → ¬ f 2 → ¬ f 3 → b ∧ ¬ c ∧ ¬ d) →
  (∀ x y : ℝ,  (x^2 + 2 = 0) = (b ∧ ¬ b → c ∧ ¬ c → d ∧ ¬ d)) :=
by
  intros;
  sorry

end identify_quadratic_equation_l151_151586


namespace minimize_distance_l151_151388

-- Definitions of points and distances
structure Point where
  x : ℝ
  y : ℝ

def distanceSquared (P Q : Point) : ℝ :=
  (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Condition points A, B, and C
def A := Point.mk 7 3
def B := Point.mk 3 0

-- Mathematical problem: Find the value of k that minimizes the sum of distances squared
theorem minimize_distance : ∃ k : ℝ, ∀ k', 
  (distanceSquared A (Point.mk 0 k) + distanceSquared B (Point.mk 0 k) ≤ 
   distanceSquared A (Point.mk 0 k') + distanceSquared B (Point.mk 0 k')) → 
  k = 3 / 2 :=
by
  sorry

end minimize_distance_l151_151388


namespace outdoor_section_width_l151_151221

theorem outdoor_section_width (Length Area Width : ℝ) (h1 : Length = 6) (h2 : Area = 24) : Width = 4 :=
by
  -- We'll use "?" to represent the parts that need to be inferred by the proof assistant. 
  sorry

end outdoor_section_width_l151_151221


namespace triangle_constructibility_l151_151113

noncomputable def constructible_triangle (a b w_c : ℝ) : Prop :=
  (2 * a * b) / (a + b) > w_c

theorem triangle_constructibility {a b w_c : ℝ} (h : (a > 0) ∧ (b > 0) ∧ (w_c > 0)) :
  constructible_triangle a b w_c ↔ True :=
by
  sorry

end triangle_constructibility_l151_151113


namespace line_passes_through_vertex_count_l151_151240

theorem line_passes_through_vertex_count :
  (∃ a : ℝ, ∀ (x : ℝ), x = 0 → (x + a = a^2)) ↔ (∀ a : ℝ, (a = 0 ∨ a = 1)) :=
by
  sorry

end line_passes_through_vertex_count_l151_151240


namespace kara_water_intake_l151_151619

-- Definitions based on the conditions
def daily_doses := 3
def week1_days := 7
def week2_days := 7
def forgot_doses_day := 2
def total_weeks := 2
def total_water := 160

-- The statement to prove
theorem kara_water_intake :
  let total_doses := (daily_doses * week1_days) + (daily_doses * week2_days - forgot_doses_day)
  ∃ (water_per_dose : ℕ), water_per_dose * total_doses = total_water ∧ water_per_dose = 4 :=
by
  sorry

end kara_water_intake_l151_151619


namespace solution_set_l151_151839

noncomputable def f : ℝ → ℝ := sorry

axiom deriv_f_pos (x : ℝ) : deriv f x > 1 - f x
axiom f_at_zero : f 0 = 3

theorem solution_set (x : ℝ) : e^x * f x > e^x + 2 ↔ x > 0 :=
by sorry

end solution_set_l151_151839


namespace system_of_inequalities_l151_151466

theorem system_of_inequalities :
  ∃ (a b : ℤ), 
  (11 > 2 * a - b) ∧ 
  (25 > 2 * b - a) ∧ 
  (42 < 3 * b - a) ∧ 
  (46 < 2 * a + b) ∧ 
  (a = 14) ∧ 
  (b = 19) := 
sorry

end system_of_inequalities_l151_151466


namespace problem1_problem2_l151_151564

-- Theorem 1: Given a^2 - b^2 = 1940:
theorem problem1 
  (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_unit_digit : a^5 % 10 = b^5 % 10) : 
  a^2 - b^2 = 1940 → 
  (a = 102 ∧ b = 92) := 
by 
  sorry

-- Theorem 2: Given a^2 - b^2 = 1920:
theorem problem2 
  (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
  (h_unit_digit : a^5 % 10 = b^5 % 10) : 
  a^2 - b^2 = 1920 → 
  (a = 101 ∧ b = 91) ∨ 
  (a = 58 ∧ b = 38) ∨ 
  (a = 47 ∧ b = 17) ∨ 
  (a = 44 ∧ b = 4) := 
by 
  sorry

end problem1_problem2_l151_151564


namespace quadratic_decreasing_l151_151643

theorem quadratic_decreasing (a : ℝ) (h : ∀ x1 x2 : ℝ, x1 ≤ x2 → x2 ≤ 4 → (x1^2 + 4*a*x1 - 2) ≥ (x2^2 + 4*a*x2 - 2)) : a ≤ -2 := 
by
  sorry

end quadratic_decreasing_l151_151643


namespace compute_expression_l151_151497

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 := by
  sorry

end compute_expression_l151_151497


namespace probability_of_red_buttons_l151_151948

noncomputable def initialJarA : ℕ := 16 -- total buttons in Jar A (6 red, 10 blue)
noncomputable def initialRedA : ℕ := 6 -- initial red buttons in Jar A
noncomputable def initialBlueA : ℕ := 10 -- initial blue buttons in Jar A

noncomputable def initialJarB : ℕ := 5 -- total buttons in Jar B (2 red, 3 blue)
noncomputable def initialRedB : ℕ := 2 -- initial red buttons in Jar B
noncomputable def initialBlueB : ℕ := 3 -- initial blue buttons in Jar B

noncomputable def transferRed : ℕ := 3
noncomputable def transferBlue : ℕ := 3

noncomputable def finalRedA : ℕ := initialRedA - transferRed
noncomputable def finalBlueA : ℕ := initialBlueA - transferBlue

noncomputable def finalRedB : ℕ := initialRedB + transferRed
noncomputable def finalBlueB : ℕ := initialBlueB + transferBlue

noncomputable def remainingJarA : ℕ := finalRedA + finalBlueA
noncomputable def finalJarB : ℕ := finalRedB + finalBlueB

noncomputable def probRedA : ℚ := finalRedA / remainingJarA
noncomputable def probRedB : ℚ := finalRedB / finalJarB

noncomputable def combinedProb : ℚ := probRedA * probRedB

theorem probability_of_red_buttons :
  combinedProb = 3 / 22 := sorry

end probability_of_red_buttons_l151_151948


namespace circumscribed_sphere_eqn_l151_151256

-- Define vertices of the tetrahedron
variables {A_1 A_2 A_3 A_4 : Point}

-- Define barycentric coordinates
variables {x_1 x_2 x_3 x_4 : ℝ}

-- Define edge lengths
variables {a_12 a_13 a_14 a_23 a_24 a_34: ℝ}

-- Define the equation of the circumscribed sphere in barycentric coordinates
theorem circumscribed_sphere_eqn (h1 : A_1 ≠ A_2) (h2 : A_1 ≠ A_3) (h3 : A_1 ≠ A_4)
                                 (h4 : A_2 ≠ A_3) (h5 : A_2 ≠ A_4) (h6 : A_3 ≠ A_4) :
    (x_1 * x_2 * a_12^2 + x_1 * x_3 * a_13^2 + x_1 * x_4 * a_14^2 +
     x_2 * x_3 * a_23^2 + x_2 * x_4 * a_24^2 + x_3 * x_4 * a_34^2) = 0 :=
 sorry

end circumscribed_sphere_eqn_l151_151256


namespace mrs_sheridan_initial_cats_l151_151677

def cats_initial (cats_given_away : ℕ) (cats_left : ℕ) : ℕ :=
  cats_given_away + cats_left

theorem mrs_sheridan_initial_cats : cats_initial 14 3 = 17 :=
by
  sorry

end mrs_sheridan_initial_cats_l151_151677


namespace mod_sum_correct_l151_151160

theorem mod_sum_correct (a b c : ℕ) (ha : a < 7) (hb : b < 7) (hc : c < 7)
    (h1 : a * b * c ≡ 1 [MOD 7])
    (h2 : 5 * c ≡ 2 [MOD 7])
    (h3 : 6 * b ≡ 3 + b [MOD 7]) :
    (a + b + c) % 7 = 4 := sorry

end mod_sum_correct_l151_151160


namespace max_remainder_is_8_l151_151468

theorem max_remainder_is_8 (d q r : ℕ) (h1 : d = 9) (h2 : q = 6) (h3 : r < d) : 
  r ≤ (d - 1) :=
by 
  sorry

end max_remainder_is_8_l151_151468


namespace arith_seq_sum_of_terms_l151_151470

theorem arith_seq_sum_of_terms 
  (a : ℕ → ℝ) (d : ℝ) 
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_pos_diff : 0 < d) 
  (h_first_three_sum : a 0 + a 1 + a 2 = 15) 
  (h_first_three_prod : a 0 * a 1 * a 2 = 80) : 
  a 10 + a 11 + a 12 = 105 := sorry

end arith_seq_sum_of_terms_l151_151470


namespace arithmetic_sequence_x_values_l151_151359

theorem arithmetic_sequence_x_values {x : ℝ} (h_nonzero : x ≠ 0) (h_arith_seq : ∃ (k : ℤ), x - k = 1/2 ∧ x + 1 - (k + 1) = (k + 1) - 1/2) (h_lt_four : x < 4) :
  x = 0.5 ∨ x = 1.5 ∨ x = 2.5 ∨ x = 3.5 :=
by
  sorry

end arithmetic_sequence_x_values_l151_151359


namespace sam_new_books_not_signed_l151_151045

noncomputable def num_books_adventure := 13
noncomputable def num_books_mystery := 17
noncomputable def num_books_scifi := 25
noncomputable def num_books_nonfiction := 10
noncomputable def num_books_comics := 5
noncomputable def num_books_total := num_books_adventure + num_books_mystery + num_books_scifi + num_books_nonfiction + num_books_comics

noncomputable def num_books_used := 42
noncomputable def num_books_signed := 10
noncomputable def num_books_borrowed := 3
noncomputable def num_books_lost := 4

noncomputable def num_books_new := num_books_total - num_books_used
noncomputable def num_books_new_not_signed := num_books_new - num_books_signed
noncomputable def num_books_final := num_books_new_not_signed - num_books_lost

theorem sam_new_books_not_signed : num_books_final = 14 :=
by
  sorry

end sam_new_books_not_signed_l151_151045


namespace son_l151_151515

theorem son's_age (S F : ℕ) (h₁ : F = 7 * (S - 8)) (h₂ : F / 4 = 14) : S = 16 :=
by {
  sorry
}

end son_l151_151515


namespace proposition_p_is_false_iff_l151_151044

def f (x : ℝ) : ℝ := abs (x - 2) + abs (x + 3)

def p (a : ℝ) : Prop := ∃ x : ℝ, f x < a

theorem proposition_p_is_false_iff (a : ℝ) : (¬p a) ↔ (a < 5) :=
by sorry

end proposition_p_is_false_iff_l151_151044


namespace count_squares_within_region_l151_151060

noncomputable def countSquares : Nat := sorry

theorem count_squares_within_region :
  countSquares = 45 :=
sorry

end count_squares_within_region_l151_151060


namespace smallest_N_l151_151096

theorem smallest_N (l m n N : ℕ) (hl : l > 1) (hm : m > 1) (hn : n > 1) :
  (l - 1) * (m - 1) * (n - 1) = 231 → l * m * n = N → N = 384 :=
sorry

end smallest_N_l151_151096


namespace maximum_ab_value_l151_151975

noncomputable def ab_max (a b : ℝ) : ℝ :=
  if a > 0 then 2 * a * a - a * a * Real.log a else 0

theorem maximum_ab_value : ∀ (a b : ℝ), (∀ (x : ℝ), (Real.exp x - a * x + a) ≥ b) →
   ab_max a b ≤ if a = Real.exp (3 / 2) then (Real.exp 3) / 2 else sorry :=
by
  intros a b h
  sorry

end maximum_ab_value_l151_151975


namespace parabola_translation_correct_l151_151410

noncomputable def translate_parabola (x y : ℝ) (h : y = -2 * x^2 - 4 * x - 6) : Prop :=
  let x' := x - 1
  let y' := y + 3
  y' = -2 * x'^2 - 1

theorem parabola_translation_correct (x y : ℝ) (h : y = -2 * x^2 - 4 * x - 6) :
  translate_parabola x y h :=
sorry

end parabola_translation_correct_l151_151410


namespace fraction_of_alvin_age_l151_151544

variable (A E F : ℚ)

-- Conditions
def edwin_older_by_six : Prop := E = A + 6
def total_age : Prop := A + E = 30.99999999
def age_relation_in_two_years : Prop := E + 2 = F * (A + 2) + 20

-- Statement to prove
theorem fraction_of_alvin_age
  (h1 : edwin_older_by_six A E)
  (h2 : total_age A E)
  (h3 : age_relation_in_two_years A E F) :
  F = 1 / 29 :=
sorry

end fraction_of_alvin_age_l151_151544


namespace how_many_times_faster_l151_151777

theorem how_many_times_faster (A B : ℝ) (h1 : A = 1 / 32) (h2 : A + B = 1 / 24) : A / B = 3 := by
  sorry

end how_many_times_faster_l151_151777


namespace g_675_eq_42_l151_151331

-- Define the function g on positive integers
def g : ℕ → ℕ := sorry

-- State the conditions
axiom g_multiplicative : ∀ (x y : ℕ), g (x * y) = g x + g y
axiom g_15 : g 15 = 18
axiom g_45 : g 45 = 24

-- The theorem we want to prove
theorem g_675_eq_42 : g 675 = 42 := 
by 
  sorry

end g_675_eq_42_l151_151331


namespace triangle_in_and_circumcircle_radius_l151_151018

noncomputable def radius_of_incircle (AC : ℝ) (BC : ℝ) (AB : ℝ) (Area : ℝ) (s : ℝ) : ℝ :=
  Area / s

noncomputable def radius_of_circumcircle (AB : ℝ) : ℝ :=
  AB / 2

theorem triangle_in_and_circumcircle_radius :
  ∀ (A B C : ℝ × ℝ) (AC : ℝ) (BC : ℝ) (AB : ℝ)
    (AngleA : ℝ) (AngleC : ℝ),
  AngleC = 90 ∧ AngleA = 60 ∧ AC = 6 ∧
  BC = AC * Real.sqrt 3 ∧ AB = 2 * AC
  → radius_of_incircle AC BC AB (18 * Real.sqrt 3) ((AC + BC + AB) / 2) = 6 * (Real.sqrt 3 - 1) / 13 ∧
    radius_of_circumcircle AB = 6 := by
  intros A B C AC BC AB AngleA AngleC h
  sorry

end triangle_in_and_circumcircle_radius_l151_151018


namespace find_value_l151_151551

variable (x y a c : ℝ)

-- Conditions
def condition1 : Prop := x * y = 2 * c
def condition2 : Prop := (1 / x ^ 2) + (1 / y ^ 2) = 3 * a

-- Proof statement
theorem find_value : condition1 x y c ∧ condition2 x y a ↔ (x + y) ^ 2 = 12 * a * c ^ 2 + 4 * c := 
by 
  -- Placeholder for the actual proof
  sorry

end find_value_l151_151551


namespace distance_between_houses_l151_151384

theorem distance_between_houses
  (alice_speed : ℕ) (bob_speed : ℕ) (alice_distance : ℕ) 
  (alice_walk_time : ℕ) (bob_walk_time : ℕ)
  (alice_start : ℕ) (bob_start : ℕ)
  (bob_start_after_alice : bob_start = alice_start + 1)
  (alice_speed_eq : alice_speed = 5)
  (bob_speed_eq : bob_speed = 4)
  (alice_distance_eq : alice_distance = 25)
  (alice_walk_time_eq : alice_walk_time = alice_distance / alice_speed)
  (bob_walk_time_eq : bob_walk_time = alice_walk_time - 1)
  (bob_distance_eq : bob_walk_time = bob_walk_time * bob_speed)
  (total_distance : ℕ)
  (total_distance_eq : total_distance = alice_distance + bob_distance) :
  total_distance = 41 :=
by sorry

end distance_between_houses_l151_151384


namespace length_AD_l151_151718

open Real

-- Define the properties of the quadrilateral
variable (A B C D: Point)
variable (angle_ABC angle_BCD: ℝ)
variable (AB BC CD: ℝ)

-- Given conditions
axiom angle_ABC_eq_135 : angle_ABC = 135 * π / 180
axiom angle_BCD_eq_120 : angle_BCD = 120 * π / 180
axiom AB_eq_sqrt_6 : AB = sqrt 6
axiom BC_eq_5_minus_sqrt_3 : BC = 5 - sqrt 3
axiom CD_eq_6 : CD = 6

-- The theorem to prove
theorem length_AD {AD : ℝ} (h : True) :
  AD = 2 * sqrt 19 :=
sorry

end length_AD_l151_151718


namespace factor_diff_of_squares_l151_151292

-- Define the expression t^2 - 49 and show it is factored as (t - 7)(t + 7)
theorem factor_diff_of_squares (t : ℝ) : t^2 - 49 = (t - 7) * (t + 7) := by
  sorry

end factor_diff_of_squares_l151_151292


namespace money_left_in_wallet_l151_151333

def olivia_initial_money : ℕ := 54
def olivia_spent_money : ℕ := 25

theorem money_left_in_wallet : olivia_initial_money - olivia_spent_money = 29 :=
by
  sorry

end money_left_in_wallet_l151_151333


namespace second_fish_length_l151_151393

-- Defining the conditions
def first_fish_length : ℝ := 0.3
def length_difference : ℝ := 0.1

-- Proof statement
theorem second_fish_length : ∀ (second_fish : ℝ), first_fish_length = second_fish + length_difference → second_fish = 0.2 :=
by 
  intro second_fish
  intro h
  sorry

end second_fish_length_l151_151393


namespace jaime_can_buy_five_apples_l151_151154

theorem jaime_can_buy_five_apples :
  ∀ (L M : ℝ),
  (L = M / 2 + 1 / 2) →
  (M / 3 = L / 4 + 1 / 2) →
  (15 / M = 5) :=
by
  intros L M h1 h2
  sorry

end jaime_can_buy_five_apples_l151_151154


namespace probability_king_then_queen_l151_151989

-- Definitions based on the conditions:
def total_cards : ℕ := 52
def ranks_per_suit : ℕ := 13
def suits : ℕ := 4
def kings : ℕ := 4
def queens : ℕ := 4

-- The problem statement rephrased as a theorem:
theorem probability_king_then_queen :
  (kings / total_cards : ℚ) * (queens / (total_cards - 1)) = 4 / 663 := 
by {
  sorry
}

end probability_king_then_queen_l151_151989


namespace find_general_term_l151_151174

-- Definition of sequence sum condition
def seq_sum_condition (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (2/3) * a n + 1/3

-- Statement of the proof problem
theorem find_general_term (a S : ℕ → ℝ) 
  (h : seq_sum_condition a S) : 
  ∀ n, a n = (-2)^(n-1) := 
by
  sorry

end find_general_term_l151_151174


namespace sufficient_but_not_necessary_necessary_but_not_sufficient_l151_151834

def M (x : ℝ) : Prop := (x + 3) * (x - 5) > 0
def P (x : ℝ) (a : ℝ) : Prop := x^2 + (a - 8)*x - 8*a ≤ 0
def I : Set ℝ := {x | 5 < x ∧ x ≤ 8}

theorem sufficient_but_not_necessary (a : ℝ) :
  (∀ x, M x ∧ P x a ↔ x ∈ I) → a = 0 :=
sorry

theorem necessary_but_not_sufficient (a : ℝ) :
  (∀ x, (M x ∧ P x a → x ∈ I) ∧ (∀ x, x ∈ I → M x ∧ P x a)) → a ≤ 3 :=
sorry

end sufficient_but_not_necessary_necessary_but_not_sufficient_l151_151834


namespace non_neg_ints_less_than_pi_l151_151916

-- Define the condition: non-negative integers with absolute value less than π
def condition (x : ℕ) : Prop := |(x : ℝ)| < Real.pi

-- Prove that the set satisfying the condition is {0, 1, 2, 3}
theorem non_neg_ints_less_than_pi :
  {x : ℕ | condition x} = {0, 1, 2, 3} := by
  sorry

end non_neg_ints_less_than_pi_l151_151916


namespace f_g_of_1_l151_151987

-- Definitions of the given functions
def f (x : ℝ) : ℝ := x^2 + 5 * x + 6
def g (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

-- The statement we need to prove
theorem f_g_of_1 : f (g 1) = 132 := by
  sorry

end f_g_of_1_l151_151987


namespace min_value_x_plus_one_over_x_plus_two_l151_151873

theorem min_value_x_plus_one_over_x_plus_two (x : ℝ) (h : x > -2) : ∃ y : ℝ, y = x + 1/(x + 2) ∧ y ≥ 0 :=
by
  sorry

end min_value_x_plus_one_over_x_plus_two_l151_151873


namespace motorcycle_travel_distance_l151_151535

noncomputable def motorcycle_distance : ℝ :=
  let t : ℝ := 1 / 2  -- time in hours (30 minutes)
  let v_bus : ℝ := 90  -- speed of the bus in km/h
  let v_motorcycle : ℝ := (2 / 3) * v_bus  -- speed of the motorcycle in km/h
  v_motorcycle * t  -- calculates the distance traveled by the motorcycle in km

theorem motorcycle_travel_distance :
  motorcycle_distance = 30 := by
  sorry

end motorcycle_travel_distance_l151_151535


namespace monthly_income_of_P_l151_151945

theorem monthly_income_of_P (P Q R : ℕ) (h1 : P + Q = 10100) (h2 : Q + R = 12500) (h3 : P + R = 10400) : 
  P = 4000 := 
by 
  sorry

end monthly_income_of_P_l151_151945


namespace student_ages_inconsistent_l151_151360

theorem student_ages_inconsistent :
  let total_students := 24
  let avg_age_total := 18
  let group1_students := 6
  let avg_age_group1 := 16
  let group2_students := 10
  let avg_age_group2 := 20
  let group3_students := 7
  let avg_age_group3 := 22
  let total_age_all_students := total_students * avg_age_total
  let total_age_group1 := group1_students * avg_age_group1
  let total_age_group2 := group2_students * avg_age_group2
  let total_age_group3 := group3_students * avg_age_group3
  total_age_all_students < total_age_group1 + total_age_group2 + total_age_group3 :=
by {
  let total_students := 24
  let avg_age_total := 18
  let group1_students := 6
  let avg_age_group1 := 16
  let group2_students := 10
  let avg_age_group2 := 20
  let group3_students := 7
  let avg_age_group3 := 22
  let total_age_all_students := total_students * avg_age_total
  let total_age_group1 := group1_students * avg_age_group1
  let total_age_group2 := group2_students * avg_age_group2
  let total_age_group3 := group3_students * avg_age_group3
  have h₁ : total_age_all_students = 24 * 18 := rfl
  have h₂ : total_age_group1 = 6 * 16 := rfl
  have h₃ : total_age_group2 = 10 * 20 := rfl
  have h₄ : total_age_group3 = 7 * 22 := rfl
  have h₅ : 432 = 24 * 18 := by norm_num
  have h₆ : 96 = 6 * 16 := by norm_num
  have h₇ : 200 = 10 * 20 := by norm_num
  have h₈ : 154 = 7 * 22 := by norm_num
  have h₉ : 432 < 96 + 200 + 154 := by norm_num
  exact h₉
}

end student_ages_inconsistent_l151_151360


namespace alex_needs_packs_of_buns_l151_151994

-- Definitions (conditions)
def guests : ℕ := 10
def burgers_per_guest : ℕ := 3
def meat_eating_guests : ℕ := guests - 1
def bread_eating_ratios : ℕ := meat_eating_guests - 1
def buns_per_pack : ℕ := 8

-- Theorem (question == answer)
theorem alex_needs_packs_of_buns : 
  (burgers_per_guest * meat_eating_guests - burgers_per_guest) / buns_per_pack = 3 := by
  sorry

end alex_needs_packs_of_buns_l151_151994


namespace inequality_proof_l151_151042

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 9 * x * y * z) :
    x / Real.sqrt (x^2 + 2 * y * z + 2) + y / Real.sqrt (y^2 + 2 * z * x + 2) + z / Real.sqrt (z^2 + 2 * x * y + 2) ≥ 1 :=
by
  sorry

end inequality_proof_l151_151042


namespace football_championship_min_games_l151_151972

theorem football_championship_min_games :
  (∃ (teams : Finset ℕ) (games : Finset (ℕ × ℕ)),
    teams.card = 20 ∧
    (∀ (a b c : ℕ), a ∈ teams → b ∈ teams → c ∈ teams → a ≠ b → b ≠ c → c ≠ a →
      (a, b) ∈ games ∨ (b, c) ∈ games ∨ (c, a) ∈ games) ∧
    games.card = 90) :=
sorry

end football_championship_min_games_l151_151972


namespace find_t_l151_151428

variable (s t : ℚ) -- Using the rational numbers since the correct answer involves a fraction

theorem find_t (h1 : 8 * s + 7 * t = 145) (h2 : s = t + 3) : t = 121 / 15 :=
by 
  sorry

end find_t_l151_151428


namespace race_results_l151_151344

-- Competitor times in seconds
def time_A : ℕ := 40
def time_B : ℕ := 50
def time_C : ℕ := 55

-- Time difference calculations
def time_diff_AB := time_B - time_A
def time_diff_AC := time_C - time_A
def time_diff_BC := time_C - time_B

theorem race_results :
  time_diff_AB = 10 ∧ time_diff_AC = 15 ∧ time_diff_BC = 5 :=
by
  -- Placeholder for proof
  sorry

end race_results_l151_151344


namespace point_not_on_transformed_plane_l151_151659

def point_A : ℝ × ℝ × ℝ := (4, 0, -3)

def plane_eq (x y z : ℝ) : ℝ := 7 * x - y + 3 * z - 1

def scale_factor : ℝ := 3

def transformed_plane_eq (x y z : ℝ) : ℝ := 7 * x - y + 3 * z - (scale_factor * 1)

theorem point_not_on_transformed_plane :
  transformed_plane_eq 4 0 (-3) ≠ 0 :=
by
  sorry

end point_not_on_transformed_plane_l151_151659


namespace all_points_lie_on_parabola_l151_151909

noncomputable def parabola_curve (u : ℝ) : ℝ × ℝ :=
  let x := 3^u - 4
  let y := 9^u - 7 * 3^u - 2
  (x, y)

theorem all_points_lie_on_parabola (u : ℝ) :
  let (x, y) := parabola_curve u
  y = x^2 + x - 6 := sorry

end all_points_lie_on_parabola_l151_151909


namespace lcm_36_105_l151_151998

theorem lcm_36_105 : Int.lcm 36 105 = 1260 := by
  sorry

end lcm_36_105_l151_151998


namespace second_and_third_shooters_cannot_win_or_lose_simultaneously_l151_151842

-- Define the conditions C1, C2, and C3
variables (C1 C2 C3 : Prop)

-- The first shooter bets that at least one of the second or third shooters will miss
def first_shooter_bet : Prop := ¬ (C2 ∧ C3)

-- The second shooter bets that if the first shooter hits, then at least one of the remaining shooters will miss
def second_shooter_bet : Prop := C1 → ¬ (C2 ∧ C3)

-- The third shooter bets that all three will hit the target on the first attempt
def third_shooter_bet : Prop := C1 ∧ C2 ∧ C3

-- Prove that it is impossible for both the second and third shooters to either win or lose their bets concurrently
theorem second_and_third_shooters_cannot_win_or_lose_simultaneously :
  ¬ ((second_shooter_bet C1 C2 C3 ∧ third_shooter_bet C1 C2 C3) ∨ (¬ second_shooter_bet C1 C2 C3 ∧ ¬ third_shooter_bet C1 C2 C3)) :=
by
  sorry

end second_and_third_shooters_cannot_win_or_lose_simultaneously_l151_151842


namespace max_arith_seq_20_terms_l151_151424

noncomputable def max_arithmetic_sequences :
  Nat :=
  180

theorem max_arith_seq_20_terms (a : Nat → Nat) :
  (∀ (k : Nat), k ≥ 1 ∧ k ≤ 20 → ∃ d : Nat, a (k + 1) = a k + d) →
  (P : Nat) = max_arithmetic_sequences :=
  by
  -- here's where the proof would go
  sorry

end max_arith_seq_20_terms_l151_151424


namespace intersection_M_N_l151_151083

open Set

variable (x : ℝ)
def M : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}
def N : Set ℝ := {-2, 0, 2}

theorem intersection_M_N : M ∩ N = {0, 2} := sorry

end intersection_M_N_l151_151083


namespace triangle_area_is_correct_l151_151423

-- Defining the points
structure Point where
  x : ℝ
  y : ℝ

-- Defining vertices A, B, C
def A : Point := { x := 2, y := -3 }
def B : Point := { x := 0, y := 4 }
def C : Point := { x := 3, y := -1 }

-- Vector from C to A
def v : Point := { x := A.x - C.x, y := A.y - C.y }

-- Vector from C to B
def w : Point := { x := B.x - C.x, y := B.y - C.y }

-- Cross product of vectors v and w in 2D
noncomputable def cross_product (v w : Point) : ℝ :=
  v.x * w.y - v.y * w.x

-- Absolute value of the cross product
noncomputable def abs_cross_product (v w : Point) : ℝ :=
  |cross_product v w|

-- Area of the triangle
noncomputable def area_of_triangle (v w : Point) : ℝ :=
  (1 / 2) * abs_cross_product v w

-- Prove the area of the triangle is 5.5
theorem triangle_area_is_correct : area_of_triangle v w = 5.5 :=
  sorry

end triangle_area_is_correct_l151_151423


namespace pi_is_irrational_l151_151283

theorem pi_is_irrational :
  (¬ ∃ (p q : ℤ), q ≠ 0 ∧ π = p / q) :=
by
  sorry

end pi_is_irrational_l151_151283


namespace find_k_l151_151038

theorem find_k : ∃ k : ℕ, (2 * (Real.sqrt (225 + k)) = (Real.sqrt (49 + k) + Real.sqrt (441 + k))) → k = 255 :=
by
  sorry

end find_k_l151_151038


namespace quadratic_value_at_6_l151_151728

def f (a b x : ℝ) : ℝ := a * x^2 + b * x - 3

theorem quadratic_value_at_6 
  (a b : ℝ) (h : a ≠ 0) 
  (h_eq : f a b 2 = f a b 4) : 
  f a b 6 = -3 :=
by
  sorry

end quadratic_value_at_6_l151_151728


namespace nell_gave_cards_l151_151634

theorem nell_gave_cards (c_original : ℕ) (c_left : ℕ) (cards_given : ℕ) :
  c_original = 528 → c_left = 252 → cards_given = c_original - c_left → cards_given = 276 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end nell_gave_cards_l151_151634


namespace arithmetic_geometric_sequence_solution_l151_151354

theorem arithmetic_geometric_sequence_solution 
  (a1 a2 b1 b2 b3 : ℝ) 
  (h1 : -2 * 2 + a2 = a1)
  (h2 : a1 * 2 - 8 = a2)
  (h3 : b2 ^ 2 = -2 * -8)
  (h4 : b2 = -4) :
  (a2 - a1) / b2 = 1 / 2 :=
by 
  sorry

end arithmetic_geometric_sequence_solution_l151_151354


namespace find_f3_l151_151282

theorem find_f3 (a b : ℝ) (f : ℝ → ℝ)
  (h1 : f 1 = 3)
  (h2 : f 2 = 6)
  (h3 : ∀ x, f x = a * x^2 + b * x + 1) :
  f 3 = 10 :=
sorry

end find_f3_l151_151282


namespace smallest_integer_form_l151_151127

theorem smallest_integer_form (m n : ℤ) : ∃ (a : ℤ), a = 2011 * m + 55555 * n ∧ a > 0 → a = 1 :=
by
  sorry

end smallest_integer_form_l151_151127


namespace least_incorrect_option_is_A_l151_151920

def dozen_units : ℕ := 12
def chairs_needed : ℕ := 4

inductive CompletionOption
| dozen
| dozens
| dozen_of
| dozens_of

def correct_option (op : CompletionOption) : Prop :=
  match op with
  | CompletionOption.dozen => dozen_units >= chairs_needed
  | CompletionOption.dozens => False
  | CompletionOption.dozen_of => False
  | CompletionOption.dozens_of => False

theorem least_incorrect_option_is_A : correct_option CompletionOption.dozen :=
by {
  sorry
}

end least_incorrect_option_is_A_l151_151920


namespace even_function_order_l151_151612

noncomputable def f (m : ℝ) (x : ℝ) := (m - 1) * x^2 + 6 * m * x + 2

theorem even_function_order (m : ℝ) (h_even : ∀ x : ℝ, f m (-x) = f m x) : 
  m = 0 ∧ f m (-2) < f m 1 ∧ f m 1 < f m 0 := by
sorry

end even_function_order_l151_151612


namespace add_and_round_58_29_l151_151491

def add_and_round_to_nearest_ten (a b : ℕ) : ℕ :=
  let sum := a + b
  let rounded_sum := if sum % 10 < 5 then sum - (sum % 10) else sum + (10 - sum % 10)
  rounded_sum

theorem add_and_round_58_29 : add_and_round_to_nearest_ten 58 29 = 90 := by
  sorry

end add_and_round_58_29_l151_151491


namespace largest_sphere_radius_on_torus_l151_151266

theorem largest_sphere_radius_on_torus :
  ∀ r : ℝ, 16 + (r - 1)^2 = (r + 2)^2 → r = 13 / 6 :=
by
  intro r
  intro h
  sorry

end largest_sphere_radius_on_torus_l151_151266


namespace combination_of_15_3_l151_151632

open Nat

theorem combination_of_15_3 : choose 15 3 = 455 :=
by
  -- The statement describes that the number of ways to choose 3 books out of 15 is 455
  sorry

end combination_of_15_3_l151_151632


namespace parabola_properties_l151_151930

-- Given conditions
variables (a b c : ℝ)
variable (h_vertex : ∃ a b c : ℝ, (∀ x, a * (x+1)^2 + 4 = ax^2 + b * x + c))
variable (h_intersection : ∃ A : ℝ, 2 < A ∧ A < 3 ∧ a * A^2 + b * A + c = 0)

-- Define the proof problem
theorem parabola_properties (h_vertex : (b = 2 * a)) (h_a : a < 0) (h_c : c = 4 + a) : 
  ∃ x : ℕ, x = 2 ∧ 
  (∀ a b c : ℝ, a * b * c < 0 → false) ∧ 
  (-4 < a ∧ a < -1 → false) ∧
  (a * c + 2 * b > 1 → false) :=
sorry

end parabola_properties_l151_151930


namespace tulip_count_l151_151714

theorem tulip_count (total_flowers : ℕ) (daisies : ℕ) (roses_ratio : ℚ)
  (tulip_count : ℕ) :
  total_flowers = 102 →
  daisies = 6 →
  roses_ratio = 5 / 6 →
  tulip_count = (total_flowers - daisies) * (1 - roses_ratio) →
  tulip_count = 16 :=
by
  intros h1 h2 h3 h4
  sorry

end tulip_count_l151_151714


namespace calc_num_articles_l151_151904

-- Definitions based on the conditions
def cost_price (C : ℝ) : ℝ := C
def selling_price (C : ℝ) : ℝ := 1.10000000000000004 * C
def num_articles (n : ℝ) (C : ℝ) (S : ℝ) : Prop := 55 * C = n * S

-- Proof Statement
theorem calc_num_articles (C : ℝ) : ∃ n : ℝ, num_articles n C (selling_price C) ∧ n = 50 :=
by sorry

end calc_num_articles_l151_151904


namespace diagonals_in_30_sided_polygon_l151_151554

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_30_sided_polygon :
  number_of_diagonals 30 = 405 :=
by sorry

end diagonals_in_30_sided_polygon_l151_151554


namespace sequence_inequality_l151_151819

theorem sequence_inequality (a : ℕ → ℕ)
  (h1 : a 0 > 0) -- Ensure all entries are positive integers.
  (h2 : ∀ k l m n : ℕ, k * l = m * n → a k + a l = a m + a n)
  {p q : ℕ} (hpq : p ∣ q) :
  a p ≤ a q :=
sorry

end sequence_inequality_l151_151819


namespace perfect_square_conditions_l151_151709

theorem perfect_square_conditions (k : ℕ) : 
  (∃ m : ℤ, k^2 - 101 * k = m^2) ↔ (k = 101 ∨ k = 2601) := 
by 
  sorry

end perfect_square_conditions_l151_151709


namespace sequence_is_aperiodic_l151_151591

noncomputable def sequence_a (a : ℕ → ℕ) : Prop :=
∀ k n : ℕ, k < 2^n → a k ≠ a (k + 2^n)

theorem sequence_is_aperiodic (a : ℕ → ℕ) (h_a : sequence_a a) : ¬(∃ p : ℕ, ∀ n k : ℕ, a k = a (k + n * p)) :=
sorry

end sequence_is_aperiodic_l151_151591


namespace value_of_expression_when_x_is_2_l151_151862

theorem value_of_expression_when_x_is_2 : 
  (3 * 2 + 4) ^ 2 = 100 := 
by
  sorry

end value_of_expression_when_x_is_2_l151_151862


namespace sequence_term_a_1000_eq_2340_l151_151817

theorem sequence_term_a_1000_eq_2340
  (a : ℕ → ℤ)
  (h1 : a 1 = 2007)
  (h2 : a 2 = 2008)
  (h_rec : ∀ n : ℕ, 1 ≤ n → a n + a (n + 1) + a (n + 2) = n) :
  a 1000 = 2340 :=
sorry

end sequence_term_a_1000_eq_2340_l151_151817


namespace meteorological_forecasts_inaccuracy_l151_151946

theorem meteorological_forecasts_inaccuracy :
  let pA_accurate := 0.8
  let pB_accurate := 0.7
  let pA_inaccurate := 1 - pA_accurate
  let pB_inaccurate := 1 - pB_accurate
  pA_inaccurate * pB_inaccurate = 0.06 :=
by
  sorry

end meteorological_forecasts_inaccuracy_l151_151946


namespace treadmill_discount_percentage_l151_151108

theorem treadmill_discount_percentage
  (p_t : ℝ) -- original price of the treadmill
  (t_p : ℝ) -- total amount paid for treadmill and plates
  (p_plate : ℝ) -- price of each plate
  (n_plate : ℕ) -- number of plates
  (h_t : p_t = 1350)
  (h_tp : t_p = 1045)
  (h_p_plate : p_plate = 50)
  (h_n_plate : n_plate = 2) :
  ((p_t - (t_p - n_plate * p_plate)) / p_t) * 100 = 30 :=
by
  sorry

end treadmill_discount_percentage_l151_151108


namespace cos_difference_l151_151715

theorem cos_difference (α β : ℝ) (h_α_acute : 0 < α ∧ α < π / 2)
                      (h_β_acute : 0 < β ∧ β < π / 2)
                      (h_cos_α : Real.cos α = 1 / 3)
                      (h_cos_sum : Real.cos (α + β) = -1 / 3) :
  Real.cos (α - β) = 23 / 27 := 
sorry

end cos_difference_l151_151715


namespace smallest_even_n_for_reducible_fraction_l151_151726

theorem smallest_even_n_for_reducible_fraction : 
  ∃ (N: ℕ), (N > 2013) ∧ (N % 2 = 0) ∧ (Nat.gcd (15 * N - 7) (22 * N - 5) > 1) ∧ N = 2144 :=
sorry

end smallest_even_n_for_reducible_fraction_l151_151726


namespace hoot_difference_l151_151070

def owl_hoot_rate : ℕ := 5
def heard_hoots_per_min : ℕ := 20
def owls_count : ℕ := 3

theorem hoot_difference :
  heard_hoots_per_min - (owls_count * owl_hoot_rate) = 5 := by
  sorry

end hoot_difference_l151_151070


namespace root_expression_value_l151_151520

variables (a b : ℝ)
noncomputable def quadratic_eq (a b : ℝ) : Prop := (a + b = 1 ∧ a * b = -1)

theorem root_expression_value (h : quadratic_eq a b) : 3 * a ^ 2 + 4 * b + (2 / a ^ 2) = 11 := sorry

end root_expression_value_l151_151520


namespace average_speed_round_trip_l151_151721

/--
Let \( d = 150 \) miles be the distance from City \( X \) to City \( Y \).
Let \( v1 = 50 \) mph be the speed from \( X \) to \( Y \).
Let \( v2 = 30 \) mph be the speed from \( Y \) to \( X \).
Then the average speed for the round trip is 37.5 mph.
-/
theorem average_speed_round_trip :
  let d := 150
  let v1 := 50
  let v2 := 30
  (2 * d) / ((d / v1) + (d / v2)) = 37.5 :=
by
  sorry

end average_speed_round_trip_l151_151721


namespace sum_of_geometric_ratios_l151_151674

theorem sum_of_geometric_ratios (k a2 a3 b2 b3 p r : ℝ)
  (h_seq1 : a2 = k * p)
  (h_seq2 : a3 = k * p^2)
  (h_seq3 : b2 = k * r)
  (h_seq4 : b3 = k * r^2)
  (h_diff : a3 - b3 = 3 * (a2 - b2) - k) :
  p + r = 2 :=
by
  sorry

end sum_of_geometric_ratios_l151_151674


namespace eval_f_neg_2_l151_151905

def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem eval_f_neg_2 : f (-2) = 19 :=
by
  sorry

end eval_f_neg_2_l151_151905


namespace infinite_solutions_l151_151758

theorem infinite_solutions (x : ℕ) :
  15 < 2 * x + 10 ↔ ∃ n : ℕ, x = n + 3 :=
by {
  sorry
}

end infinite_solutions_l151_151758


namespace meadow_total_revenue_correct_l151_151496

-- Define the given quantities and conditions as Lean definitions
def total_diapers : ℕ := 192000
def price_per_diaper : ℝ := 4.0
def bundle_discount : ℝ := 0.05
def purchase_discount : ℝ := 0.05
def tax_rate : ℝ := 0.10

-- Define a function that calculates the revenue from selling all the diapers
def calculate_revenue (total_diapers : ℕ) (price_per_diaper : ℝ) (bundle_discount : ℝ) 
    (purchase_discount : ℝ) (tax_rate : ℝ) : ℝ :=
  let gross_revenue := total_diapers * price_per_diaper
  let bundle_discounted_revenue := gross_revenue * (1 - bundle_discount)
  let purchase_discounted_revenue := bundle_discounted_revenue * (1 - purchase_discount)
  let taxed_revenue := purchase_discounted_revenue * (1 + tax_rate)
  taxed_revenue

-- The main theorem to prove that the calculated revenue matches the expected value
theorem meadow_total_revenue_correct : 
  calculate_revenue total_diapers price_per_diaper bundle_discount purchase_discount tax_rate = 762432 := 
by
  sorry

end meadow_total_revenue_correct_l151_151496


namespace other_student_questions_l151_151990

theorem other_student_questions (m k o : ℕ) (h1 : m = k - 3) (h2 : k = o + 8) (h3 : m = 40) : o = 35 :=
by
  -- proof goes here
  sorry

end other_student_questions_l151_151990


namespace range_of_a_l151_151272

variable (a : ℝ)
variable (f : ℝ → ℝ)

-- Conditions
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def fWhenNegative (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f x = 9 * x + a^2 / x + 7

def fNonNegativeCondition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → f x ≥ a + 1

-- Theorem to prove
theorem range_of_a (odd_f : isOddFunction f) (f_neg : fWhenNegative f a) 
  (nonneg_cond : fNonNegativeCondition f a) : 
  a ≤ -8 / 7 :=
by
  sorry

end range_of_a_l151_151272


namespace friend_selling_price_l151_151193

-- Definitions and conditions
def original_cost_price : ℝ := 51724.14

def loss_percentage : ℝ := 0.13
def gain_percentage : ℝ := 0.20

def selling_price_man (CP : ℝ) : ℝ := (1 - loss_percentage) * CP
def selling_price_friend (SP1 : ℝ) : ℝ := (1 + gain_percentage) * SP1

-- Prove that the friend's selling price is 54,000 given the conditions
theorem friend_selling_price :
  selling_price_friend (selling_price_man original_cost_price) = 54000 :=
by
  sorry

end friend_selling_price_l151_151193


namespace Louis_ate_whole_boxes_l151_151219

def package_size := 6
def total_lemon_heads := 54

def whole_boxes : ℕ := total_lemon_heads / package_size

theorem Louis_ate_whole_boxes :
  whole_boxes = 9 :=
by
  sorry

end Louis_ate_whole_boxes_l151_151219


namespace part1_solution_part2_solution_l151_151225

theorem part1_solution (x : ℝ) (h1 : (2 * x) / (x - 2) + 3 / (2 - x) = 1) : x = 1 := by
  sorry

theorem part2_solution (x : ℝ) 
  (h1 : 2 * x - 1 ≥ 3 * (x - 1)) 
  (h2 : (5 - x) / 2 < x + 3) : -1 / 3 < x ∧ x ≤ 2 := by
  sorry

end part1_solution_part2_solution_l151_151225


namespace even_function_derivative_l151_151638

theorem even_function_derivative (f : ℝ → ℝ)
  (h_even : ∀ x, f (-x) = f x)
  (h_deriv_pos : ∀ x > 0, deriv f x = (x - 1) * (x - 2)) : f (-2) < f 1 :=
sorry

end even_function_derivative_l151_151638


namespace union_area_of_reflected_triangles_l151_151173

open Real

noncomputable def pointReflected (P : ℝ × ℝ) (line_y : ℝ) : ℝ × ℝ :=
  (P.1, 2 * line_y - P.2)

def areaOfTriangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem union_area_of_reflected_triangles :
  let A := (2, 6)
  let B := (5, -2)
  let C := (7, 3)
  let line_y := 2
  let A' := pointReflected A line_y
  let B' := pointReflected B line_y
  let C' := pointReflected C line_y
  areaOfTriangle A B C + areaOfTriangle A' B' C' = 29 := sorry

end union_area_of_reflected_triangles_l151_151173


namespace arthur_bakes_muffins_l151_151207

-- Definitions of the conditions
def james_muffins : ℚ := 9.58333333299999
def multiplier : ℚ := 12.0

-- Statement of the problem
theorem arthur_bakes_muffins : 
  abs (multiplier * james_muffins - 115) < 1 :=
by
  sorry

end arthur_bakes_muffins_l151_151207


namespace exists_unique_adjacent_sums_in_circle_l151_151342

theorem exists_unique_adjacent_sums_in_circle :
  ∃ (f : Fin 10 → Fin 11),
    (∀ (i j : Fin 10), i ≠ j → (f i + f (i + 1)) % 11 ≠ (f j + f (j + 1)) % 11) :=
sorry

end exists_unique_adjacent_sums_in_circle_l151_151342


namespace audit_options_correct_l151_151460

-- Define the initial number of ORs and GTUs
def initial_ORs : ℕ := 13
def initial_GTUs : ℕ := 15

-- Define the number of ORs and GTUs visited in the first week
def visited_ORs : ℕ := 2
def visited_GTUs : ℕ := 3

-- Calculate the remaining ORs and GTUs
def remaining_ORs : ℕ := initial_ORs - visited_ORs
def remaining_GTUs : ℕ := initial_GTUs - visited_GTUs

-- Calculate the number of ways to choose 2 ORs from remaining ORs
def choose_ORs : ℕ := Nat.choose remaining_ORs 2

-- Calculate the number of ways to choose 3 GTUs from remaining GTUs
def choose_GTUs : ℕ := Nat.choose remaining_GTUs 3

-- The final function to calculate the number of options
def number_of_options : ℕ := choose_ORs * choose_GTUs

-- The proof statement asserting the number of options is 12100
theorem audit_options_correct : number_of_options = 12100 := by
    sorry -- Proof will be filled in here

end audit_options_correct_l151_151460


namespace find_number_of_raccoons_squirrels_opossums_l151_151034

theorem find_number_of_raccoons_squirrels_opossums
  (R : ℕ)
  (total_animals : ℕ)
  (number_of_squirrels : ℕ := 6 * R)
  (number_of_opossums : ℕ := 2 * R)
  (total : ℕ := R + number_of_squirrels + number_of_opossums) 
  (condition : total_animals = 168)
  (correct_total : total = total_animals) :
  ∃ R : ℕ, R + 6 * R + 2 * R = total_animals :=
by
  sorry

end find_number_of_raccoons_squirrels_opossums_l151_151034


namespace binary_remainder_div_4_is_1_l151_151874

def binary_to_base_10_last_two_digits (b1 b0 : Nat) : Nat :=
  2 * b1 + b0

noncomputable def remainder_of_binary_by_4 (n : Nat) : Nat :=
  match n with
  | 111010110101 => binary_to_base_10_last_two_digits 0 1
  | _ => 0

theorem binary_remainder_div_4_is_1 :
  remainder_of_binary_by_4 111010110101 = 1 := by
  sorry

end binary_remainder_div_4_is_1_l151_151874


namespace velvet_needed_for_box_l151_151448

theorem velvet_needed_for_box :
  let long_side_area := 2 * (8 * 6)
  let short_side_area := 2 * (5 * 6)
  let top_and_bottom_area := 2 * 40
  long_side_area + short_side_area + top_and_bottom_area = 236 := by
{
  let long_side_area := 2 * (8 * 6)
  let short_side_area := 2 * (5 * 6)
  let top_and_bottom_area := 2 * 40
  sorry
}

end velvet_needed_for_box_l151_151448


namespace find_other_number_l151_151078

open BigOperators

noncomputable def other_number (n : ℕ) : Prop := n = 12

theorem find_other_number (n : ℕ) (h_lcm : Nat.lcm 8 n = 24) (h_hcf : Nat.gcd 8 n = 4) : other_number n := 
by
  sorry

end find_other_number_l151_151078


namespace g_increasing_g_multiplicative_g_special_case_g_18_value_l151_151321

def g (n : ℕ) : ℕ :=
sorry

theorem g_increasing : ∀ n : ℕ, n > 0 → g (n + 1) > g n :=
sorry

theorem g_multiplicative : ∀ m n : ℕ, m > 0 → n > 0 → g (m * n) = g m * g n :=
sorry

theorem g_special_case : ∀ m n : ℕ, m > 0 → n > 0 → m ≠ n → m ^ n = n ^ m → g m = n ∨ g n = m :=
sorry

theorem g_18_value : g 18 = 324 :=
sorry

end g_increasing_g_multiplicative_g_special_case_g_18_value_l151_151321


namespace minimum_apples_to_guarantee_18_one_color_l151_151662

theorem minimum_apples_to_guarantee_18_one_color :
  let red := 32
  let green := 24
  let yellow := 22
  let blue := 15
  let orange := 14
  ∀ n, (n >= 81) →
  (∃ red_picked green_picked yellow_picked blue_picked orange_picked : ℕ,
    red_picked + green_picked + yellow_picked + blue_picked + orange_picked = n
    ∧ red_picked ≤ red ∧ green_picked ≤ green ∧ yellow_picked ≤ yellow ∧ blue_picked ≤ blue ∧ orange_picked ≤ orange
    ∧ (red_picked = 18 ∨ green_picked = 18 ∨ yellow_picked = 18 ∨ blue_picked = 18 ∨ orange_picked = 18)) :=
by {
  -- The proof is omitted for now.
  sorry
}

end minimum_apples_to_guarantee_18_one_color_l151_151662


namespace equilateral_triangle_stack_impossible_l151_151088

theorem equilateral_triangle_stack_impossible :
  ¬ ∃ n : ℕ, 3 * 55 = 6 * n :=
by
  sorry

end equilateral_triangle_stack_impossible_l151_151088


namespace g_sum_eq_neg_one_l151_151681

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Main theorem to prove g(1) + g(-1) = -1 given the conditions
theorem g_sum_eq_neg_one
  (h1 : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y)
  (h2 : f (-2) = f 1)
  (h3 : f 1 ≠ 0) :
  g 1 + g (-1) = -1 :=
sorry

end g_sum_eq_neg_one_l151_151681


namespace total_students_in_lunchroom_l151_151433

theorem total_students_in_lunchroom 
  (students_per_table : ℕ) 
  (number_of_tables : ℕ) 
  (h1 : students_per_table = 6) 
  (h2 : number_of_tables = 34) : 
  students_per_table * number_of_tables = 204 := by
  sorry

end total_students_in_lunchroom_l151_151433


namespace f_in_neg_interval_l151_151203

variables (f : ℝ → ℝ)

-- Conditions
def is_even := ∀ x, f x = f (-x)
def symmetry := ∀ x, f (2 + x) = f (2 - x)
def in_interval := ∀ x, 0 < x ∧ x < 2 → f x = 1 / x

-- Target statement
theorem f_in_neg_interval
  (h_even : is_even f)
  (h_symm : symmetry f)
  (h_interval : in_interval f)
  (x : ℝ)
  (hx : -4 < x ∧ x < -2) :
  f x = 1 / (x + 4) :=
sorry

end f_in_neg_interval_l151_151203


namespace abs_iff_sq_gt_l151_151392

theorem abs_iff_sq_gt (x y : ℝ) : (|x| > |y|) ↔ (x^2 > y^2) :=
by sorry

end abs_iff_sq_gt_l151_151392


namespace distance_covered_l151_151590

-- Define the conditions
def speed_still_water : ℕ := 30   -- 30 kmph
def current_speed : ℕ := 6        -- 6 kmph
def time_downstream : ℕ := 24     -- 24 seconds

-- Proving the distance covered downstream
theorem distance_covered (s_still s_current t : ℕ) (h_s_still : s_still = speed_still_water) (h_s_current : s_current = current_speed) (h_t : t = time_downstream):
  (s_still + s_current) * 1000 / 3600 * t = 240 :=
by sorry

end distance_covered_l151_151590


namespace exponential_quotient_l151_151509

variable {x a b : ℝ}

theorem exponential_quotient (h1 : x^a = 3) (h2 : x^b = 5) : x^(a-b) = 3 / 5 :=
sorry

end exponential_quotient_l151_151509


namespace pete_ten_dollar_bills_l151_151456

theorem pete_ten_dollar_bills (owes dollars bills: ℕ) (bill_value_per_bottle : ℕ) (num_bottles : ℕ) (ten_dollar_bills : ℕ):
  owes = 90 →
  dollars = 40 →
  bill_value_per_bottle = 5 →
  num_bottles = 20 →
  dollars + (num_bottles * bill_value_per_bottle) + (ten_dollar_bills * 10) = owes →
  ten_dollar_bills = 4 :=
by
  sorry

end pete_ten_dollar_bills_l151_151456


namespace range_of_quadratic_expression_l151_151523

theorem range_of_quadratic_expression :
  (∃ x : ℝ, y = 2 * x^2 - 4 * x + 12) ↔ (y ≥ 10) :=
by
  sorry

end range_of_quadratic_expression_l151_151523


namespace problem1_problem2_l151_151855

def f (x a : ℝ) : ℝ := abs (1 - x - a) + abs (2 * a - x)

theorem problem1 (a : ℝ) (h : f 1 a < 3) : -2/3 < a ∧ a < 4/3 :=
  sorry

theorem problem2 (a x : ℝ) (h : a ≥ 2/3) : f x a ≥ 1 :=
  sorry

end problem1_problem2_l151_151855


namespace largest_possible_perimeter_l151_151480

noncomputable def max_perimeter_triangle : ℤ :=
  let a : ℤ := 7
  let b : ℤ := 9
  let x : ℤ := 15
  a + b + x

theorem largest_possible_perimeter (x : ℤ) (h1 : 7 + 9 > x) (h2 : 7 + x > 9) (h3 : 9 + x > 7) : max_perimeter_triangle = 31 := by
  sorry

end largest_possible_perimeter_l151_151480


namespace simplify_fraction_l151_151949

theorem simplify_fraction : (90 : ℚ) / (150 : ℚ) = (3 : ℚ) / (5 : ℚ) := by
  sorry

end simplify_fraction_l151_151949


namespace completing_the_square_l151_151800

theorem completing_the_square (x : ℝ) : x^2 + 2 * x - 5 = 0 → (x + 1)^2 = 6 := by
  intro h
  -- Starting from h and following the steps outlined to complete the square.
  sorry

end completing_the_square_l151_151800


namespace horner_eval_v3_at_minus4_l151_151748

def f (x : ℤ) : ℤ := 12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

def horner_form (x : ℤ) : ℤ :=
  let a6 := 3
  let a5 := 5
  let a4 := 6
  let a3 := 79
  let a2 := -8
  let a1 := 35
  let a0 := 12
  let v := a6
  let v1 := v * x + a5
  let v2 := v1 * x + a4
  let v3 := v2 * x + a3
  let v4 := v3 * x + a2
  let v5 := v4 * x + a1
  let v6 := v5 * x + a0
  v3

theorem horner_eval_v3_at_minus4 :
  horner_form (-4) = -57 :=
by
  sorry

end horner_eval_v3_at_minus4_l151_151748


namespace trees_left_after_typhoon_l151_151508

variable (initial_trees : ℕ)
variable (died_trees : ℕ)
variable (remaining_trees : ℕ)

theorem trees_left_after_typhoon :
  initial_trees = 20 →
  died_trees = 16 →
  remaining_trees = initial_trees - died_trees →
  remaining_trees = 4 :=
by
  intros h_initial h_died h_remaining
  rw [h_initial, h_died] at h_remaining
  exact h_remaining

end trees_left_after_typhoon_l151_151508


namespace finite_transformation_l151_151641

-- Define the function representing the number transformation
def transform (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 5

-- Define the predicate stating that the process terminates
def process_terminates (n : ℕ) : Prop :=
  ∃ k : ℕ, k > 0 ∧ transform^[k] n = 1

-- Lean 4 statement for the theorem
theorem finite_transformation (n : ℕ) (h : n > 1) : process_terminates n ↔ ¬ (∃ m : ℕ, m > 0 ∧ n = 5 * m) :=
by
  sorry

end finite_transformation_l151_151641


namespace all_ones_l151_151494

theorem all_ones (k : ℕ) (h₁ : k ≥ 2) (n : ℕ → ℕ) (h₂ : ∀ i, 1 ≤ i → i < k → n (i + 1) ∣ (2 ^ n i - 1))
(h₃ : n 1 ∣ (2 ^ n k - 1)) : (∀ i, 1 ≤ i → i ≤ k → n i = 1) :=
by
  sorry

end all_ones_l151_151494


namespace apples_total_l151_151969

theorem apples_total (initial_apples : ℕ) (additional_apples : ℕ) (total_apples : ℕ) : 
  initial_apples = 56 → 
  additional_apples = 49 → 
  total_apples = initial_apples + additional_apples → 
  total_apples = 105 :=
by 
  intros h_initial h_additional h_total 
  rw [h_initial, h_additional] at h_total 
  exact h_total

end apples_total_l151_151969


namespace fixed_monthly_fee_l151_151431

/-
  We want to prove that given two conditions:
  1. x + y = 12.48
  2. x + 2y = 17.54
  The fixed monthly fee (x) is 7.42.
-/

theorem fixed_monthly_fee (x y : ℝ) 
  (h1 : x + y = 12.48) 
  (h2 : x + 2 * y = 17.54) : 
  x = 7.42 := 
sorry

end fixed_monthly_fee_l151_151431


namespace graph_is_finite_distinct_points_l151_151133

def cost (n : ℕ) : ℕ := 18 * n + 3

theorem graph_is_finite_distinct_points : 
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 20 → 
  ∀ (m : ℕ), 1 ≤ m ∧ m ≤ 20 → 
  (cost n = cost m → n = m) ∧
  ∀ x : ℕ, ∃ n : ℕ, 1 ≤ n ∧ n ≤ 20 ∧ cost n = x :=
by
  sorry

end graph_is_finite_distinct_points_l151_151133


namespace number_of_paper_cups_is_40_l151_151878

noncomputable def cost_paper_plate : ℝ := sorry
noncomputable def cost_paper_cup : ℝ := sorry
noncomputable def num_paper_cups_in_second_purchase : ℝ := sorry

-- Conditions
axiom first_condition : 100 * cost_paper_plate + 200 * cost_paper_cup = 7.50
axiom second_condition : 20 * cost_paper_plate + num_paper_cups_in_second_purchase * cost_paper_cup = 1.50

-- Goal
theorem number_of_paper_cups_is_40 : num_paper_cups_in_second_purchase = 40 := 
by 
  sorry

end number_of_paper_cups_is_40_l151_151878


namespace original_denominator_is_nine_l151_151651

theorem original_denominator_is_nine (d : ℕ) : 
  (2 + 5) / (d + 5) = 1 / 2 → d = 9 := 
by sorry

end original_denominator_is_nine_l151_151651


namespace linear_function_no_second_quadrant_l151_151672

theorem linear_function_no_second_quadrant (x y : ℝ) (h : y = 2 * x - 3) :
  ¬ ((x < 0) ∧ (y > 0)) :=
by {
  sorry
}

end linear_function_no_second_quadrant_l151_151672


namespace number_of_moles_H2SO4_formed_l151_151376

-- Define the moles of reactants
def initial_moles_SO2 : ℕ := 1
def initial_moles_H2O2 : ℕ := 1

-- Given the balanced chemical reaction
-- SO2 + H2O2 → H2SO4
def balanced_reaction := (1, 1) -- Representing the reactant coefficients for SO2 and H2O2

-- Define the number of moles of product formed
def moles_H2SO4 (moles_SO2 moles_H2O2 : ℕ) : ℕ :=
moles_SO2 -- Since according to balanced equation, 1 mole of each reactant produces 1 mole of product

theorem number_of_moles_H2SO4_formed :
  moles_H2SO4 initial_moles_SO2 initial_moles_H2O2 = 1 := by
  sorry

end number_of_moles_H2SO4_formed_l151_151376


namespace total_boys_and_girls_sum_to_41_l151_151976

theorem total_boys_and_girls_sum_to_41 (Rs : ℕ) (amount_per_boy : ℕ) (amount_per_girl : ℕ) (total_amount : ℕ) (num_boys : ℕ) :
  Rs = 1 ∧ amount_per_boy = 12 * Rs ∧ amount_per_girl = 8 * Rs ∧ total_amount = 460 * Rs ∧ num_boys = 33 →
  ∃ num_girls : ℕ, num_boys + num_girls = 41 :=
by
  sorry

end total_boys_and_girls_sum_to_41_l151_151976


namespace find_numbers_l151_151571

def is_solution (a b : ℕ) : Prop :=
  a + b = 432 ∧ (max a b) = 5 * (min a b) ∧ (max a b = 360 ∧ min a b = 72)

theorem find_numbers : ∃ a b : ℕ, is_solution a b :=
by
  sorry

end find_numbers_l151_151571


namespace find_b_value_l151_151575

theorem find_b_value : 
  ∀ (a b : ℝ), 
    (a^3 * b^4 = 2048) ∧ (a = 8) → b = Real.sqrt 2 := 
by 
sorry

end find_b_value_l151_151575


namespace average_minutes_per_day_is_correct_l151_151713
-- Import required library for mathematics

-- Define the conditions
def sixth_grade_minutes := 10
def seventh_grade_minutes := 12
def eighth_grade_minutes := 8
def sixth_grade_ratio := 3
def eighth_grade_ratio := 1/2

-- We use noncomputable since we'll rely on some real number operations that are not trivially computable.
noncomputable def total_minutes_per_week (s : ℝ) : ℝ :=
  sixth_grade_minutes * (sixth_grade_ratio * s) * 2 + 
  seventh_grade_minutes * s * 2 + 
  eighth_grade_minutes * (eighth_grade_ratio * s) * 1

noncomputable def total_students (s : ℝ) : ℝ :=
  sixth_grade_ratio * s + s + eighth_grade_ratio * s

noncomputable def average_minutes_per_day : ℝ :=
  (total_minutes_per_week 1) / (total_students 1 / 5)

theorem average_minutes_per_day_is_correct : average_minutes_per_day = 176 / 9 :=
by
  sorry

end average_minutes_per_day_is_correct_l151_151713


namespace value_of_f_at_3_l151_151687

def f (x : ℝ) := 2 * x - 1

theorem value_of_f_at_3 : f 3 = 5 := by
  sorry

end value_of_f_at_3_l151_151687


namespace union_complement_real_domain_l151_151195

noncomputable def M : Set ℝ := {x | -2 < x ∧ x < 2}
noncomputable def N : Set ℝ := {x | -2 < x}

theorem union_complement_real_domain :
  M ∪ (Set.univ \ N) = {x : ℝ | x < 2} :=
by
  sorry

end union_complement_real_domain_l151_151195


namespace profit_percentage_l151_151304

theorem profit_percentage (C S : ℝ) (h : 17 * C = 16 * S) : (S - C) / C * 100 = 6.25 := by
  sorry

end profit_percentage_l151_151304


namespace correct_value_l151_151790

-- Given condition
def incorrect_calculation (x : ℝ) : Prop := (x + 12) / 8 = 8

-- Theorem to prove the correct value
theorem correct_value (x : ℝ) (h : incorrect_calculation x) : (x - 12) * 9 = 360 :=
by
  sorry

end correct_value_l151_151790


namespace gemstones_needed_for_sets_l151_151089

-- Define the number of magnets per earring
def magnets_per_earring : ℕ := 2

-- Define the number of buttons per earring as half the number of magnets
def buttons_per_earring (magnets : ℕ) : ℕ := magnets / 2

-- Define the number of gemstones per earring as three times the number of buttons
def gemstones_per_earring (buttons : ℕ) : ℕ := 3 * buttons

-- Define the number of earrings per set
def earrings_per_set : ℕ := 2

-- Define the number of sets
def sets : ℕ := 4

-- Prove that Rebecca needs 24 gemstones for 4 sets of earrings given the conditions
theorem gemstones_needed_for_sets :
  gemstones_per_earring (buttons_per_earring magnets_per_earring) * earrings_per_set * sets = 24 :=
by
  sorry

end gemstones_needed_for_sets_l151_151089


namespace log_neq_x_minus_one_l151_151473

theorem log_neq_x_minus_one (x : ℝ) (h₁ : 0 < x) : Real.log x ≠ x - 1 :=
sorry

end log_neq_x_minus_one_l151_151473


namespace reporters_not_covering_politics_l151_151381

def total_reporters : ℝ := 8000
def politics_local : ℝ := 0.12 + 0.08 + 0.08 + 0.07 + 0.06 + 0.05 + 0.04 + 0.03 + 0.02 + 0.01
def politics_non_local : ℝ := 0.15
def politics_total : ℝ := politics_local + politics_non_local

theorem reporters_not_covering_politics :
  1 - politics_total = 0.29 :=
by
  -- Required definition and intermediate proof steps.
  sorry

end reporters_not_covering_politics_l151_151381


namespace plan_Y_cheaper_than_X_plan_Z_cheaper_than_X_l151_151772

theorem plan_Y_cheaper_than_X (x : ℕ) : 
  ∃ x, 2500 + 7 * x < 15 * x ∧ ∀ y, y < x → ¬ (2500 + 7 * y < 15 * y) := 
sorry

theorem plan_Z_cheaper_than_X (x : ℕ) : 
  ∃ x, 3000 + 6 * x < 15 * x ∧ ∀ y, y < x → ¬ (3000 + 6 * y < 15 * y) := 
sorry

end plan_Y_cheaper_than_X_plan_Z_cheaper_than_X_l151_151772


namespace find_a1_l151_151222

theorem find_a1 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_rec : ∀ n ≥ 2, a n + 2 * S n * S (n - 1) = 0)
  (h_S5 : S 5 = 1 / 11) : 
  a 1 = 1 / 3 := 
sorry

end find_a1_l151_151222


namespace total_snowfall_l151_151087

theorem total_snowfall (morning_snowfall : ℝ) (afternoon_snowfall : ℝ) (h_morning : morning_snowfall = 0.125) (h_afternoon : afternoon_snowfall = 0.5) :
  morning_snowfall + afternoon_snowfall = 0.625 :=
by 
  sorry

end total_snowfall_l151_151087


namespace geometric_sequence_sum_l151_151142

theorem geometric_sequence_sum (a : ℕ → ℝ)
  (h1 : a 1 + a 2 = 1/2)
  (h2 : a 3 + a 4 = 1)
  (h_geom : ∀ n, a n + a (n+1) = (a 1 + a 2) * 2^(n-1)) :
  a 7 + a 8 + a 9 + a 10 = 12 := 
sorry

end geometric_sequence_sum_l151_151142


namespace find_n_l151_151472

noncomputable def positive_geometric_sequence := ℕ → ℝ

def is_geometric_sequence (a : positive_geometric_sequence) :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def conditions (a : positive_geometric_sequence) :=
  is_geometric_sequence a ∧
  a 0 * a 1 * a 2 = 4 ∧
  a 3 * a 4 * a 5 = 12 ∧
  ∃ n : ℕ, a (n - 1) * a n * a (n + 1) = 324

theorem find_n (a : positive_geometric_sequence) (h : conditions a) : ∃ n : ℕ, n = 14 :=
by
  sorry

end find_n_l151_151472


namespace tommys_books_l151_151196

-- Define the cost of each book
def book_cost : ℕ := 5

-- Define the amount Tommy already has
def tommy_money : ℕ := 13

-- Define the amount Tommy needs to save up
def tommy_goal : ℕ := 27

-- Prove the number of books Tommy wants to buy
theorem tommys_books : tommy_goal + tommy_money = 40 ∧ (tommy_goal + tommy_money) / book_cost = 8 :=
by
  sorry

end tommys_books_l151_151196


namespace circles_intersect_at_two_points_l151_151597

theorem circles_intersect_at_two_points : 
  let C1 := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 9}
  let C2 := {p : ℝ × ℝ | p.1^2 + (p.2 - 6)^2 = 36}
  ∃ pts : Finset (ℝ × ℝ), pts.card = 2 ∧ ∀ p ∈ pts, p ∈ C1 ∧ p ∈ C2 := 
sorry

end circles_intersect_at_two_points_l151_151597


namespace sculpture_and_base_height_l151_151131

def height_sculpture : ℕ := 2 * 12 + 10
def height_base : ℕ := 8
def total_height : ℕ := 42

theorem sculpture_and_base_height :
  height_sculpture + height_base = total_height :=
by
  -- provide the necessary proof steps here
  sorry

end sculpture_and_base_height_l151_151131


namespace strictly_increasing_0_to_e_l151_151680

noncomputable def ln (x : ℝ) : ℝ := Real.log x

noncomputable def f (x : ℝ) : ℝ := ln x / x

theorem strictly_increasing_0_to_e :
  ∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → 0 < (1 - ln x) / (x^2) :=
by
  sorry

end strictly_increasing_0_to_e_l151_151680


namespace graph_not_pass_through_second_quadrant_l151_151614

theorem graph_not_pass_through_second_quadrant 
    (k : ℝ) (b : ℝ) (h1 : k = 1) (h2 : b = -2) : 
    ¬ ∃ (x y : ℝ), y = k * x + b ∧ x < 0 ∧ y > 0 := 
by
  sorry

end graph_not_pass_through_second_quadrant_l151_151614


namespace count_paths_word_l151_151589

def move_right_or_down_paths (n : ℕ) : ℕ := 2^n

theorem count_paths_word (n : ℕ) (w : String) (start : Char) (end_ : Char) :
    w = "строка" ∧ start = 'C' ∧ end_ = 'A' ∧ n = 5 →
    move_right_or_down_paths n = 32 :=
by
  intro h
  cases h
  sorry

end count_paths_word_l151_151589


namespace villages_population_equal_l151_151522

def population_x (initial_population rate_decrease : Int) (n : Int) := initial_population - rate_decrease * n
def population_y (initial_population rate_increase : Int) (n : Int) := initial_population + rate_increase * n

theorem villages_population_equal
    (initial_population_x : Int) (rate_decrease_x : Int)
    (initial_population_y : Int) (rate_increase_y : Int)
    (h₁ : initial_population_x = 76000) (h₂ : rate_decrease_x = 1200)
    (h₃ : initial_population_y = 42000) (h₄ : rate_increase_y = 800) :
    ∃ n : Int, population_x initial_population_x rate_decrease_x n = population_y initial_population_y rate_increase_y n ∧ n = 17 :=
by
    sorry

end villages_population_equal_l151_151522


namespace part1_ABC_inquality_part2_ABCD_inquality_l151_151653

theorem part1_ABC_inquality (a b c ABC : ℝ) : 
  (ABC <= (a^2 + b^2) / 4) -> 
  (ABC <= (b^2 + c^2) / 4) -> 
  (ABC <= (a^2 + c^2) / 4) -> 
    (ABC < (a^2 + b^2 + c^2) / 6) :=
sorry

theorem part2_ABCD_inquality (a b c d ABC BCD CDA DAB ABCD : ℝ) :
  (ABCD = 1/2 * ((ABC) + (BCD) + (CDA) + (DAB))) -> 
  (ABC < (a^2 + b^2 + c^2) / 6) -> 
  (BCD < (b^2 + c^2 + d^2) / 6) -> 
  (CDA < (c^2 + d^2 + a^2) / 6) -> 
  (DAB < (d^2 + a^2 + b^2) / 6) -> 
    (ABCD < (a^2 + b^2 + c^2 + d^2) / 6) :=
sorry

end part1_ABC_inquality_part2_ABCD_inquality_l151_151653


namespace gain_percentage_l151_151095

theorem gain_percentage (MP CP : ℝ) (h1 : 0.90 * MP = 1.17 * CP) :
  (((MP - CP) / CP) * 100) = 30 := 
by
  sorry

end gain_percentage_l151_151095


namespace commercials_count_l151_151779

-- Given conditions as definitions
def total_airing_time : ℤ := 90         -- 1.5 hours in minutes
def commercial_time : ℤ := 10           -- each commercial lasts 10 minutes
def show_time : ℤ := 60                 -- TV show (without commercials) lasts 60 minutes

-- Statement: Prove that the number of commercials is 3
theorem commercials_count :
  (total_airing_time - show_time) / commercial_time = 3 :=
sorry

end commercials_count_l151_151779


namespace intersection_correct_l151_151098

def A (x : ℝ) : Prop := |x| > 4
def B (x : ℝ) : Prop := -2 < x ∧ x ≤ 6
def intersection (x : ℝ) : Prop := B x ∧ A x

theorem intersection_correct :
  ∀ x : ℝ, intersection x ↔ 4 < x ∧ x ≤ 6 := 
by
  sorry

end intersection_correct_l151_151098


namespace time_to_eat_quarter_l151_151531

noncomputable def total_nuts : ℕ := sorry

def rate_first_crow (N : ℕ) := N / 40
def rate_second_crow (N : ℕ) := N / 36

theorem time_to_eat_quarter (N : ℕ) (T : ℝ) :
  (rate_first_crow N + rate_second_crow N) * T = (1 / 4 : ℝ) * N → 
  T = (90 / 19 : ℝ) :=
by
  intros h
  sorry

end time_to_eat_quarter_l151_151531


namespace business_hours_correct_l151_151035

-- Define the business hours
def start_time : ℕ := 8 * 60 + 30   -- 8:30 in minutes
def end_time : ℕ := 22 * 60 + 30    -- 22:30 in minutes

-- Calculate total business hours in minutes and convert it to hours
def total_business_hours : ℕ := (end_time - start_time) / 60

-- State the business hour condition (which says the total business hour is 15 hours).
def business_hour_claim : ℕ := 15

-- Formulate the statement to prove: the claim that the total business hours are 15 hours is false.
theorem business_hours_correct : total_business_hours ≠ business_hour_claim := by
  sorry

end business_hours_correct_l151_151035


namespace francie_has_3_dollars_remaining_l151_151870

def francies_remaining_money : ℕ :=
  let allowance1 := 5 * 8
  let allowance2 := 6 * 6
  let total_savings := allowance1 + allowance2
  let remaining_after_clothes := total_savings / 2
  remaining_after_clothes - 35

theorem francie_has_3_dollars_remaining :
  francies_remaining_money = 3 := 
sorry

end francie_has_3_dollars_remaining_l151_151870


namespace parabola_vertex_origin_through_point_l151_151482

theorem parabola_vertex_origin_through_point :
  (∃ p, p > 0 ∧ x^2 = 2 * p * y ∧ (x, y) = (-4, 4) → x^2 = 4 * y) ∨
  (∃ p, p > 0 ∧ y^2 = -2 * p * x ∧ (x, y) = (-4, 4) → y^2 = -4 * x) :=
sorry

end parabola_vertex_origin_through_point_l151_151482


namespace inequality_must_hold_l151_151856

theorem inequality_must_hold (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : a - d > b - c :=
by {
  sorry
}

end inequality_must_hold_l151_151856


namespace greatest_cars_with_ac_not_racing_stripes_l151_151727

-- Definitions
def total_cars : ℕ := 100
def cars_without_ac : ℕ := 47
def cars_with_ac : ℕ := total_cars - cars_without_ac
def at_least_racing_stripes : ℕ := 53

-- Prove that the greatest number of cars that could have air conditioning but not racing stripes is 53
theorem greatest_cars_with_ac_not_racing_stripes :
  ∃ maximum_cars_with_ac_not_racing_stripes, 
    maximum_cars_with_ac_not_racing_stripes = cars_with_ac - 0 ∧
    maximum_cars_with_ac_not_racing_stripes = 53 := 
by
  sorry

end greatest_cars_with_ac_not_racing_stripes_l151_151727


namespace marching_band_max_l151_151165

-- Define the conditions
variables (m k n : ℕ)

-- Lean statement of the problem
theorem marching_band_max (H1 : m = k^2 + 9) (H2 : m = n * (n + 5)) : m = 234 :=
sorry

end marching_band_max_l151_151165


namespace smaller_cube_size_l151_151226

theorem smaller_cube_size
  (original_cube_side : ℕ)
  (number_of_smaller_cubes : ℕ)
  (painted_cubes : ℕ)
  (unpainted_cubes : ℕ) :
  original_cube_side = 3 → 
  number_of_smaller_cubes = 27 → 
  painted_cubes = 26 → 
  unpainted_cubes = 1 →
  (∃ (side : ℕ), side = original_cube_side / 3 ∧ side = 1) :=
by
  intros h1 h2 h3 h4
  use 1
  have h : 1 = original_cube_side / 3 := sorry
  exact ⟨h, rfl⟩

end smaller_cube_size_l151_151226


namespace range_a_real_numbers_l151_151377

theorem range_a_real_numbers (x a : ℝ) : 
  (∀ x : ℝ, (x - a) * (1 - (x + a)) < 1) → (a ∈ Set.univ) :=
by
  sorry

end range_a_real_numbers_l151_151377


namespace f_zero_f_odd_f_inequality_solution_l151_151204

open Real

-- Given definitions
variables {f : ℝ → ℝ}
variable (h_inc : ∀ x y, x < y → f x < f y)
variable (h_eq : ∀ x y, y * f x - x * f y = x * y * (x^2 - y^2))

-- Prove that f(0) = 0
theorem f_zero : f 0 = 0 := 
sorry

-- Prove that f is an odd function
theorem f_odd : ∀ x, f (-x) = -f x := 
sorry

-- Prove the range of x satisfying the given inequality
theorem f_inequality_solution : {x : ℝ | f (x^2 + 1) + f (3 * x - 5) < 0} = {x : ℝ | -4 < x ∧ x < 1} :=
sorry

end f_zero_f_odd_f_inequality_solution_l151_151204


namespace sum_ad_eq_two_l151_151524

theorem sum_ad_eq_two (a b c d : ℝ) 
  (h1 : a + b = 4) 
  (h2 : b + c = 7) 
  (h3 : c + d = 5) : 
  a + d = 2 :=
by
  sorry

end sum_ad_eq_two_l151_151524


namespace sum_of_two_numbers_l151_151980

theorem sum_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : x * y = 200) : (x + y = 30) :=
by sorry

end sum_of_two_numbers_l151_151980


namespace value_of_m_l151_151755

theorem value_of_m (m : ℝ) (h : m ≠ 0)
  (h_roots : ∀ x, m * x^2 + 8 * m * x + 60 = 0 ↔ x = -5 ∨ x = -3) :
  m = 4 :=
sorry

end value_of_m_l151_151755


namespace merchant_profit_l151_151229

theorem merchant_profit 
  (CP MP SP profit : ℝ)
  (markup_percentage discount_percentage : ℝ)
  (h1 : CP = 100)
  (h2 : markup_percentage = 0.40)
  (h3 : discount_percentage = 0.10)
  (h4 : MP = CP + (markup_percentage * CP))
  (h5 : SP = MP - (discount_percentage * MP))
  (h6 : profit = SP - CP) :
  profit / CP * 100 = 26 :=
by sorry

end merchant_profit_l151_151229


namespace size_of_angle_C_max_value_of_a_add_b_l151_151691

variable (A B C a b c : ℝ)
variable (h₀ : 0 < A ∧ A < π / 2)
variable (h₁ : 0 < B ∧ B < π / 2)
variable (h₂ : 0 < C ∧ C < π / 2)
variable (h₃ : a = 2 * c * sin A / sqrt 3)
variable (h₄ : a * a + b * b - 2 * a * b * cos (π / 3) = c * c)

theorem size_of_angle_C (h₅: a ≠ 0):
  C = π / 3 :=
by sorry

theorem max_value_of_a_add_b (h₆: c = 2):
  a + b ≤ 4 :=
by sorry

end size_of_angle_C_max_value_of_a_add_b_l151_151691


namespace find_p_power_l151_151974

theorem find_p_power (p : ℕ) (h1 : p % 2 = 0) (h2 : (p + 1) % 10 = 7) : 
  (p % 10)^3 % 10 = (p % 10)^1 % 10 :=
by
  sorry

end find_p_power_l151_151974


namespace average_riding_speed_l151_151648

theorem average_riding_speed
  (initial_reading : ℕ) (final_reading : ℕ) (time_day1 : ℕ) (time_day2 : ℕ)
  (h_initial : initial_reading = 2332)
  (h_final : final_reading = 2552)
  (h_time_day1 : time_day1 = 5)
  (h_time_day2 : time_day2 = 4) :
  (final_reading - initial_reading) / (time_day1 + time_day2) = 220 / 9 :=
by
  sorry

end average_riding_speed_l151_151648


namespace quadratic_roots_l151_151640

theorem quadratic_roots : ∀ x : ℝ, (x^2 - 6 * x + 5 = 0) ↔ (x = 5 ∨ x = 1) :=
by sorry

end quadratic_roots_l151_151640


namespace solve_for_a_l151_151217

theorem solve_for_a (a x : ℝ) (h1 : 3 * x - 5 = x + a) (h2 : x = 2) : a = -1 :=
by
  sorry

end solve_for_a_l151_151217


namespace min_time_proof_l151_151469

/-
  Problem: 
  Given 5 colored lights that each can shine in one of the colors {red, orange, yellow, green, blue},
  and the colors are all different, and the interval between two consecutive flashes is 5 seconds.
  Define the ordered shining of these 5 lights once as a "flash", where each flash lasts 5 seconds.
  We need to show that the minimum time required to achieve all different flashes (120 flashes) is equal to 1195 seconds.
-/

def min_time_required : Nat :=
  let num_flashes := 5 * 4 * 3 * 2 * 1
  let flash_time := 5 * num_flashes
  let interval_time := 5 * (num_flashes - 1)
  flash_time + interval_time

theorem min_time_proof : min_time_required = 1195 := by
  sorry

end min_time_proof_l151_151469


namespace total_days_spent_on_island_l151_151315

noncomputable def first_expedition_weeks := 3
noncomputable def second_expedition_weeks := first_expedition_weeks + 2
noncomputable def last_expedition_weeks := 2 * second_expedition_weeks
noncomputable def total_weeks := first_expedition_weeks + second_expedition_weeks + last_expedition_weeks
noncomputable def total_days := 7 * total_weeks

theorem total_days_spent_on_island : total_days = 126 := by
  sorry

end total_days_spent_on_island_l151_151315


namespace smallest_k_value_for_screws_packs_l151_151792

theorem smallest_k_value_for_screws_packs :
  ∃ k : ℕ, k = 60 ∧ (∃ x y : ℕ, (k = 10 * x ∧ k = 12 * y) ∧ x ≠ y) := sorry

end smallest_k_value_for_screws_packs_l151_151792


namespace distinct_real_roots_of_quadratic_find_m_and_other_root_l151_151007

theorem distinct_real_roots_of_quadratic (m : ℝ) (h_neg_m : m < 0) : 
    ∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ (∀ x, x^2 - 2*x + m = 0 → (x = x₁ ∨ x = x₂))) := 
by 
  sorry

theorem find_m_and_other_root (m : ℝ) (h_neg_m : m < 0) (root_minus_one : ∀ x, x^2 - 2*x + m = 0 → x = -1):
    m = -3 ∧ (∃ x, x^2 - 2*x - 3 = 0 ∧ x = 3) := 
by 
  sorry

end distinct_real_roots_of_quadratic_find_m_and_other_root_l151_151007


namespace find_a_m_l151_151553

theorem find_a_m :
  ∃ a m : ℤ,
    (a = -2) ∧ (m = -1 ∨ m = 3) ∧ 
    (∀ x : ℝ, (a - 1) * x^2 + a * x + 1 = 0 → 
               (m^2 + m) * x^2 + 3 * m * x - 3 = 0) := sorry

end find_a_m_l151_151553


namespace population_control_l151_151559

   noncomputable def population_growth (initial_population : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ :=
   initial_population * (1 + growth_rate / 100) ^ years

   theorem population_control {initial_population : ℝ} {threshold_population : ℝ} {growth_rate : ℝ} {years : ℕ} :
     initial_population = 1.3 ∧ threshold_population = 1.4 ∧ growth_rate = 0.74 ∧ years = 10 →
     population_growth initial_population growth_rate years < threshold_population :=
   by
     intros
     sorry
   
end population_control_l151_151559


namespace square_side_length_l151_151876

theorem square_side_length (x : ℝ) 
  (h : x^2 = 6^2 + 8^2) : x = 10 := 
by sorry

end square_side_length_l151_151876


namespace ratio_of_falls_l151_151885

variable (SteveFalls : ℕ) (StephFalls : ℕ) (SonyaFalls : ℕ)
variable (H1 : SteveFalls = 3)
variable (H2 : StephFalls = SteveFalls + 13)
variable (H3 : SonyaFalls = 6)

theorem ratio_of_falls : SonyaFalls / (StephFalls / 2) = 3 / 4 := by
  sorry

end ratio_of_falls_l151_151885


namespace compute_expression_l151_151152

theorem compute_expression : 6^2 - 4 * 5 + 4^2 = 32 :=
by sorry

end compute_expression_l151_151152


namespace first_class_product_probability_l151_151122

theorem first_class_product_probability
  (defective_rate : ℝ) (first_class_rate_qualified : ℝ)
  (H_def_rate : defective_rate = 0.04)
  (H_first_class_rate_qualified : first_class_rate_qualified = 0.75) :
  (1 - defective_rate) * first_class_rate_qualified = 0.72 :=
by
  sorry

end first_class_product_probability_l151_151122


namespace inverse_matrix_equation_of_line_l_l151_151868

noncomputable def M : Matrix (Fin 2) (Fin 2) ℚ := ![![1, 2], ![3, 4]]
noncomputable def M_inv : Matrix (Fin 2) (Fin 2) ℚ := ![![-2, 1], ![3/2, -1/2]]

theorem inverse_matrix :
  M⁻¹ = M_inv :=
by
  sorry

def transformed_line (x y : ℚ) : Prop := 2 * (x + 2 * y) - (3 * x + 4 * y) = 4 

theorem equation_of_line_l (x y : ℚ) :
  transformed_line x y → x + 4 = 0 :=
by
  sorry

end inverse_matrix_equation_of_line_l_l151_151868


namespace minimum_value_of_16b_over_ac_l151_151151

noncomputable def minimum_16b_over_ac (a b c : ℝ) (A B C : ℝ) : ℝ :=
  if (0 < B) ∧ (B < Real.pi / 2) ∧
     (Real.cos B ^ 2 + (1 / 2) * Real.sin (2 * B) = 1) ∧
     ((Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) = 3)) then
    16 * b / (a * c)
  else 0

theorem minimum_value_of_16b_over_ac (a b c : ℝ) (A B C : ℝ)
  (h1 : 0 < B)
  (h2 : B < Real.pi / 2)
  (h3 : Real.cos B ^ 2 + (1 / 2) * Real.sin (2 * B) = 1)
  (h4 : Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) = 3) :
  minimum_16b_over_ac a b c A B C = 16 * (2 - Real.sqrt 2) / 3 := 
sorry

end minimum_value_of_16b_over_ac_l151_151151


namespace compare_neg_thirds_and_halves_l151_151353

theorem compare_neg_thirds_and_halves : (-1 : ℚ) / 3 > (-1 : ℚ) / 2 :=
by
  sorry

end compare_neg_thirds_and_halves_l151_151353


namespace tenth_digit_of_expression_l151_151451

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def tenth_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem tenth_digit_of_expression : 
  tenth_digit ((factorial 5 * factorial 5 - factorial 5 * factorial 3) / 5) = 3 :=
by
  -- proof omitted
  sorry

end tenth_digit_of_expression_l151_151451


namespace tan_x_eq_2_solution_l151_151937

noncomputable def solution_set_tan_2 : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.arctan 2}

theorem tan_x_eq_2_solution :
  {x : ℝ | Real.tan x = 2} = solution_set_tan_2 :=
by
  sorry

end tan_x_eq_2_solution_l151_151937


namespace multiple_of_1897_l151_151313

theorem multiple_of_1897 (n : ℕ) : ∃ k : ℤ, 2903^n - 803^n - 464^n + 261^n = k * 1897 := by
  sorry

end multiple_of_1897_l151_151313


namespace sin_sq_sub_cos_sq_l151_151875

-- Given condition
variable {α : ℝ}
variable (h : Real.sin α = Real.sqrt 5 / 5)

-- Proof goal
theorem sin_sq_sub_cos_sq (h : Real.sin α = Real.sqrt 5 / 5) : Real.sin α ^ 2 - Real.cos α ^ 2 = -3 / 5 := sorry

end sin_sq_sub_cos_sq_l151_151875


namespace arithmetic_sequence_problem_l151_151137

variable {a : ℕ → ℕ} -- Assuming a_n is a function from natural numbers to natural numbers

theorem arithmetic_sequence_problem (h1 : a 1 + a 2 = 10) (h2 : a 4 = a 3 + 2) :
  a 3 + a 4 = 18 :=
sorry

end arithmetic_sequence_problem_l151_151137


namespace monthly_income_l151_151704

variable {I : ℝ} -- George's monthly income

def donated_to_charity (I : ℝ) := 0.60 * I -- 60% of the income left
def paid_in_taxes (I : ℝ) := 0.75 * donated_to_charity I -- 75% of the remaining income after donation
def saved_for_future (I : ℝ) := 0.80 * paid_in_taxes I -- 80% of the remaining income after taxes
def expenses (I : ℝ) := saved_for_future I - 125 -- Remaining income after groceries and transportation expenses
def remaining_for_entertainment := 150 -- $150 left for entertainment and miscellaneous expenses

theorem monthly_income : I = 763.89 := 
by
  -- Using the conditions of the problem
  sorry

end monthly_income_l151_151704


namespace scout_earnings_weekend_l151_151912

-- Define the conditions
def base_pay_per_hour : ℝ := 10.00
def saturday_hours : ℝ := 6
def saturday_customers : ℝ := 5
def saturday_tip_per_customer : ℝ := 5.00
def sunday_hours : ℝ := 8
def sunday_customers_with_3_tip : ℝ := 5
def sunday_customers_with_7_tip : ℝ := 5
def sunday_tip_3_per_customer : ℝ := 3.00
def sunday_tip_7_per_customer : ℝ := 7.00
def overtime_multiplier : ℝ := 1.5

-- Statement to prove earnings for the weekend is $255.00
theorem scout_earnings_weekend : 
  (base_pay_per_hour * saturday_hours + saturday_customers * saturday_tip_per_customer) +
  (base_pay_per_hour * overtime_multiplier * sunday_hours + 
   sunday_customers_with_3_tip * sunday_tip_3_per_customer +
   sunday_customers_with_7_tip * sunday_tip_7_per_customer) = 255 :=
by
  sorry

end scout_earnings_weekend_l151_151912


namespace quadratic_inequality_solution_l151_151335

theorem quadratic_inequality_solution (x : ℝ) : 
  ((x - 1) * x ≥ 2) ↔ (x ≤ -1 ∨ x ≥ 2) := 
sorry

end quadratic_inequality_solution_l151_151335


namespace max_sum_of_arithmetic_sequence_l151_151427

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n : ℕ, n > 0 → 4 * a (n + 1) = 4 * a n - 7) →
  a 1 = 25 →
  (∀ n : ℕ, S n = (n * (50 - (7/4 : ℚ) * (n - 1))) / 2) →
  ∃ n : ℕ, n = 15 ∧ S n = 765 / 4 :=
by
  sorry

end max_sum_of_arithmetic_sequence_l151_151427


namespace intersection_point_of_planes_l151_151601

theorem intersection_point_of_planes :
  ∃ (x y z : ℚ), 
    3 * x - y + 4 * z = 2 ∧ 
    -3 * x + 4 * y - 3 * z = 4 ∧ 
    -x + y - z = 5 ∧ 
    x = -55 ∧ 
    y = -11 ∧ 
    z = 39 := 
by
  sorry

end intersection_point_of_planes_l151_151601


namespace students_enrolled_in_all_three_l151_151067

variables {total_students at_least_one robotics_students dance_students music_students at_least_two_students all_three_students : ℕ}

-- Given conditions
axiom H1 : total_students = 25
axiom H2 : at_least_one = total_students
axiom H3 : robotics_students = 15
axiom H4 : dance_students = 12
axiom H5 : music_students = 10
axiom H6 : at_least_two_students = 11

-- We need to prove the number of students enrolled in all three workshops is 1
theorem students_enrolled_in_all_three : all_three_students = 1 :=
sorry

end students_enrolled_in_all_three_l151_151067


namespace total_stamps_l151_151216

-- Definitions for the conditions.
def snowflake_stamps : ℕ := 11
def truck_stamps : ℕ := snowflake_stamps + 9
def rose_stamps : ℕ := truck_stamps - 13

-- Statement to prove the total number of stamps.
theorem total_stamps : (snowflake_stamps + truck_stamps + rose_stamps) = 38 :=
by
  sorry

end total_stamps_l151_151216


namespace triangle_side_length_l151_151443

/-
  Given a triangle ABC with sides |AB| = c, |AC| = b, and centroid G, incenter I,
  if GI is perpendicular to BC, then we need to prove that |BC| = (b+c)/2.
-/
variable {A B C G I : Type}
variable {AB AC BC : ℝ} -- Lengths of the sides
variable {b c : ℝ} -- Given lengths
variable {G_centroid : IsCentroid A B C G} -- G is the centroid of triangle ABC
variable {I_incenter : IsIncenter A B C I} -- I is the incenter of triangle ABC
variable {G_perp_BC : IsPerpendicular G I BC} -- G I ⊥ BC

theorem triangle_side_length (h1 : |AB| = c) (h2 : |AC| = b) :
  |BC| = (b + c) / 2 := 
sorry

end triangle_side_length_l151_151443


namespace find_n_l151_151367

theorem find_n (n : ℕ) (h : Nat.lcm n (n - 30) = n + 1320) : n = 165 := 
sorry

end find_n_l151_151367


namespace positive_integers_satisfy_condition_l151_151209

theorem positive_integers_satisfy_condition :
  ∃! n : ℕ, (n > 0 ∧ 30 - 6 * n > 18) :=
by
  sorry

end positive_integers_satisfy_condition_l151_151209


namespace parabola_vertex_coordinates_l151_151284

theorem parabola_vertex_coordinates :
  ∃ (x y : ℝ), y = (x - 2)^2 ∧ (x, y) = (2, 0) :=
sorry

end parabola_vertex_coordinates_l151_151284


namespace linear_dependence_condition_l151_151417

theorem linear_dependence_condition (k : ℝ) :
  (∃ (a b : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ (a * 1 + b * 4 = 0) ∧ (a * 2 + b * k = 0) ∧ (a * 1 + b * 2 = 0)) ↔ k = 8 := 
by sorry

end linear_dependence_condition_l151_151417


namespace min_sum_distances_to_corners_of_rectangle_center_l151_151933

theorem min_sum_distances_to_corners_of_rectangle_center (P A B C D : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (1, 0))
  (hC : C = (1, 1))
  (hD : D = (0, 1))
  (hP_center : P = (0.5, 0.5)) :
  ∀ Q, (dist Q A + dist Q B + dist Q C + dist Q D) ≥ (dist P A + dist P B + dist P C + dist P D) := 
sorry

end min_sum_distances_to_corners_of_rectangle_center_l151_151933


namespace train_speed_kmph_l151_151737

theorem train_speed_kmph (length time : ℝ) (h_length : length = 90) (h_time : time = 8.999280057595392) :
  (length / time) * 3.6 = 36.003 :=
by
  rw [h_length, h_time]
  norm_num
  sorry -- the norm_num tactic might simplify this enough, otherwise further steps would be added here.

end train_speed_kmph_l151_151737


namespace composite_expression_l151_151701

theorem composite_expression (n : ℕ) (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ 3^(2*n+1) - 2^(2*n+1) - 6^n = a * b :=
sorry

end composite_expression_l151_151701


namespace find_correct_average_of_numbers_l151_151075

variable (nums : List ℝ)
variable (n : ℕ) (avg_wrong avg_correct : ℝ) (wrong_val correct_val : ℝ)

noncomputable def correct_average (nums : List ℝ) (wrong_val correct_val : ℝ) : ℝ :=
  let correct_sum := nums.sum - wrong_val + correct_val
  correct_sum / nums.length

theorem find_correct_average_of_numbers
  (h₀ : n = 10)
  (h₁ : avg_wrong = 15)
  (h₂ : wrong_val = 26)
  (h₃ : correct_val = 36)
  (h₄ : avg_correct = 16)
  (nums : List ℝ) :
  avg_wrong * n - wrong_val + correct_val = avg_correct * n := 
sorry

end find_correct_average_of_numbers_l151_151075


namespace total_musicians_is_98_l151_151422

-- Define the number of males and females in the orchestra
def males_in_orchestra : ℕ := 11
def females_in_orchestra : ℕ := 12

-- Define the total number of musicians in the orchestra
def total_in_orchestra : ℕ := males_in_orchestra + females_in_orchestra

-- Define the number of musicians in the band as twice the number in the orchestra
def total_in_band : ℕ := 2 * total_in_orchestra

-- Define the number of males and females in the choir
def males_in_choir : ℕ := 12
def females_in_choir : ℕ := 17

-- Define the total number of musicians in the choir
def total_in_choir : ℕ := males_in_choir + females_in_choir

-- Prove that the total number of musicians in the orchestra, band, and choir is 98
theorem total_musicians_is_98 : total_in_orchestra + total_in_band + total_in_choir = 98 :=
by {
  -- Adding placeholders for the proof steps
  sorry
}

end total_musicians_is_98_l151_151422


namespace conversion_base8_to_base10_l151_151769

theorem conversion_base8_to_base10 : 
  (4 * 8^3 + 5 * 8^2 + 3 * 8^1 + 2 * 8^0) = 2394 := by 
  sorry

end conversion_base8_to_base10_l151_151769


namespace c_plus_d_is_even_l151_151729

-- Define the conditions
variables {c d : ℕ}
variables (m n : ℕ) (hc : c = 6 * m) (hd : d = 9 * n)

-- State the theorem to be proven
theorem c_plus_d_is_even : 
  (c = 6 * m) → (d = 9 * n) → Even (c + d) :=
by
  -- Proof steps would go here
  sorry

end c_plus_d_is_even_l151_151729


namespace root_quad_eqn_l151_151805

theorem root_quad_eqn (a : ℝ) (h : a^2 - a - 50 = 0) : a^3 - 51 * a = 50 :=
sorry

end root_quad_eqn_l151_151805


namespace boss_total_amount_l151_151061

def number_of_staff : ℕ := 20
def rate_per_day : ℕ := 100
def number_of_days : ℕ := 30
def petty_cash_amount : ℕ := 1000

theorem boss_total_amount (number_of_staff : ℕ) (rate_per_day : ℕ) (number_of_days : ℕ) (petty_cash_amount : ℕ) :
  let total_allowance_one_staff := rate_per_day * number_of_days
  let total_allowance_all_staff := total_allowance_one_staff * number_of_staff
  total_allowance_all_staff + petty_cash_amount = 61000 := by
  sorry

end boss_total_amount_l151_151061


namespace ninth_grade_students_eq_l151_151287

-- Let's define the conditions
def total_students : ℕ := 50
def seventh_grade_students (x : ℕ) : ℕ := 2 * x - 1
def eighth_grade_students (x : ℕ) : ℕ := x

-- Define the expression for ninth grade students based on the conditions
def ninth_grade_students (x : ℕ) : ℕ :=
  total_students - (seventh_grade_students x + eighth_grade_students x)

-- The theorem statement to prove
theorem ninth_grade_students_eq (x : ℕ) : ninth_grade_students x = 51 - 3 * x :=
by
  sorry

end ninth_grade_students_eq_l151_151287


namespace Kim_sales_on_Friday_l151_151663

theorem Kim_sales_on_Friday (tuesday_sales : ℕ) (tuesday_discount_rate : ℝ) 
    (monday_increase_rate : ℝ) (wednesday_increase_rate : ℝ) 
    (thursday_decrease_rate : ℝ) (friday_increase_rate : ℝ) 
    (final_friday_sales : ℕ) :
    tuesday_sales = 800 →
    tuesday_discount_rate = 0.05 →
    monday_increase_rate = 0.50 →
    wednesday_increase_rate = 1.5 →
    thursday_decrease_rate = 0.20 →
    friday_increase_rate = 1.3 →
    final_friday_sales = 1310 :=
by
  sorry

end Kim_sales_on_Friday_l151_151663


namespace solution_set_of_inequality_system_l151_151374

theorem solution_set_of_inequality_system (x : ℝ) :
  (x + 1 ≥ 0) ∧ (x - 2 < 0) ↔ (-1 ≤ x ∧ x < 2) :=
by
  sorry

end solution_set_of_inequality_system_l151_151374


namespace hyperbola_eccentricity_l151_151679

theorem hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b = a) 
  (h₄ : ∀ c, (c = Real.sqrt (a^2 + b^2)) → (b * c / Real.sqrt (a^2 + b^2) = a)) :
  (Real.sqrt (2) = (c / a)) :=
by
  sorry

end hyperbola_eccentricity_l151_151679


namespace solve_for_y_l151_151832

theorem solve_for_y (y : ℝ) (h : (y / 5) / 3 = 5 / (y / 3)) : y = 15 ∨ y = -15 :=
by sorry

end solve_for_y_l151_151832


namespace two_pow_n_minus_one_prime_imp_n_prime_l151_151027

theorem two_pow_n_minus_one_prime_imp_n_prime (n : ℕ) (h : Nat.Prime (2^n - 1)) : Nat.Prime n := 
sorry

end two_pow_n_minus_one_prime_imp_n_prime_l151_151027


namespace reciprocal_expression_equals_two_l151_151550

theorem reciprocal_expression_equals_two (x y : ℝ) (h : x * y = 1) : 
  (x + 1 / y) * (2 * y - 1 / x) = 2 := by
  sorry

end reciprocal_expression_equals_two_l151_151550


namespace find_dividend_l151_151514

theorem find_dividend (k : ℕ) (quotient : ℕ) (dividend : ℕ) (h1 : k = 14) (h2 : quotient = 4) (h3 : dividend = quotient * k) : dividend = 56 :=
by
  sorry

end find_dividend_l151_151514


namespace range_of_x_l151_151300

theorem range_of_x (x : ℝ) (h1 : 1/x < 3) (h2 : 1/x > -2) :
  x > 1/3 ∨ x < -1/2 :=
sorry

end range_of_x_l151_151300


namespace set_equivalence_l151_151031

-- Define the given set using the condition.
def given_set : Set ℕ := {x | x ∈ {x | 0 < x} ∧ x - 3 < 2}

-- Define the enumerated set.
def enumerated_set : Set ℕ := {1, 2, 3, 4}

-- Statement of the proof problem.
theorem set_equivalence : given_set = enumerated_set :=
by
  -- The proof is omitted
  sorry

end set_equivalence_l151_151031


namespace fewer_hours_worked_l151_151273

noncomputable def total_earnings_summer := 6000
noncomputable def total_weeks_summer := 10
noncomputable def hours_per_week_summer := 50
noncomputable def total_earnings_school_year := 8000
noncomputable def total_weeks_school_year := 40

noncomputable def hourly_wage := total_earnings_summer / (hours_per_week_summer * total_weeks_summer)
noncomputable def total_hours_school_year := total_earnings_school_year / hourly_wage
noncomputable def hours_per_week_school_year := total_hours_school_year / total_weeks_school_year
noncomputable def fewer_hours_per_week := hours_per_week_summer - hours_per_week_school_year

theorem fewer_hours_worked :
  fewer_hours_per_week = hours_per_week_summer - (total_earnings_school_year / hourly_wage / total_weeks_school_year) := by
  sorry

end fewer_hours_worked_l151_151273


namespace eight_percent_is_64_l151_151823

-- Definition of the condition
variable (x : ℝ)

-- The theorem that states the problem to be proven
theorem eight_percent_is_64 (h : (8 / 100) * x = 64) : x = 800 :=
sorry

end eight_percent_is_64_l151_151823


namespace square_not_end_with_four_identical_digits_l151_151546

theorem square_not_end_with_four_identical_digits (n : ℕ) (d : ℕ) :
  n = d * d → ¬ (d ≠ 0 ∧ (n % 10000 = d ^ 4)) :=
by
  sorry

end square_not_end_with_four_identical_digits_l151_151546


namespace grandpa_max_pieces_l151_151751

theorem grandpa_max_pieces (m n : ℕ) (h : (m - 3) * (n - 3) = 9) : m * n = 112 :=
sorry

end grandpa_max_pieces_l151_151751


namespace find_divisor_l151_151694

theorem find_divisor (d : ℕ) (h1 : 2319 % d = 0) (h2 : 2304 % d = 0) (h3 : (2319 - 2304) % d = 0) : d = 3 :=
  sorry

end find_divisor_l151_151694


namespace directrix_of_given_parabola_l151_151239

noncomputable def parabola_directrix (y : ℝ) : Prop :=
  let focus_x : ℝ := -1
  let point_on_parabola : ℝ × ℝ := (-1 / 4 * y^2, y)
  let PF_sq := (point_on_parabola.1 - focus_x)^2 + (point_on_parabola.2)^2
  let PQ_sq := (point_on_parabola.1 - 1)^2
  PF_sq = PQ_sq

theorem directrix_of_given_parabola : parabola_directrix y :=
sorry

end directrix_of_given_parabola_l151_151239


namespace min_value_of_reciprocal_sum_l151_151660

theorem min_value_of_reciprocal_sum (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + y = 2) : 
  ∃ (z : ℝ), z = (1 / x + 1 / y) ∧ z = (3 / 2 + Real.sqrt 2) :=
sorry

end min_value_of_reciprocal_sum_l151_151660


namespace primes_with_large_gap_exists_l151_151853

noncomputable def exists_primes_with_large_gap_and_composites_between : Prop :=
  ∃ p q : ℕ, p < q ∧ Nat.Prime p ∧ Nat.Prime q ∧ q - p > 2015 ∧ (∀ n : ℕ, p < n ∧ n < q → ¬Nat.Prime n)

theorem primes_with_large_gap_exists : exists_primes_with_large_gap_and_composites_between := sorry

end primes_with_large_gap_exists_l151_151853


namespace solution_set_of_inequality_l151_151167

theorem solution_set_of_inequality :
  { x : ℝ | x^2 - 5 * x + 6 ≤ 0 } = { x : ℝ | 2 ≤ x ∧ x ≤ 3 } :=
sorry

end solution_set_of_inequality_l151_151167


namespace graph_passes_through_point_l151_151039

theorem graph_passes_through_point : ∀ (a : ℝ), a > 0 ∧ a ≠ 1 → (∃ x y, (x, y) = (0, 2) ∧ y = a^x + 1) :=
by
  intros a ha
  use 0
  use 2
  obtain ⟨ha1, ha2⟩ := ha
  have h : a^0 = 1 := by simp
  simp [h]
  sorry

end graph_passes_through_point_l151_151039


namespace parallelogram_proof_l151_151101

noncomputable def parallelogram_ratio (AP AB AQ AD AC AT : ℝ) (hP : AP / AB = 61 / 2022) (hQ : AQ / AD = 61 / 2065) (h_intersect : true) : ℕ :=
if h : AC / AT = 4087 / 61 then 67 else 0

theorem parallelogram_proof :
  ∀ (ABCD : Type) (P : Type) (Q : Type) (T : Type) 
     (AP AB AQ AD AC AT : ℝ) 
     (hP : AP / AB = 61 / 2022) 
     (hQ : AQ / AD = 61 / 2065)
     (h_intersect : true),
  parallelogram_ratio AP AB AQ AD AC AT hP hQ h_intersect = 67 :=
by
  sorry

end parallelogram_proof_l151_151101


namespace parallel_lines_slope_equal_l151_151542

theorem parallel_lines_slope_equal (m : ℝ) : 
  (∃ m : ℝ, -(m+4)/(m+2) = -(m+2)/(m+1)) → m = 0 := 
by
  sorry

end parallel_lines_slope_equal_l151_151542


namespace decreasing_interval_l151_151786

noncomputable def f (x : ℝ) := x^2 * Real.exp x

theorem decreasing_interval : ∀ x : ℝ, x > -2 ∧ x < 0 → deriv f x < 0 := 
by
  intro x h
  sorry

end decreasing_interval_l151_151786


namespace sum_of_b_for_one_solution_l151_151305

theorem sum_of_b_for_one_solution :
  let A := 3
  let C := 12
  ∀ b : ℝ, ((b + 5)^2 - 4 * A * C = 0) → (b = 7 ∨ b = -17) → (7 + (-17)) = -10 :=
by
  intro A C b
  sorry

end sum_of_b_for_one_solution_l151_151305


namespace find_distance_between_foci_l151_151345

noncomputable def distance_between_foci (pts : List (ℝ × ℝ)) : ℝ :=
  let c := (1, -1)  -- center of the ellipse
  let x1 := (1, 3)
  let x2 := (1, -5)
  let y := (7, -5)
  let b := 4       -- semi-minor axis length
  let a := 2 * Real.sqrt 13  -- semi-major axis length
  let foci_distance := 2 * Real.sqrt (a^2 - b^2)
  foci_distance

theorem find_distance_between_foci :
  distance_between_foci [(1, 3), (7, -5), (1, -5)] = 12 :=
by
  sorry

end find_distance_between_foci_l151_151345


namespace sin_identity_l151_151891

variable (α : ℝ)
axiom alpha_def : α = Real.pi / 7

theorem sin_identity : (Real.sin (3 * α)) ^ 2 - (Real.sin α) ^ 2 = Real.sin (2 * α) * Real.sin (3 * α) := 
by 
  sorry

end sin_identity_l151_151891


namespace sandy_receives_correct_change_l151_151923

-- Define the costs of each item
def cost_cappuccino : ℕ := 2
def cost_iced_tea : ℕ := 3
def cost_cafe_latte : ℝ := 1.5
def cost_espresso : ℕ := 1

-- Define the quantities ordered
def qty_cappuccino : ℕ := 3
def qty_iced_tea : ℕ := 2
def qty_cafe_latte : ℕ := 2
def qty_espresso : ℕ := 2

-- Calculate the total cost
def total_cost : ℝ := (qty_cappuccino * cost_cappuccino) + 
                      (qty_iced_tea * cost_iced_tea) + 
                      (qty_cafe_latte * cost_cafe_latte) + 
                      (qty_espresso * cost_espresso)

-- Define the amount paid
def amount_paid : ℝ := 20

-- Calculate the change
def change : ℝ := amount_paid - total_cost

theorem sandy_receives_correct_change : change = 3 := by
  -- Detailed steps would go here
  sorry

end sandy_receives_correct_change_l151_151923


namespace rectangular_plot_area_l151_151406

/-- The ratio between the length and the breadth of a rectangular plot is 7 : 5.
    If the perimeter of the plot is 288 meters, then the area of the plot is 5040 square meters.
-/
theorem rectangular_plot_area
    (L B : ℝ)
    (h1 : L / B = 7 / 5)
    (h2 : 2 * (L + B) = 288) :
    L * B = 5040 :=
by
  sorry

end rectangular_plot_area_l151_151406


namespace find_product_l151_151368

theorem find_product (a b c d : ℚ) 
  (h₁ : 2 * a + 4 * b + 6 * c + 8 * d = 48)
  (h₂ : 4 * (d + c) = b)
  (h₃ : 4 * b + 2 * c = a)
  (h₄ : c + 1 = d) :
  a * b * c * d = -319603200 / 10503489 := sorry

end find_product_l151_151368


namespace price_of_book_l151_151899

-- Definitions based on the problem conditions
def money_xiaowang_has (p : ℕ) : ℕ := 2 * p - 6
def money_xiaoli_has (p : ℕ) : ℕ := 2 * p - 31

def combined_money (p : ℕ) : ℕ := money_xiaowang_has p + money_xiaoli_has p

-- Lean statement to prove the price of each book
theorem price_of_book (p : ℕ) : combined_money p = 3 * p → p = 37 :=
by
  sorry

end price_of_book_l151_151899


namespace oranges_in_bin_l151_151055

theorem oranges_in_bin (initial_oranges : ℕ) (oranges_thrown_away : ℕ) (oranges_added : ℕ) 
  (h1 : initial_oranges = 50) (h2 : oranges_thrown_away = 40) (h3 : oranges_added = 24) 
  : initial_oranges - oranges_thrown_away + oranges_added = 34 := 
by
  -- Simplification and calculation here
  sorry

end oranges_in_bin_l151_151055


namespace tan_double_angle_l151_151158

open Real

-- Given condition
def condition (x : ℝ) : Prop := tan x - 1 / tan x = 3 / 2

-- Main theorem to prove
theorem tan_double_angle (x : ℝ) (h : condition x) : tan (2 * x) = -4 / 3 := by
  sorry

end tan_double_angle_l151_151158


namespace passengers_on_ship_l151_151056

theorem passengers_on_ship :
  (∀ P : ℕ, 
    (P / 12) + (P / 8) + (P / 3) + (P / 6) + 35 = P) → P = 120 :=
by 
  sorry

end passengers_on_ship_l151_151056


namespace find_k_l151_151271

-- Define the functions f and g
def f (x : ℝ) : ℝ := 4 * x^2 + 3 * x + 5
def g (x : ℝ) (k : ℝ) : ℝ := x^2 + k * x - 7

-- Define the given condition f(5) - g(5) = 20
def condition (k : ℝ) : Prop := f 5 - g 5 k = 20

-- The theorem to prove that k = 16.4
theorem find_k : ∃ k : ℝ, condition k ∧ k = 16.4 :=
by
  sorry

end find_k_l151_151271


namespace earth_surface_inhabitable_fraction_l151_151887

theorem earth_surface_inhabitable_fraction :
  (1 / 3 : ℝ) * (2 / 3 : ℝ) = 2 / 9 := 
by 
  sorry

end earth_surface_inhabitable_fraction_l151_151887


namespace friends_count_l151_151254

def bananas_total : ℝ := 63
def bananas_per_friend : ℝ := 21.0

theorem friends_count : bananas_total / bananas_per_friend = 3 := sorry

end friends_count_l151_151254


namespace min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l151_151269

theorem min_arithmetic_mean_of_consecutive_naturals_div_by_1111 :
  ∀ a : ℕ, (∃ k, (a * (a + 1) * (a + 2) * (a + 3) * (a + 4) * (a + 5) * (a + 6) * (a + 7) * (a + 8)) = 1111 * k) →
  (a + 4 ≥ 97) :=
sorry

end min_arithmetic_mean_of_consecutive_naturals_div_by_1111_l151_151269


namespace isosceles_triangle_perimeter_l151_151952

noncomputable def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

def roots_of_quadratic_eq := {x : ℕ | x^2 - 5 * x + 6 = 0}

theorem isosceles_triangle_perimeter
  (a b c : ℕ)
  (h_isosceles : is_isosceles_triangle a b c)
  (h_roots : (a ∈ roots_of_quadratic_eq) ∧ (b ∈ roots_of_quadratic_eq) ∧ (c ∈ roots_of_quadratic_eq)) :
  (a + b + c = 7 ∨ a + b + c = 8) :=
by
  sorry

end isosceles_triangle_perimeter_l151_151952


namespace points_within_distance_5_l151_151054

noncomputable def distance (x y z : ℝ) : ℝ := Real.sqrt (x^2 + y^2 + z^2)

def within_distance (x y z : ℝ) (d : ℝ) : Prop := distance x y z ≤ d

def A := (1, 1, 1)
def B := (1, 2, 2)
def C := (2, -3, 5)
def D := (3, 0, 4)

theorem points_within_distance_5 :
  within_distance 1 1 1 5 ∧
  within_distance 1 2 2 5 ∧
  ¬ within_distance 2 (-3) 5 5 ∧
  within_distance 3 0 4 5 :=
by {
  sorry
}

end points_within_distance_5_l151_151054


namespace lightest_ball_box_is_blue_l151_151872

-- Define the weights and counts of balls
def yellow_ball_weight : ℕ := 50
def yellow_ball_count_per_box : ℕ := 50
def white_ball_weight : ℕ := 45
def white_ball_count_per_box : ℕ := 60
def blue_ball_weight : ℕ := 55
def blue_ball_count_per_box : ℕ := 40

-- Calculate the total weight of balls per type
def yellow_box_weight : ℕ := yellow_ball_weight * yellow_ball_count_per_box
def white_box_weight : ℕ := white_ball_weight * white_ball_count_per_box
def blue_box_weight : ℕ := blue_ball_weight * blue_ball_count_per_box

theorem lightest_ball_box_is_blue :
  (blue_box_weight < yellow_box_weight) ∧ (blue_box_weight < white_box_weight) :=
by
  -- Proof can go here
  sorry

end lightest_ball_box_is_blue_l151_151872


namespace raking_yard_time_l151_151477

theorem raking_yard_time (your_rate : ℚ) (brother_rate : ℚ) (combined_rate : ℚ) (combined_time : ℚ) :
  your_rate = 1 / 30 ∧ 
  brother_rate = 1 / 45 ∧ 
  combined_rate = your_rate + brother_rate ∧ 
  combined_time = 1 / combined_rate → 
  combined_time = 18 := 
by 
  sorry

end raking_yard_time_l151_151477


namespace solve_for_x_l151_151081

theorem solve_for_x (x : ℝ) (h : 2 * x - 5 = 15) : x = 10 :=
sorry

end solve_for_x_l151_151081


namespace A_speed_is_10_l151_151608

noncomputable def A_walking_speed (v t : ℝ) := 
  v * (t + 7) = 140 ∧ v * (t + 7) = 20 * t

theorem A_speed_is_10 (v t : ℝ) 
  (h1 : v * (t + 7) = 140)
  (h2 : v * (t + 7) = 20 * t) :
  v = 10 :=
sorry

end A_speed_is_10_l151_151608


namespace income_calculation_l151_151877

-- Define the conditions
def ratio (i e : ℕ) : Prop := 9 * e = 8 * i
def savings (i e : ℕ) : Prop := i - e = 4000

-- The theorem statement
theorem income_calculation (i e : ℕ) (h1 : ratio i e) (h2 : savings i e) : i = 36000 := by
  sorry

end income_calculation_l151_151877


namespace solve_recurrence_relation_l151_151699

noncomputable def a_n (n : ℕ) : ℝ := 2 * 4^n - 2 * n + 2
noncomputable def b_n (n : ℕ) : ℝ := 2 * 4^n + 2 * n - 2

theorem solve_recurrence_relation :
  a_n 0 = 4 ∧ b_n 0 = 0 ∧
  (∀ n : ℕ, a_n (n + 1) = 3 * a_n n + b_n n - 4) ∧
  (∀ n : ℕ, b_n (n + 1) = 2 * a_n n + 2 * b_n n + 2) :=
by
  sorry

end solve_recurrence_relation_l151_151699


namespace integer_solution_l151_151085

theorem integer_solution (n : ℤ) (h1 : n + 15 > 16) (h2 : -3 * n > -9) : n = 2 :=
by
  sorry

end integer_solution_l151_151085


namespace proportional_b_value_l151_151768

theorem proportional_b_value (b : ℚ) : (∃ k : ℚ, k ≠ 0 ∧ (∀ x : ℚ, x + 2 - 3 * b = k * x)) ↔ b = 2 / 3 :=
by
  sorry

end proportional_b_value_l151_151768


namespace relationship_among_abc_l151_151919

noncomputable
def a := 0.2 ^ 1.5

noncomputable
def b := 2 ^ 0.1

noncomputable
def c := 0.2 ^ 1.3

theorem relationship_among_abc : a < c ∧ c < b := by
  sorry

end relationship_among_abc_l151_151919


namespace find_a_n_l151_151326

variable (a : ℕ → ℝ)
axiom a_pos : ∀ n, a n > 0
axiom a1_eq : a 1 = 1
axiom rec_relation : ∀ n, a n * (n * a n - a (n + 1)) = (n + 1) * (a (n + 1)) ^ 2

theorem find_a_n : ∀ n, a n = 1 / n := by
  sorry

end find_a_n_l151_151326


namespace find_unknown_number_l151_151833

def unknown_number (x : ℝ) : Prop :=
  (0.5^3) - (0.1^3 / 0.5^2) + x + (0.1^2) = 0.4

theorem find_unknown_number : ∃ (x : ℝ), unknown_number x ∧ x = 0.269 :=
by
  sorry

end find_unknown_number_l151_151833


namespace alcohol_percentage_in_original_solution_l151_151263

theorem alcohol_percentage_in_original_solution
  (P : ℚ)
  (alcohol_in_new_mixture : ℚ)
  (original_solution_volume : ℚ)
  (added_water_volume : ℚ)
  (new_mixture_volume : ℚ)
  (percentage_in_new_mixture : ℚ) :
  original_solution_volume = 11 →
  added_water_volume = 3 →
  new_mixture_volume = original_solution_volume + added_water_volume →
  percentage_in_new_mixture = 33 →
  alcohol_in_new_mixture = (percentage_in_new_mixture / 100) * new_mixture_volume →
  (P / 100) * original_solution_volume = alcohol_in_new_mixture →
  P = 42 :=
by
  sorry

end alcohol_percentage_in_original_solution_l151_151263


namespace solve_proof_problem_l151_151053

variables (a b c d : ℝ)

noncomputable def proof_problem : Prop :=
  a = 3 * b ∧ b = 3 * c ∧ c = 5 * d → (a * c) / (b * d) = 15

theorem solve_proof_problem : proof_problem a b c d :=
by
  sorry

end solve_proof_problem_l151_151053


namespace find_coordinates_of_C_l151_151801

structure Point where
  x : Int
  y : Int

def isSymmetricalAboutXAxis (A B : Point) : Prop :=
  A.x = B.x ∧ A.y = -B.y

def isSymmetricalAboutOrigin (B C : Point) : Prop :=
  C.x = -B.x ∧ C.y = -B.y

theorem find_coordinates_of_C :
  ∃ C : Point, let A := Point.mk 2 (-3)
               let B := Point.mk 2 3
               isSymmetricalAboutXAxis A B →
               isSymmetricalAboutOrigin B C →
               C = Point.mk (-2) (-3) :=
by
  sorry

end find_coordinates_of_C_l151_151801


namespace symmetric_points_origin_l151_151973

theorem symmetric_points_origin (a b : ℤ) (h1 : a = -5) (h2 : b = -1) : a - b = -4 :=
by
  sorry

end symmetric_points_origin_l151_151973


namespace hyperbola_foci_problem_l151_151733

noncomputable def hyperbola (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1

noncomputable def foci_1 : ℝ × ℝ := (-Real.sqrt 5, 0)
noncomputable def foci_2 : ℝ × ℝ := (Real.sqrt 5, 0)

noncomputable def point_on_hyperbola (P : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2

noncomputable def vector (A B : ℝ × ℝ) : ℝ × ℝ :=
  (B.1 - A.1, B.2 - A.2)

noncomputable def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v1.1 + v2.2 * v2.2

noncomputable def orthogonal (P : ℝ × ℝ) : Prop :=
  dot_product (vector P foci_1) (vector P foci_2) = 0

noncomputable def distance (A B : ℝ × ℝ) : ℝ :=
  Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

noncomputable def required_value (P : ℝ × ℝ) : ℝ :=
  distance P foci_1 * distance P foci_2

theorem hyperbola_foci_problem (P : ℝ × ℝ) : 
  point_on_hyperbola P → orthogonal P → required_value P = 2 := 
sorry

end hyperbola_foci_problem_l151_151733


namespace shifted_linear_function_correct_l151_151139

def original_function (x : ℝ) : ℝ := 5 * x - 8
def shifted_function (x : ℝ) : ℝ := original_function x + 4

theorem shifted_linear_function_correct (x : ℝ) :
  shifted_function x = 5 * x - 4 :=
by
  sorry

end shifted_linear_function_correct_l151_151139


namespace Jessica_cut_40_roses_l151_151361

-- Define the problem's conditions as variables
variables (initialVaseRoses : ℕ) (finalVaseRoses : ℕ) (rosesGivenToSarah : ℕ)

-- Define the number of roses Jessica cut from her garden
def rosesCutFromGarden (initialVaseRoses finalVaseRoses rosesGivenToSarah : ℕ) : ℕ :=
  (finalVaseRoses - initialVaseRoses) + rosesGivenToSarah

-- Problem statement: Prove Jessica cut 40 roses from her garden
theorem Jessica_cut_40_roses (initialVaseRoses finalVaseRoses rosesGivenToSarah : ℕ) :
  initialVaseRoses = 7 →
  finalVaseRoses = 37 →
  rosesGivenToSarah = 10 →
  rosesCutFromGarden initialVaseRoses finalVaseRoses rosesGivenToSarah = 40 :=
by
  intros h1 h2 h3
  sorry

end Jessica_cut_40_roses_l151_151361


namespace gcd_231_154_l151_151474

def find_gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_231_154 : find_gcd 231 154 = 77 := by
  sorry

end gcd_231_154_l151_151474


namespace selection_of_projects_l151_151866

-- Mathematical definitions
def numberOfWaysToSelect2ProjectsFrom4KeyAnd6General (key: Finset ℕ) (general: Finset ℕ) : ℕ :=
  (key.card.choose 2) * (general.card.choose 2)

def numberOfWaysToSelectAtLeastOneProjectAorB (key: Finset ℕ) (general: Finset ℕ) (A B: ℕ) : ℕ :=
  let total_ways := (key.card.choose 2) * (general.card.choose 2)
  let ways_without_A := ((key.erase A).card.choose 2) * (general.card.choose 2)
  let ways_without_B := (key.card.choose 2) * ((general.erase B).card.choose 2)
  let ways_without_A_and_B := ((key.erase A).card.choose 2) * ((general.erase B).card.choose 2)
  total_ways - ways_without_A_and_B

-- Theorem we need to prove
theorem selection_of_projects (key general: Finset ℕ) (A B: ℕ) (hA: A ∈ key) (hB: B ∈ general) (h_key_card: key.card = 4) (h_general_card: general.card = 6) :
  numberOfWaysToSelectAtLeastOneProjectAorB key general A B = 60 := 
sorry

end selection_of_projects_l151_151866


namespace fourth_term_geometric_sequence_l151_151028

theorem fourth_term_geometric_sequence :
  let a := (6: ℝ)^(1/2)
  let b := (6: ℝ)^(1/6)
  let c := (6: ℝ)^(1/12)
  b = a * r ∧ c = a * r^2 → (a * r^3) = 1 := 
by
  sorry

end fourth_term_geometric_sequence_l151_151028


namespace hexagon_area_of_circle_l151_151213

noncomputable def radius (area : ℝ) : ℝ :=
  Real.sqrt (area / Real.pi)

noncomputable def area_equilateral_triangle (s : ℝ) : ℝ :=
  (s * s * Real.sqrt 3) / 4

theorem hexagon_area_of_circle {r : ℝ} (h : π * r^2 = 100 * π) :
  6 * area_equilateral_triangle r = 150 * Real.sqrt 3 :=
by
  sorry

end hexagon_area_of_circle_l151_151213


namespace rotated_triangle_surface_area_l151_151944

theorem rotated_triangle_surface_area :
  ∀ (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
    (ACLength : ℝ) (BCLength : ℝ) (right_angle : ℝ -> ℝ -> ℝ -> Prop)
    (pi_def : Real) (surface_area : ℝ -> ℝ -> ℝ),
    (right_angle 90 0 90) → (ACLength = 3) → (BCLength = 4) →
    surface_area ACLength BCLength = 24 * pi_def  :=
by
  sorry

end rotated_triangle_surface_area_l151_151944


namespace find_C_in_terms_of_D_l151_151063

noncomputable def h (C D x : ℝ) : ℝ := C * x - 3 * D ^ 2
noncomputable def k (D x : ℝ) : ℝ := D * x + 1

theorem find_C_in_terms_of_D (C D : ℝ) (h_eq : h C D (k D 2) = 0) (h_def : ∀ x, h C D x = C * x - 3 * D ^ 2) (k_def : ∀ x, k D x = D * x + 1) (D_ne_neg1 : D ≠ -1) : 
C = (3 * D ^ 2) / (2 * D + 1) := 
by 
  sorry

end find_C_in_terms_of_D_l151_151063


namespace find_x_l151_151141

theorem find_x (x : ℤ) (h : (1 + 2 + 4 + 5 + 6 + 9 + 9 + 10 + 12 + x) / 10 = 7) : x = 12 :=
by
  sorry

end find_x_l151_151141


namespace fractions_correct_l151_151024
-- Broader import to ensure all necessary libraries are included.

-- Definitions of the conditions
def batman_homes_termite_ridden : ℚ := 1/3
def batman_homes_collapsing : ℚ := 7/10 * batman_homes_termite_ridden
def robin_homes_termite_ridden : ℚ := 3/7
def robin_homes_collapsing : ℚ := 4/5 * robin_homes_termite_ridden
def joker_homes_termite_ridden : ℚ := 1/2
def joker_homes_collapsing : ℚ := 3/8 * joker_homes_termite_ridden

-- Definitions of the fractions of homes that are termite-ridden but not collapsing
def batman_non_collapsing_fraction : ℚ := batman_homes_termite_ridden - batman_homes_collapsing
def robin_non_collapsing_fraction : ℚ := robin_homes_termite_ridden - robin_homes_collapsing
def joker_non_collapsing_fraction : ℚ := joker_homes_termite_ridden - joker_homes_collapsing

-- Proof statement
theorem fractions_correct :
  batman_non_collapsing_fraction = 1/10 ∧
  robin_non_collapsing_fraction = 3/35 ∧
  joker_non_collapsing_fraction = 5/16 :=
sorry

end fractions_correct_l151_151024


namespace monotonic_increasing_interval_l151_151939

theorem monotonic_increasing_interval : ∀ x : ℝ, (x > 2) → ((x-3) * Real.exp x > 0) :=
sorry

end monotonic_increasing_interval_l151_151939


namespace largest_possible_perimeter_l151_151880

theorem largest_possible_perimeter (y : ℤ) (hy1 : 3 ≤ y) (hy2 : y < 16) : 7 + 9 + y ≤ 31 := 
by
  sorry

end largest_possible_perimeter_l151_151880


namespace second_difference_is_quadratic_l151_151072

theorem second_difference_is_quadratic (f : ℕ → ℝ) 
  (h : ∀ n : ℕ, (f (n + 2) - 2 * f (n + 1) + f n) = 2) :
  ∃ (a b : ℝ), ∀ (n : ℕ), f n = n^2 + a * n + b :=
by
  sorry

end second_difference_is_quadratic_l151_151072


namespace shaded_area_fraction_l151_151252

/-- The fraction of the larger square's area that is inside the shaded rectangle 
    formed by the points (2,2), (3,2), (3,5), and (2,5) on a 6 by 6 grid 
    is 1/12. -/
theorem shaded_area_fraction : 
  let grid_size := 6
  let rectangle_points := [(2, 2), (3, 2), (3, 5), (2, 5)]
  let rectangle_length := 1
  let rectangle_height := 3
  let rectangle_area := rectangle_length * rectangle_height
  let square_area := grid_size^2
  rectangle_area / square_area = 1 / 12 := 
by 
  sorry

end shaded_area_fraction_l151_151252


namespace remainder_div_3973_28_l151_151606

theorem remainder_div_3973_28 : (3973 % 28) = 9 := by
  sorry

end remainder_div_3973_28_l151_151606


namespace gcd_180_126_l151_151871

theorem gcd_180_126 : Nat.gcd 180 126 = 18 := by
  sorry

end gcd_180_126_l151_151871


namespace nate_total_run_l151_151537

def field_length := 168
def initial_run := 4 * field_length
def additional_run := 500
def total_run := initial_run + additional_run

theorem nate_total_run : total_run = 1172 := by
  sorry

end nate_total_run_l151_151537


namespace part1_part2_l151_151327

noncomputable def a : ℝ := 2 + Real.sqrt 3
noncomputable def b : ℝ := 2 - Real.sqrt 3

theorem part1 : a * b = 1 := 
by 
  unfold a b
  sorry

theorem part2 : a^2 + b^2 - a * b = 13 :=
by 
  unfold a b
  sorry

end part1_part2_l151_151327


namespace other_root_of_quadratic_l151_151066

theorem other_root_of_quadratic (p x : ℝ) (h : 7 * x^2 + p * x - 9 = 0) (root1 : x = -3) : 
  x = 3 / 7 :=
by
  sorry

end other_root_of_quadratic_l151_151066


namespace hypotenuse_is_18_8_l151_151337

def right_triangle_hypotenuse_perimeter_area (a b c : ℝ) : Prop :=
  a + b + c = 40 ∧ (1/2 * a * b = 24) ∧ (a^2 + b^2 = c^2)

theorem hypotenuse_is_18_8 : ∃ (a b c : ℝ), right_triangle_hypotenuse_perimeter_area a b c ∧ c = 18.8 :=
by
  sorry

end hypotenuse_is_18_8_l151_151337


namespace camping_trip_percentage_l151_151698

theorem camping_trip_percentage (T : ℝ)
  (h1 : 16 / 100 ≤ 1)
  (h2 : T - 16 / 100 ≤ 1)
  (h3 : T = 64 / 100) :
  T = 64 / 100 := by
  sorry

end camping_trip_percentage_l151_151698


namespace min_xy_l151_151830

variable {x y : ℝ}

theorem min_xy (hx : x > 0) (hy : y > 0) (h : 10 * x + 2 * y + 60 = x * y) : x * y ≥ 180 := 
sorry

end min_xy_l151_151830


namespace quadratic_eq_m_neg1_l151_151094

theorem quadratic_eq_m_neg1 (m : ℝ) (h1 : (m - 3) ≠ 0) (h2 : m^2 - 2*m - 3 = 0) : m = -1 :=
sorry

end quadratic_eq_m_neg1_l151_151094


namespace seventeen_in_base_three_l151_151983

theorem seventeen_in_base_three : (17 : ℕ) = 1 * 3^2 + 2 * 3^1 + 2 * 3^0 :=
by
  -- This is the arithmetic representation of the conversion,
  -- proving that 17 in base 10 equals 122 in base 3
  sorry

end seventeen_in_base_three_l151_151983


namespace diet_soda_bottles_l151_151565

theorem diet_soda_bottles (R D : ℕ) 
  (h1 : R = 60)
  (h2 : R = D + 41) :
  D = 19 :=
by {
  sorry
}

end diet_soda_bottles_l151_151565


namespace lizzy_wealth_after_loan_l151_151341

theorem lizzy_wealth_after_loan 
  (initial_wealth : ℕ)
  (loan : ℕ)
  (interest_rate : ℕ)
  (h1 : initial_wealth = 30)
  (h2 : loan = 15)
  (h3 : interest_rate = 20)
  : (initial_wealth - loan) + (loan + loan * interest_rate / 100) = 33 :=
by
  sorry

end lizzy_wealth_after_loan_l151_151341


namespace sum_of_squares_of_consecutive_integers_divisible_by_5_l151_151739

theorem sum_of_squares_of_consecutive_integers_divisible_by_5 (n : ℤ) :
  (n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2) % 5 = 0 :=
by
  sorry

end sum_of_squares_of_consecutive_integers_divisible_by_5_l151_151739


namespace study_group_number_l151_151396

theorem study_group_number (b : ℤ) :
  (¬ (b % 2 = 0) ∧ (b + b^3 < 8000) ∧ ¬ (∃ r : ℚ, r^2 = 13) ∧ (b % 7 = 0)
  ∧ (∃ r : ℚ, r = b) ∧ ¬ (b % 14 = 0)) →
  b = 7 :=
by
  sorry

end study_group_number_l151_151396


namespace slices_per_pizza_l151_151208

theorem slices_per_pizza (num_pizzas num_slices : ℕ) (h1 : num_pizzas = 17) (h2 : num_slices = 68) :
  (num_slices / num_pizzas) = 4 :=
by
  sorry

end slices_per_pizza_l151_151208


namespace rhombus_area_l151_151247

-- Define the parameters given in the problem
namespace MathProof

def perimeter (EFGH : ℝ) : ℝ := 80
def diagonal_EG (EFGH : ℝ) : ℝ := 30

-- Considering the rhombus EFGH with the given perimeter and diagonal
theorem rhombus_area : 
  ∃ (area : ℝ), area = 150 * Real.sqrt 7 ∧ 
  (perimeter EFGH = 80) ∧ 
  (diagonal_EG EFGH = 30) :=
  sorry
end MathProof

end rhombus_area_l151_151247


namespace sqrt_of_1_5625_eq_1_25_l151_151809

theorem sqrt_of_1_5625_eq_1_25 : Real.sqrt 1.5625 = 1.25 :=
  sorry

end sqrt_of_1_5625_eq_1_25_l151_151809


namespace score_of_juniors_correct_l151_151742

-- Let the total number of students be 20
def total_students : ℕ := 20

-- 20% of the students are juniors
def juniors_percent : ℝ := 0.20

-- Total number of juniors
def number_of_juniors : ℕ := 4 -- 20% of 20

-- The remaining are seniors
def number_of_seniors : ℕ := 16 -- 80% of 20

-- Overall average score of all students
def overall_average_score : ℝ := 85

-- Average score of the seniors
def seniors_average_score : ℝ := 84

-- Calculate the total score of all students
def total_score : ℝ := overall_average_score * total_students

-- Calculate the total score of the seniors
def total_score_of_seniors : ℝ := seniors_average_score * number_of_seniors

-- We need to prove that the score of each junior
def score_of_each_junior : ℝ := 89

theorem score_of_juniors_correct :
  (total_score - total_score_of_seniors) / number_of_juniors = score_of_each_junior :=
by
  sorry

end score_of_juniors_correct_l151_151742


namespace inequality_solution_set_l151_151332

theorem inequality_solution_set (x : ℝ) : (x^2 ≥ 4) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by sorry

end inequality_solution_set_l151_151332


namespace fraction_of_juan_chocolates_given_to_tito_l151_151404

variable (n : ℕ)
variable (Juan Angela Tito : ℕ)
variable (f : ℝ)

-- Conditions
def chocolates_Angela_Tito : Angela = 3 * Tito := 
by sorry

def chocolates_Juan_Angela : Juan = 4 * Angela := 
by sorry

def equal_distribution : (Juan + Angela + Tito) = 16 * n := 
by sorry

-- Theorem to prove
theorem fraction_of_juan_chocolates_given_to_tito (n : ℕ) 
  (H1 : Angela = 3 * Tito)
  (H2 : Juan = 4 * Angela)
  (H3 : Juan + Angela + Tito = 16 * n) :
  f = 13 / 36 :=
by sorry

end fraction_of_juan_chocolates_given_to_tito_l151_151404


namespace visitors_not_ill_l151_151134

theorem visitors_not_ill (total_visitors : ℕ) (percent_ill : ℕ) (H1 : total_visitors = 500) (H2 : percent_ill = 40) : 
  total_visitors * (100 - percent_ill) / 100 = 300 := 
by 
  sorry

end visitors_not_ill_l151_151134


namespace find_power_l151_151766

theorem find_power (some_power : ℕ) (k : ℕ) :
  k = 8 → (1/2 : ℝ)^some_power * (1/81 : ℝ)^k = 1/(18^16 : ℝ) → some_power = 16 :=
by
  intro h1 h2
  rw [h1] at h2
  sorry

end find_power_l151_151766


namespace max_diagonal_intersections_l151_151382

theorem max_diagonal_intersections (n : ℕ) (h : n ≥ 4) : 
    ∃ k, k = n * (n - 1) * (n - 2) * (n - 3) / 24 :=
by
    sorry

end max_diagonal_intersections_l151_151382


namespace domain_of_f_2x_minus_3_l151_151889

noncomputable def f (x : ℝ) := 2 * x + 1

theorem domain_of_f_2x_minus_3 :
  (∀ x, 1 ≤ 2 * x - 3 ∧ 2 * x - 3 ≤ 5 → (2 ≤ x ∧ x ≤ 4)) :=
by
  sorry

end domain_of_f_2x_minus_3_l151_151889


namespace relationship_between_heights_is_correlated_l151_151710

theorem relationship_between_heights_is_correlated :
  (∃ r : ℕ, (r = 1 ∨ r = 2 ∨ r = 3 ∨ r = 4) ∧ r = 2) := by
  sorry

end relationship_between_heights_is_correlated_l151_151710


namespace bryan_travel_ratio_l151_151958

theorem bryan_travel_ratio
  (walk_time : ℕ)
  (bus_time : ℕ)
  (evening_walk_time : ℕ)
  (total_travel_hours : ℕ)
  (days_per_year : ℕ)
  (minutes_per_hour : ℕ)
  (minutes_total : ℕ)
  (daily_travel_time : ℕ) :
  walk_time = 5 →
  bus_time = 20 →
  evening_walk_time = 5 →
  total_travel_hours = 365 →
  days_per_year = 365 →
  minutes_per_hour = 60 →
  minutes_total = total_travel_hours * minutes_per_hour →
  daily_travel_time = (walk_time + bus_time + evening_walk_time) * 2 →
  (minutes_total / daily_travel_time = days_per_year) →
  (walk_time + bus_time + evening_walk_time) = daily_travel_time / 2 →
  (walk_time + bus_time + evening_walk_time) = daily_travel_time / 2 :=
by
  intros
  sorry

end bryan_travel_ratio_l151_151958


namespace consecutive_sum_impossible_l151_151784

theorem consecutive_sum_impossible (n : ℕ) :
  (¬ (∃ (a b : ℕ), a < b ∧ n = (b - a + 1) * (a + b) / 2)) ↔ ∃ s : ℕ, n = 2 ^ s :=
sorry

end consecutive_sum_impossible_l151_151784


namespace floor_sqrt_245_l151_151567

theorem floor_sqrt_245 : (Int.floor (Real.sqrt 245)) = 15 :=
by
  sorry

end floor_sqrt_245_l151_151567


namespace points_on_line_l151_151499

theorem points_on_line (x y : ℝ) (h : x + y = 0) : y = -x :=
by
  sorry

end points_on_line_l151_151499


namespace Katya_saves_enough_l151_151645

theorem Katya_saves_enough {h c_pool_sauna x y : ℕ} (hc : h = 275) (hcs : c_pool_sauna = 250)
  (hx : x = y + 200) (heq : x + y = c_pool_sauna) : (h / (c_pool_sauna - x)) = 11 :=
by
  sorry

end Katya_saves_enough_l151_151645


namespace situps_difference_l151_151049

def ken_situps : ℕ := 20
def nathan_situps : ℕ := 2 * ken_situps
def bob_situps : ℕ := (ken_situps + nathan_situps) / 2
def emma_situps : ℕ := bob_situps / 3

theorem situps_difference : 
  (nathan_situps + bob_situps + emma_situps) - ken_situps = 60 := by
  sorry

end situps_difference_l151_151049


namespace determine_time_Toronto_l151_151655

noncomputable def timeDifferenceBeijingToronto: ℤ := -12

def timeBeijing: ℕ × ℕ := (1, 8) -- (day, hour) format for simplicity: October 1st, 8:00

def timeToronto: ℕ × ℕ := (30, 20) -- Expected result in (day, hour): September 30th, 20:00

theorem determine_time_Toronto :
  timeDifferenceBeijingToronto = -12 →
  timeBeijing = (1, 8) →
  timeToronto = (30, 20) :=
by
  -- proof to be written 
  sorry

end determine_time_Toronto_l151_151655


namespace probability_computation_l151_151602

noncomputable def probability_inside_sphere : ℝ :=
  let volume_of_cube : ℝ := 64
  let volume_of_sphere : ℝ := (4/3) * Real.pi * (2^3)
  volume_of_sphere / volume_of_cube

theorem probability_computation :
  probability_inside_sphere = Real.pi / 6 :=
by
  sorry

end probability_computation_l151_151602


namespace fraction_transformation_l151_151420

theorem fraction_transformation (a b x: ℝ) (h1 : x = a / b) (h2 : a ≠ b) (h3 : b ≠ 0) :
  (a + 2 * b) / (a - 2 * b) = (x + 2) / (x - 2) :=
by sorry

end fraction_transformation_l151_151420


namespace difference_highest_lowest_score_l151_151915

-- Definitions based on conditions
def total_innings : ℕ := 46
def avg_innings : ℕ := 61
def highest_score : ℕ := 202
def avg_excl_highest_lowest : ℕ := 58
def innings_excl_highest_lowest : ℕ := 44

-- Calculated total runs
def total_runs : ℕ := total_innings * avg_innings
def total_runs_excl_highest_lowest : ℕ := innings_excl_highest_lowest * avg_excl_highest_lowest
def sum_of_highest_lowest : ℕ := total_runs - total_runs_excl_highest_lowest
def lowest_score : ℕ := sum_of_highest_lowest - highest_score

theorem difference_highest_lowest_score 
  (h1: total_runs = total_innings * avg_innings)
  (h2: avg_excl_highest_lowest * innings_excl_highest_lowest = total_runs_excl_highest_lowest)
  (h3: sum_of_highest_lowest = total_runs - total_runs_excl_highest_lowest)
  (h4: highest_score = 202)
  (h5: lowest_score = sum_of_highest_lowest - highest_score)
  : highest_score - lowest_score = 150 :=
by
  -- We only need to state the theorem, so we can skip the proof.
  -- The exact statements of conditions and calculations imply the result.
  sorry

end difference_highest_lowest_score_l151_151915


namespace inequality_proof_l151_151270

theorem inequality_proof (x y z : ℝ) : 
    x^4 + y^4 + z^2 + 1 ≥ 2 * x * (x * y^2 - x + z + 1) :=
by
  sorry

end inequality_proof_l151_151270


namespace find_alpha_l151_151579

theorem find_alpha (α : ℝ) (k : ℤ) 
  (h : ∃ (k : ℤ), α + 30 = k * 360 + 180) : 
  α = k * 360 + 150 :=
by 
  sorry

end find_alpha_l151_151579


namespace ratio_diminished_to_total_l151_151576

-- Definitions related to the conditions
def N := 240
def P := 60
def fifth_part_increased (N : ℕ) : ℕ := (N / 5) + 6
def part_diminished (P : ℕ) : ℕ := P - 6

-- The proof problem statement
theorem ratio_diminished_to_total 
  (h1 : fifth_part_increased N = part_diminished P) : 
  (P - 6) / N = 9 / 40 :=
by sorry

end ratio_diminished_to_total_l151_151576


namespace Rachel_made_total_amount_l151_151248

def cost_per_bar : ℝ := 3.25
def total_bars_sold : ℕ := 25 - 7
def total_amount_made : ℝ := total_bars_sold * cost_per_bar

theorem Rachel_made_total_amount :
  total_amount_made = 58.50 :=
by
  sorry

end Rachel_made_total_amount_l151_151248


namespace louisa_average_speed_l151_151685

def average_speed (v : ℝ) : Prop :=
  (350 / v) - (200 / v) = 3

theorem louisa_average_speed :
  ∃ v : ℝ, average_speed v ∧ v = 50 := 
by
  use 50
  unfold average_speed
  sorry

end louisa_average_speed_l151_151685


namespace cost_of_each_cake_l151_151906

-- Define the conditions
def cakes : ℕ := 3
def payment_by_john : ℕ := 18
def total_payment : ℕ := payment_by_john * 2

-- Statement to prove that each cake costs $12
theorem cost_of_each_cake : (total_payment / cakes) = 12 := by
  sorry

end cost_of_each_cake_l151_151906


namespace total_money_spent_l151_151799

def candy_bar_cost : ℕ := 14
def cookie_box_cost : ℕ := 39
def total_spent : ℕ := candy_bar_cost + cookie_box_cost

theorem total_money_spent : total_spent = 53 := by
  sorry

end total_money_spent_l151_151799


namespace B_profit_l151_151317

-- Definitions based on conditions
def investment_ratio (B_invest A_invest : ℕ) : Prop := A_invest = 3 * B_invest
def period_ratio (B_period A_period : ℕ) : Prop := A_period = 2 * B_period
def total_profit (total : ℕ) : Prop := total = 28000
def B_share (total : ℕ) := total / 7

-- Theorem statement based on the proof problem
theorem B_profit (B_invest A_invest B_period A_period total : ℕ)
  (h1 : investment_ratio B_invest A_invest)
  (h2 : period_ratio B_period A_period)
  (h3 : total_profit total) :
  B_share total = 4000 :=
by
  sorry

end B_profit_l151_151317


namespace parabola_standard_equations_l151_151536

noncomputable def parabola_focus_condition (x y : ℝ) : Prop := 
  x + 2 * y + 3 = 0

theorem parabola_standard_equations (x y : ℝ) 
  (h : parabola_focus_condition x y) :
  (y ^ 2 = -12 * x) ∨ (x ^ 2 = -6 * y) :=
by
  sorry

end parabola_standard_equations_l151_151536


namespace angela_sleep_difference_l151_151040

theorem angela_sleep_difference :
  let december_sleep_hours := 6.5
  let january_sleep_hours := 8.5
  let december_days := 31
  let january_days := 31
  (january_sleep_hours * january_days) - (december_sleep_hours * december_days) = 62 :=
by
  sorry

end angela_sleep_difference_l151_151040


namespace total_wheels_correct_l151_151510

-- Define the initial state of the garage
def initial_bicycles := 20
def initial_cars := 10
def initial_motorcycles := 5
def initial_tricycles := 3
def initial_quads := 2

-- Define the changes in the next hour
def bicycles_leaving := 7
def cars_arriving := 4
def motorcycles_arriving := 3
def motorcycles_leaving := 2

-- Define the damaged vehicles
def damaged_bicycles := 5  -- each missing 1 wheel
def damaged_cars := 2      -- each missing 1 wheel
def damaged_motorcycle := 1 -- missing 2 wheels

-- Define the number of wheels per type of vehicle
def bicycle_wheels := 2
def car_wheels := 4
def motorcycle_wheels := 2
def tricycle_wheels := 3
def quad_wheels := 4

-- Calculate the state of vehicles at the end of the hour
def final_bicycles := initial_bicycles - bicycles_leaving
def final_cars := initial_cars + cars_arriving
def final_motorcycles := initial_motorcycles + motorcycles_arriving - motorcycles_leaving

-- Calculate the total wheels in the garage at the end of the hour
def total_wheels : Nat := 
  (final_bicycles - damaged_bicycles) * bicycle_wheels + damaged_bicycles +
  (final_cars - damaged_cars) * car_wheels + damaged_cars * 3 +
  (final_motorcycles - damaged_motorcycle) * motorcycle_wheels +
  initial_tricycles * tricycle_wheels +
  initial_quads * quad_wheels

-- The goal is to prove that the total number of wheels in the garage is 102 at the end of the hour
theorem total_wheels_correct : total_wheels = 102 := 
  by
    sorry

end total_wheels_correct_l151_151510


namespace exists_k_l151_151030

-- Definitions of the conditions
def sequence_def (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a (n+1) = Nat.lcm (a n) (a (n-1)) - Nat.lcm (a (n-1)) (a (n-2))

theorem exists_k (a : ℕ → ℕ) (a₁ a₂ a₃ : ℕ) (h₁ : a 1 = a₁) (h₂ : a 2 = a₂) (h₃ : a 3 = a₃)
  (h_seq : sequence_def a) : ∃ k : ℕ, k ≤ a₃ + 4 ∧ a k = 0 := 
sorry

end exists_k_l151_151030


namespace smallest_prime_with_digit_sum_18_l151_151860

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_18 : ∃ p : ℕ, Prime p ∧ 18 = sum_of_digits p ∧ (∀ q : ℕ, (Prime q ∧ 18 = sum_of_digits q) → p ≤ q) :=
by
  sorry

end smallest_prime_with_digit_sum_18_l151_151860


namespace cos_double_angle_from_sin_shift_l151_151867

theorem cos_double_angle_from_sin_shift (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : 
  Real.cos (2 * θ) = -7 / 25 := 
by 
  sorry

end cos_double_angle_from_sin_shift_l151_151867


namespace Isabel_initial_flowers_l151_151394

-- Constants for conditions
def b := 7  -- Number of bouquets after wilting
def fw := 10  -- Number of wilted flowers
def n := 8  -- Number of flowers in each bouquet

-- Theorem statement
theorem Isabel_initial_flowers (h1 : b = 7) (h2 : fw = 10) (h3 : n = 8) : 
  (b * n + fw = 66) := by
  sorry

end Isabel_initial_flowers_l151_151394


namespace longest_side_of_triangle_l151_151745

-- Defining variables and constants
variables (x : ℕ)

-- Defining the side lengths of the triangle
def side1 := 7
def side2 := x + 4
def side3 := 2 * x + 1

-- Defining the perimeter of the triangle
def perimeter := side1 + side2 + side3

-- Statement of the main theorem
theorem longest_side_of_triangle (h : perimeter x = 36) : max side1 (max (side2 x) (side3 x)) = 17 :=
by sorry

end longest_side_of_triangle_l151_151745


namespace unique_solution_l151_151511

theorem unique_solution (m n : ℤ) (h : 231 * m^2 = 130 * n^2) : m = 0 ∧ n = 0 :=
by {
  sorry
}

end unique_solution_l151_151511


namespace grocery_delivery_amount_l151_151836

theorem grocery_delivery_amount (initial_savings final_price trips : ℕ) 
(fixed_charge : ℝ) (percent_charge : ℝ) (total_saved : ℝ) : 
  initial_savings = 14500 →
  final_price = 14600 →
  trips = 40 →
  fixed_charge = 1.5 →
  percent_charge = 0.05 →
  total_saved = final_price - initial_savings →
  60 + percent_charge * G = total_saved →
  G = 800 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end grocery_delivery_amount_l151_151836


namespace initial_percentage_salt_l151_151791

theorem initial_percentage_salt :
  ∀ (P : ℝ),
  let Vi := 64 
  let Vf := 80
  let target_percent := 0.08
  (Vi * P = Vf * target_percent) → P = 0.1 :=
by
  intros P Vi Vf target_percent h
  have h1 : Vi = 64 := rfl
  have h2 : Vf = 80 := rfl
  have h3 : target_percent = 0.08 := rfl
  rw [h1, h2, h3] at h
  sorry

end initial_percentage_salt_l151_151791


namespace correlation_coefficient_is_one_l151_151362

noncomputable def correlation_coefficient (x_vals y_vals : List ℝ) : ℝ := sorry

theorem correlation_coefficient_is_one 
  (n : ℕ) 
  (x y : Fin n → ℝ) 
  (h1 : n ≥ 2) 
  (h2 : ∃ i j, i ≠ j ∧ x i ≠ x j) 
  (h3 : ∀ i, y i = 3 * x i + 1) : 
  correlation_coefficient (List.ofFn x) (List.ofFn y) = 1 := 
sorry

end correlation_coefficient_is_one_l151_151362


namespace abigail_saving_period_l151_151488

-- Define the conditions
def amount_saved_each_month : ℕ := 4000
def total_amount_saved : ℕ := 48000

-- State the theorem
theorem abigail_saving_period : total_amount_saved / amount_saved_each_month = 12 := by
  -- Proof would go here
  sorry

end abigail_saving_period_l151_151488


namespace Bill_order_combinations_l151_151703

def donut_combinations (num_donuts num_kinds : ℕ) : ℕ :=
  Nat.choose (num_donuts + num_kinds - 1) (num_kinds - 1)

theorem Bill_order_combinations : donut_combinations 10 5 = 126 :=
by
  -- This would be the place to insert the proof steps, but we're using sorry as the placeholder.
  sorry

end Bill_order_combinations_l151_151703


namespace cages_needed_l151_151630

theorem cages_needed (initial_puppies sold_puppies puppies_per_cage : ℕ) (h1 : initial_puppies = 13) (h2 : sold_puppies = 7) (h3 : puppies_per_cage = 2) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 3 := 
by
  sorry

end cages_needed_l151_151630


namespace area_percentage_change_is_neg_4_percent_l151_151036

noncomputable def percent_change_area (L W : ℝ) : ℝ :=
  let A_initial := L * W
  let A_new := (1.20 * L) * (0.80 * W)
  ((A_new - A_initial) / A_initial) * 100

theorem area_percentage_change_is_neg_4_percent (L W : ℝ) :
  percent_change_area L W = -4 :=
by
  sorry

end area_percentage_change_is_neg_4_percent_l151_151036


namespace sum_angles_triangle_complement_l151_151950

theorem sum_angles_triangle_complement (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 180 - C = 130) : A + B = 130 :=
by
  have hC : C = 50 := by linarith
  linarith

end sum_angles_triangle_complement_l151_151950


namespace distinct_solutions_equation_l151_151927

theorem distinct_solutions_equation (a b : ℝ) (h1 : a ≠ b) (h2 : a > b) (h3 : ∀ x, (3 * x - 9) / (x^2 + 3 * x - 18) = x + 1) (sol_a : x = a) (sol_b : x = b) :
  a - b = 1 :=
sorry

end distinct_solutions_equation_l151_151927


namespace factorize_expression_l151_151928

-- Define variables m and n
variables (m n : ℤ)

-- The theorem stating the equality
theorem factorize_expression : m^3 * n - m * n = m * n * (m - 1) * (m + 1) :=
by sorry

end factorize_expression_l151_151928


namespace number_of_cats_adopted_l151_151624

theorem number_of_cats_adopted (c : ℕ) 
  (h1 : 50 * c + 3 * 100 + 2 * 150 = 700) :
  c = 2 :=
by
  sorry

end number_of_cats_adopted_l151_151624


namespace pond_depth_range_l151_151981

theorem pond_depth_range (d : ℝ) (adam_false : d < 10) (ben_false : d > 8) (carla_false : d ≠ 7) : 
    8 < d ∧ d < 10 :=
by
  sorry

end pond_depth_range_l151_151981


namespace range_of_a_l151_151633

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a-1) * x^2 + 2 * (a-1) * x - 4 ≥ 0 -> false) ↔ -3 < a ∧ a ≤ 1 := by
  sorry

end range_of_a_l151_151633


namespace polygon_sides_eq_eight_l151_151625

theorem polygon_sides_eq_eight (n : ℕ) (h : (n - 2) * 180 = 3 * 360) : n = 8 := by 
  sorry

end polygon_sides_eq_eight_l151_151625


namespace percent_motorists_exceeding_speed_limit_l151_151741

-- Definitions based on conditions:
def total_motorists := 100
def percent_receiving_tickets := 10
def percent_exceeding_no_ticket := 50

-- The Lean 4 statement to prove the question
theorem percent_motorists_exceeding_speed_limit :
  (percent_receiving_tickets + (percent_receiving_tickets * percent_exceeding_no_ticket / 100)) = 20 :=
by
  sorry

end percent_motorists_exceeding_speed_limit_l151_151741


namespace product_of_fractions_l151_151555

theorem product_of_fractions :
  (1 / 2) * (3 / 5) * (5 / 6) = 1 / 4 := 
by
  sorry

end product_of_fractions_l151_151555


namespace binom_10_3_eq_120_l151_151068

def binom (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem binom_10_3_eq_120 : binom 10 3 = 120 :=
by 
sorry

end binom_10_3_eq_120_l151_151068


namespace D_144_l151_151959

def D (n : ℕ) : ℕ :=
  if n = 1 then 1 else sorry

theorem D_144 : D 144 = 51 := by
  sorry

end D_144_l151_151959


namespace total_legs_l151_151064

-- Define the number of octopuses
def num_octopuses : ℕ := 5

-- Define the number of legs per octopus
def legs_per_octopus : ℕ := 8

-- The total number of legs should be num_octopuses * legs_per_octopus
theorem total_legs : num_octopuses * legs_per_octopus = 40 :=
by
  -- The proof is omitted
  sorry

end total_legs_l151_151064


namespace gcd_f_50_51_l151_151245

def f (x : ℤ) : ℤ :=
  x ^ 2 - 2 * x + 2023

theorem gcd_f_50_51 : Int.gcd (f 50) (f 51) = 11 := by
  sorry

end gcd_f_50_51_l151_151245


namespace polygon_num_sides_and_exterior_angle_l151_151058

theorem polygon_num_sides_and_exterior_angle 
  (n : ℕ) (x : ℕ) 
  (h : (n - 2) * 180 + x = 1350) 
  (hx : 0 < x ∧ x < 180) 
  : (n = 9) ∧ (x = 90) := 
by 
  sorry

end polygon_num_sides_and_exterior_angle_l151_151058


namespace trip_duration_is_6_hours_l151_151318

def distance_1 := 55 * 4
def total_distance (A : ℕ) := distance_1 + 70 * A
def total_time (A : ℕ) := 4 + A
def average_speed (A : ℕ) := total_distance A / total_time A

theorem trip_duration_is_6_hours (A : ℕ) (h : 60 = average_speed A) : total_time A = 6 :=
by
  sorry

end trip_duration_is_6_hours_l151_151318


namespace geometric_series_first_term_l151_151858

theorem geometric_series_first_term (a : ℝ) (r : ℝ) (s : ℝ) 
  (h1 : r = -1/3) (h2 : s = 12) (h3 : s = a / (1 - r)) : a = 16 :=
by
  -- Placeholder for the proof
  sorry

end geometric_series_first_term_l151_151858


namespace directrix_of_parabola_l151_151114

theorem directrix_of_parabola (p : ℝ) (hp : 2 * p = 4) : 
  (∃ x : ℝ, x = -1) :=
by
  sorry

end directrix_of_parabola_l151_151114


namespace derivative_y_l151_151459

noncomputable def y (x : ℝ) : ℝ :=
  (Real.sqrt (9 * x^2 - 12 * x + 5)) * Real.arctan (3 * x - 2) - 
  Real.log (3 * x - 2 + Real.sqrt (9 * x^2 - 12 * x + 5))

theorem derivative_y (x : ℝ) :
  ∃ (f' : ℝ → ℝ), deriv y x = f' x ∧ f' x = (9 * x - 6) * Real.arctan (3 * x - 2) / 
  Real.sqrt (9 * x^2 - 12 * x + 5) :=
sorry

end derivative_y_l151_151459


namespace point_not_on_line_l151_151119

theorem point_not_on_line (a c : ℝ) (h : a * c > 0) : ¬(0 = 2500 * a + c) := by
  sorry

end point_not_on_line_l151_151119


namespace sufficient_condition_hyperbola_l151_151237

theorem sufficient_condition_hyperbola (m : ℝ) (h : 5 < m) : 
  ∃ a b : ℝ, (a > 0) ∧ (b < 0) ∧ (∀ x y : ℝ, (x^2)/(a) + (y^2)/(b) = 1) := 
sorry

end sufficient_condition_hyperbola_l151_151237


namespace minimum_value_of_f_l151_151274

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3 * x + 9

-- State the theorem about the minimum value of the function
theorem minimum_value_of_f : ∃ x : ℝ, f x = 7 ∧ ∀ y : ℝ, f y ≥ 7 := sorry

end minimum_value_of_f_l151_151274


namespace coin_draws_expected_value_l151_151922

theorem coin_draws_expected_value :
  ∃ f : ℕ → ℝ, (∀ (n : ℕ), n ≥ 4 → f n = (3 : ℝ)) := sorry

end coin_draws_expected_value_l151_151922


namespace fraction_eggs_given_to_Sofia_l151_151112

variables (m : ℕ) -- Number of eggs Mia has
def Sofia_eggs := 3 * m
def Pablo_eggs := 4 * Sofia_eggs
def Lucas_eggs := 0

theorem fraction_eggs_given_to_Sofia (h1 : Pablo_eggs = 12 * m) :
  (1 : ℚ) / (12 : ℚ) = 1 / 12 := by sorry

end fraction_eggs_given_to_Sofia_l151_151112


namespace cost_of_mens_t_shirt_l151_151390

-- Definitions based on conditions
def womens_price : ℕ := 18
def womens_interval : ℕ := 30
def mens_interval : ℕ := 40
def shop_open_hours_per_day : ℕ := 12
def total_earnings_per_week : ℕ := 4914

-- Auxiliary definitions based on conditions
def t_shirts_sold_per_hour (interval : ℕ) : ℕ := 60 / interval
def t_shirts_sold_per_day (interval : ℕ) : ℕ := shop_open_hours_per_day * t_shirts_sold_per_hour interval
def t_shirts_sold_per_week (interval : ℕ) : ℕ := t_shirts_sold_per_day interval * 7

def weekly_earnings_womens : ℕ := womens_price * t_shirts_sold_per_week womens_interval
def weekly_earnings_mens : ℕ := total_earnings_per_week - weekly_earnings_womens
def mens_price : ℚ := weekly_earnings_mens / t_shirts_sold_per_week mens_interval

-- The statement to be proved
theorem cost_of_mens_t_shirt : mens_price = 15 := by
  sorry

end cost_of_mens_t_shirt_l151_151390


namespace sum_coeff_expansion_l151_151540

theorem sum_coeff_expansion (x y : ℝ) : 
  (x + 2 * y)^4 = 81 := sorry

end sum_coeff_expansion_l151_151540


namespace sum_of_areas_of_two_parks_l151_151541

theorem sum_of_areas_of_two_parks :
  let side1 := 11
  let side2 := 5
  let area1 := side1 * side1
  let area2 := side2 * side2
  area1 + area2 = 146 := 
by 
  sorry

end sum_of_areas_of_two_parks_l151_151541


namespace contrapositive_proposition_l151_151185

theorem contrapositive_proposition
  (a b c d : ℝ) 
  (h : a + c ≠ b + d) : a ≠ b ∨ c ≠ d :=
sorry

end contrapositive_proposition_l151_151185


namespace lcm_18_60_is_180_l151_151181

theorem lcm_18_60_is_180 : Nat.lcm 18 60 = 180 := 
  sorry

end lcm_18_60_is_180_l151_151181


namespace count_solutions_congruence_l151_151416

theorem count_solutions_congruence : 
  ∃ (s : Finset ℕ), s.card = 4 ∧ ∀ x ∈ s, x + 20 ≡ 75 [MOD 45] ∧ x < 150 :=
sorry

end count_solutions_congruence_l151_151416


namespace nicky_catches_up_time_l151_151478

theorem nicky_catches_up_time
  (head_start : ℕ := 12)
  (cristina_speed : ℕ := 5)
  (nicky_speed : ℕ := 3)
  (head_start_distance : ℕ := nicky_speed * head_start)
  (time_to_catch_up : ℕ := 36 / 2) -- 36 is the head start distance of 36 meters
  (total_time : ℕ := time_to_catch_up + head_start)  -- Total time Nicky runs before Cristina catches up
  : total_time = 30 := sorry

end nicky_catches_up_time_l151_151478


namespace remainder_of_b97_is_52_l151_151450

def b (n : ℕ) : ℕ := 7^n + 9^n

theorem remainder_of_b97_is_52 : (b 97) % 81 = 52 := 
sorry

end remainder_of_b97_is_52_l151_151450


namespace arith_seq_problem_l151_151453

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n, a n = a1 + (n - 1) * d

theorem arith_seq_problem 
  (a : ℕ → ℝ) (a1 d : ℝ)
  (h1 : arithmetic_sequence a a1 d)
  (h2 : a 1 + 3 * a 8 + a 15 = 120) : 
  3 * a 9 - a 11 = 48 :=
by 
  sorry

end arith_seq_problem_l151_151453


namespace water_height_in_cylinder_l151_151684

theorem water_height_in_cylinder :
  let r_cone := 10 -- Radius of the cone in cm
  let h_cone := 15 -- Height of the cone in cm
  let r_cylinder := 20 -- Radius of the cylinder in cm
  let volume_cone := (1 / 3) * Real.pi * r_cone^2 * h_cone
  volume_cone = 500 * Real.pi -> 
  let h_cylinder := volume_cone / (Real.pi * r_cylinder^2)
  h_cylinder = 1.25 := 
by
  intros r_cone h_cone r_cylinder volume_cone h_volume
  let h_cylinder := volume_cone / (Real.pi * r_cylinder^2)
  have : h_cylinder = 1.25 := by
    sorry
  exact this

end water_height_in_cylinder_l151_151684


namespace union_P_Q_l151_151997

noncomputable def P : Set ℤ := {x | x^2 - x = 0}
noncomputable def Q : Set ℤ := {x | ∃ y : ℝ, y = Real.sqrt (1 - x^2)}

theorem union_P_Q : P ∪ Q = {-1, 0, 1} :=
by 
  sorry

end union_P_Q_l151_151997


namespace range_of_a_l151_151840

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, 2 * x ^ 2 + (a - 1) * x + 1 / 2 ≤ 0) → (-1 < a ∧ a < 3) :=
by 
  sorry

end range_of_a_l151_151840


namespace avg_visitors_sundays_l151_151545

-- Definitions
def days_in_month := 30
def avg_visitors_per_day_month := 750
def avg_visitors_other_days := 700
def sundays_in_month := 5
def other_days := days_in_month - sundays_in_month

-- Main statement to prove
theorem avg_visitors_sundays (S : ℕ) 
  (H1 : days_in_month = 30) 
  (H2 : avg_visitors_per_day_month = 750) 
  (H3 : avg_visitors_other_days = 700) 
  (H4 : sundays_in_month = 5) 
  (H5 : other_days = days_in_month - sundays_in_month) 
  :
  (sundays_in_month * S + other_days * avg_visitors_other_days) = avg_visitors_per_day_month * days_in_month 
  → S = 1000 :=
by 
  sorry

end avg_visitors_sundays_l151_151545


namespace max_m_value_l151_151291

theorem max_m_value (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : 2 / a + 1 / b = 1 / 4) : ∃ m : ℝ, (∀ a b : ℝ,  a > 0 ∧ b > 0 ∧ (2 / a + 1 / b = 1 / 4) → 2 * a + b ≥ 4 * m) ∧ m = 7 / 4 :=
sorry

end max_m_value_l151_151291


namespace megatek_employees_in_manufacturing_l151_151104

theorem megatek_employees_in_manufacturing :
  let total_degrees := 360
  let manufacturing_degrees := 108
  (manufacturing_degrees / total_degrees.toFloat) * 100 = 30 := 
by
  sorry

end megatek_employees_in_manufacturing_l151_151104


namespace problem_l151_151631

-- Define the problem
theorem problem {a b c : ℤ} (h1 : a = c + 1) (h2 : b - 1 = a) :
  (a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2 = 6 := 
sorry

end problem_l151_151631


namespace trapezoid_median_l151_151843

theorem trapezoid_median
  (h : ℝ)
  (area_triangle : ℝ)
  (area_trapezoid : ℝ)
  (bt : ℝ)
  (bt_sum : ℝ)
  (ht_positive : h ≠ 0)
  (triangle_area : area_triangle = (1/2) * bt * h)
  (trapezoid_area : area_trapezoid = area_triangle)
  (trapezoid_bt_sum : bt_sum = 40)
  (triangle_bt : bt = 24)
  : (bt_sum / 2) = 20 :=
by
  sorry

end trapezoid_median_l151_151843


namespace arctan_sum_eq_pi_div_4_l151_151015

noncomputable def n : ℤ := 27

theorem arctan_sum_eq_pi_div_4 :
  (Real.arctan (1 / 2) + Real.arctan (1 / 4) + Real.arctan (1 / 5) + Real.arctan (1 / n) = Real.pi / 4) :=
sorry

end arctan_sum_eq_pi_div_4_l151_151015


namespace train_length_eq_l151_151171

theorem train_length_eq (L : ℝ) (time_tree time_platform length_platform : ℝ)
  (h_tree : time_tree = 60) (h_platform : time_platform = 105) (h_length_platform : length_platform = 450)
  (h_speed_eq : L / time_tree = (L + length_platform) / time_platform) :
  L = 600 :=
by
  sorry

end train_length_eq_l151_151171


namespace combination_eq_permutation_div_factorial_l151_151562

-- Step d): Lean 4 Statement

variables (n k : ℕ)

-- Define combination C_n^k is any k-element subset of an n-element set
def combination (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

-- Define permutation A_n^k is the number of ways to arrange k elements out of n elements
def permutation (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- Statement to prove: C_n^k = A_n^k / k!
theorem combination_eq_permutation_div_factorial :
  combination n k = permutation n k / (Nat.factorial k) :=
by
  sorry

end combination_eq_permutation_div_factorial_l151_151562


namespace solution_to_quadratic_inequality_l151_151621

theorem solution_to_quadratic_inequality :
  {x : ℝ | x^2 + 3*x < 10} = {x : ℝ | -5 < x ∧ x < 2} :=
sorry

end solution_to_quadratic_inequality_l151_151621


namespace convex_polyhedron_faces_same_edges_l151_151892

theorem convex_polyhedron_faces_same_edges (n : ℕ) (f : Fin n → ℕ) 
  (n_ge_4 : 4 ≤ n)
  (h : ∀ i : Fin n, 3 ≤ f i ∧ f i ≤ n - 1) : 
  ∃ (i j : Fin n), i ≠ j ∧ f i = f j := 
by
  sorry

end convex_polyhedron_faces_same_edges_l151_151892


namespace total_snakes_seen_l151_151695

-- Define the number of snakes in each breeding ball
def snakes_in_first_breeding_ball : Nat := 15
def snakes_in_second_breeding_ball : Nat := 20
def snakes_in_third_breeding_ball : Nat := 25
def snakes_in_fourth_breeding_ball : Nat := 30
def snakes_in_fifth_breeding_ball : Nat := 35
def snakes_in_sixth_breeding_ball : Nat := 40
def snakes_in_seventh_breeding_ball : Nat := 45

-- Define the number of pairs of extra snakes
def extra_pairs_of_snakes : Nat := 23

-- Define the total number of snakes observed
def total_snakes_observed : Nat :=
  snakes_in_first_breeding_ball +
  snakes_in_second_breeding_ball +
  snakes_in_third_breeding_ball +
  snakes_in_fourth_breeding_ball +
  snakes_in_fifth_breeding_ball +
  snakes_in_sixth_breeding_ball +
  snakes_in_seventh_breeding_ball +
  (extra_pairs_of_snakes * 2)

theorem total_snakes_seen : total_snakes_observed = 256 := by
  sorry

end total_snakes_seen_l151_151695


namespace geometric_sequence_arithmetic_sequence_l151_151231

def seq₃ := 7
def rec_rel (a : ℕ → ℕ) := ∀ n, n ≥ 2 → a n = 2 * a (n - 1) + a 2 - 2

-- Problem Part 1: Prove that {a_n+1} is a geometric sequence
theorem geometric_sequence (a : ℕ → ℕ) (h_rec_rel : rec_rel a) :
  ∃ r, ∀ n, n ≥ 1 → (a n + 1) = r * (a (n - 1) + 1) :=
sorry

-- Problem Part 2: Given a general formula, prove n, a_n, and S_n form an arithmetic sequence
def general_formula (a : ℕ → ℕ) := ∀ n, a n = 2^n - 1
def sum_formula (S : ℕ → ℕ) := ∀ n, S n = 2^(n+1) - n - 2

theorem arithmetic_sequence (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h_general : general_formula a) (h_sum : sum_formula S) :
  ∀ n, n + S n = 2 * a n :=
sorry

end geometric_sequence_arithmetic_sequence_l151_151231


namespace slices_left_l151_151573

-- Conditions
def total_slices : ℕ := 16
def fraction_eaten : ℚ := 3/4
def fraction_left : ℚ := 1 - fraction_eaten

-- Proof statement
theorem slices_left : total_slices * fraction_left = 4 := by
  sorry

end slices_left_l151_151573


namespace area_of_BEIH_l151_151136

structure Point where
  x : ℚ
  y : ℚ

def B : Point := ⟨0, 0⟩
def A : Point := ⟨0, 2⟩
def D : Point := ⟨2, 2⟩
def C : Point := ⟨2, 0⟩
def E : Point := ⟨0, 1⟩
def F : Point := ⟨1, 0⟩
def I : Point := ⟨2/5, 6/5⟩
def H : Point := ⟨2/3, 2/3⟩

def quadrilateral_area (p1 p2 p3 p4 : Point) : ℚ :=
  (1/2 : ℚ) * 
  ((p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y) - 
   (p1.y * p2.x + p2.y * p3.x + p3.y * p4.x + p4.y * p1.x))

theorem area_of_BEIH : quadrilateral_area B E I H = 7 / 15 := sorry

end area_of_BEIH_l151_151136


namespace coloring_ways_l151_151692

-- Define the vertices and edges of the graph
def vertices : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8}

def edges : Finset (ℕ × ℕ) :=
  { (0, 1), (1, 2), (2, 0),  -- First triangle
    (3, 4), (4, 5), (5, 3),  -- Middle triangle
    (6, 7), (7, 8), (8, 6),  -- Third triangle
    (2, 5),   -- Connecting top horizontal edge
    (1, 7) }  -- Connecting bottom horizontal edge

-- Define the number of colors available
def colors := 4

-- Define a function to count the valid colorings given the vertices and edges
noncomputable def countValidColorings (vertices : Finset ℕ) (edges : Finset (ℕ × ℕ)) (colors : ℕ) : ℕ := sorry

-- The theorem statement
theorem coloring_ways : countValidColorings vertices edges colors = 3456 := 
sorry

end coloring_ways_l151_151692


namespace ttakjis_count_l151_151447

theorem ttakjis_count (n : ℕ) (initial_residual new_residual total_ttakjis : ℕ) :
  initial_residual = 36 → 
  new_residual = 3 → 
  total_ttakjis = n^2 + initial_residual → 
  total_ttakjis = (n + 1)^2 + new_residual → 
  total_ttakjis = 292 :=
by
  sorry

end ttakjis_count_l151_151447


namespace geometric_sequence_common_ratio_l151_151003

theorem geometric_sequence_common_ratio (q : ℝ) (a : ℕ → ℝ) 
  (h1 : a 2 = 1/2)
  (h2 : a 5 = 4)
  (h3 : ∀ n, a n = a 1 * q^(n - 1)) : 
  q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l151_151003


namespace tomatoes_multiplier_l151_151913

theorem tomatoes_multiplier (before_vacation : ℕ) (grown_during_vacation : ℕ)
  (h1 : before_vacation = 36)
  (h2 : grown_during_vacation = 3564) :
  (before_vacation + grown_during_vacation) / before_vacation = 100 :=
by
  -- Insert proof here later
  sorry

end tomatoes_multiplier_l151_151913


namespace meal_total_l151_151618

noncomputable def meal_price (appetizer entree dessert drink sales_tax tip : ℝ) : ℝ :=
  let total_before_tax := appetizer + (2 * entree) + dessert + (2 * drink)
  let tax_amount := (sales_tax / 100) * total_before_tax
  let subtotal := total_before_tax + tax_amount
  let tip_amount := (tip / 100) * subtotal
  subtotal + tip_amount

theorem meal_total : 
  meal_price 9 20 11 6.5 7.5 22 = 95.75 :=
by
  sorry

end meal_total_l151_151618


namespace not_always_greater_quotient_l151_151656

theorem not_always_greater_quotient (a : ℝ) (b : ℝ) (ha : a ≠ 0) (hb : 0 < b) : ¬ (∀ b < 1, a / b > a) ∧ ¬ (∀ b > 1, a / b > a) :=
by sorry

end not_always_greater_quotient_l151_151656


namespace professor_oscar_review_questions_l151_151164

-- Define the problem conditions.
def students_per_class : ℕ := 35
def questions_per_exam : ℕ := 10
def number_of_classes : ℕ := 5

-- Define the number of questions that must be reviewed.
def total_questions_to_review : ℕ := 1750

-- The theorem to be proved.
theorem professor_oscar_review_questions :
  students_per_class * questions_per_exam * number_of_classes = total_questions_to_review :=
by
  -- Here we write 'sorry' since we are not providing the full proof.
  sorry

end professor_oscar_review_questions_l151_151164


namespace total_and_average_games_l151_151897

def football_games_per_month : List Nat := [29, 35, 48, 43, 56, 36]
def baseball_games_per_month : List Nat := [15, 19, 23, 14, 18, 17]
def basketball_games_per_month : List Nat := [17, 21, 14, 32, 22, 27]

def total_games (games_per_month : List Nat) : Nat :=
  List.sum games_per_month

def average_games (total : Nat) (months : Nat) : Nat :=
  total / months

theorem total_and_average_games :
  total_games football_games_per_month + total_games baseball_games_per_month + total_games basketball_games_per_month = 486
  ∧ average_games (total_games football_games_per_month + total_games baseball_games_per_month + total_games basketball_games_per_month) 6 = 81 :=
by
  sorry

end total_and_average_games_l151_151897


namespace inequality_ratios_l151_151898

theorem inequality_ratios (a b c d : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) (h4 : c > d) :
  (c / a) > (d / b) :=
sorry

end inequality_ratios_l151_151898


namespace factorize_expression_l151_151827

theorem factorize_expression (a b : ℝ) : a^2 * b - 9 * b = b * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l151_151827


namespace simplify_tan_pi_over_24_add_tan_7pi_over_24_l151_151192

theorem simplify_tan_pi_over_24_add_tan_7pi_over_24 :
  let a := Real.tan (Real.pi / 24)
  let b := Real.tan (7 * Real.pi / 24)
  a + b = 2 * Real.sqrt 6 - 2 * Real.sqrt 3 :=
by
  -- conditions and definitions:
  let tan_eq_sin_div_cos := ∀ x, Real.tan x = Real.sin x / Real.cos x
  let sin_add := ∀ a b, Real.sin (a + b) = Real.sin a * Real.cos b + Real.cos a * Real.sin b
  let cos_mul := ∀ a b, Real.cos a * Real.cos b = 1 / 2 * (Real.cos (a + b) + Real.cos (a - b))
  let sin_pi_over_3 := Real.sin (Real.pi / 3) = Real.sqrt 3 / 2
  let cos_pi_over_3 := Real.cos (Real.pi / 3) = 1 / 2
  let cos_pi_over_4 := Real.cos (Real.pi / 4) = Real.sqrt 2 / 2
  have cond1 := tan_eq_sin_div_cos
  have cond2 := sin_add
  have cond3 := cos_mul
  have cond4 := sin_pi_over_3
  have cond5 := cos_pi_over_3
  have cond6 := cos_pi_over_4
  sorry

end simplify_tan_pi_over_24_add_tan_7pi_over_24_l151_151192


namespace correct_answer_of_john_l151_151218

theorem correct_answer_of_john (x : ℝ) (h : 5 * x + 4 = 104) : (x + 5) / 4 = 6.25 :=
by
  sorry

end correct_answer_of_john_l151_151218


namespace carol_blocks_l151_151775

theorem carol_blocks (x : ℕ) (h : x - 25 = 17) : x = 42 :=
sorry

end carol_blocks_l151_151775


namespace degree_at_least_three_l151_151387

noncomputable def p : Polynomial ℤ := sorry
noncomputable def q : Polynomial ℤ := sorry

theorem degree_at_least_three (h1 : p.degree ≥ 1)
                              (h2 : q.degree ≥ 1)
                              (h3 : (∃ xs : Fin 33 → ℤ, ∀ i, p.eval (xs i) * q.eval (xs i) - 2015 = 0)) :
  p.degree ≥ 3 ∧ q.degree ≥ 3 := 
sorry

end degree_at_least_three_l151_151387


namespace max_principals_in_10_years_l151_151664

theorem max_principals_in_10_years :
  ∀ (term_length : ℕ) (P : ℕ → Prop),
  (∀ n, P n → 3 ≤ n ∧ n ≤ 5) → 
  ∃ (n : ℕ), (n ≤ 10 / 3 ∧ P n) ∧ n = 3 :=
by
  sorry

end max_principals_in_10_years_l151_151664


namespace total_marks_math_physics_l151_151145

variables (M P C : ℕ)
axiom condition1 : C = P + 20
axiom condition2 : (M + C) / 2 = 45

theorem total_marks_math_physics : M + P = 70 :=
by sorry

end total_marks_math_physics_l151_151145


namespace distance_to_nearest_edge_l151_151838

theorem distance_to_nearest_edge (wall_width picture_width : ℕ) (h1 : wall_width = 19) (h2 : picture_width = 3) (h3 : 2 * x + picture_width = wall_width) :
  x = 8 :=
by
  sorry

end distance_to_nearest_edge_l151_151838


namespace lucy_bought_cakes_l151_151964

theorem lucy_bought_cakes (cookies chocolate total c : ℕ) (h1 : cookies = 4) (h2 : chocolate = 16) (h3 : total = 42) (h4 : c = total - (cookies + chocolate)) : c = 22 := by
  sorry

end lucy_bought_cakes_l151_151964


namespace max_value_is_one_sixteenth_l151_151754

noncomputable def max_value_expression (t : ℝ) : ℝ :=
  (3^t - 4 * t) * t / 9^t

theorem max_value_is_one_sixteenth : 
  ∃ t : ℝ, max_value_expression t = 1 / 16 :=
sorry

end max_value_is_one_sixteenth_l151_151754


namespace michael_total_revenue_l151_151942

def price_large : ℕ := 22
def price_medium : ℕ := 16
def price_small : ℕ := 7

def qty_large : ℕ := 2
def qty_medium : ℕ := 2
def qty_small : ℕ := 3

def total_revenue : ℕ :=
  (price_large * qty_large) +
  (price_medium * qty_medium) +
  (price_small * qty_small)

theorem michael_total_revenue : total_revenue = 97 :=
  by sorry

end michael_total_revenue_l151_151942


namespace quadratic_roots_in_range_l151_151348

theorem quadratic_roots_in_range (a : ℝ) (α β : ℝ)
  (h_eq : ∀ x : ℝ, x^2 + (a^2 + 1) * x + a - 2 = 0)
  (h_root1 : α > 1)
  (h_root2 : β < -1)
  (h_viete_sum : α + β = -(a^2 + 1))
  (h_viete_prod : α * β = a - 2) :
  0 < a ∧ a < 2 :=
  sorry

end quadratic_roots_in_range_l151_151348


namespace not_necessarily_divisor_l151_151646

def consecutive_product (k : ℤ) : ℤ := k * (k + 1) * (k + 2) * (k + 3)

theorem not_necessarily_divisor (k : ℤ) (hk : 8 ∣ consecutive_product k) : ¬ (48 ∣ consecutive_product k) :=
sorry

end not_necessarily_divisor_l151_151646


namespace arithmetic_seq_b3_b6_l151_151623

theorem arithmetic_seq_b3_b6 (b : ℕ → ℕ) (d : ℕ) 
  (h_seq : ∀ n, b n = b 1 + n * d)
  (h_increasing : ∀ n, b (n + 1) > b n)
  (h_b4_b5 : b 4 * b 5 = 30) :
  b 3 * b 6 = 28 := 
sorry

end arithmetic_seq_b3_b6_l151_151623


namespace cost_of_book_l151_151547

theorem cost_of_book (cost_album : ℝ) (discount_rate : ℝ) (cost_CD : ℝ) (cost_book : ℝ) 
  (h1 : cost_album = 20)
  (h2 : discount_rate = 0.30)
  (h3 : cost_CD = cost_album * (1 - discount_rate))
  (h4 : cost_book = cost_CD + 4) :
  cost_book = 18 := by
  sorry

end cost_of_book_l151_151547


namespace checkerboard_inequivalent_color_schemes_l151_151398

/-- 
  We consider a 7x7 checkerboard where two squares are painted yellow, and the remaining 
  are painted green. Two color schemes are equivalent if one can be obtained from 
  the other by rotations of 0°, 90°, 180°, or 270°. We aim to prove that the 
  number of inequivalent color schemes is 312. 
-/
theorem checkerboard_inequivalent_color_schemes : 
  let n := 7
  let total_squares := n * n
  let total_pairs := total_squares.choose 2
  let symmetric_pairs := 24
  let nonsymmetric_pairs := total_pairs - symmetric_pairs
  let unique_symmetric_pairs := symmetric_pairs 
  let unique_nonsymmetric_pairs := nonsymmetric_pairs / 4
  unique_symmetric_pairs + unique_nonsymmetric_pairs = 312 :=
by sorry

end checkerboard_inequivalent_color_schemes_l151_151398


namespace ab_value_l151_151903

theorem ab_value (a b : ℝ) (h1 : a^2 + b^2 = 1) (h2 : a^4 + b^4 = 5 / 8) : ab = (Real.sqrt 3) / 4 :=
by
  sorry

end ab_value_l151_151903


namespace bread_slices_per_loaf_l151_151778

theorem bread_slices_per_loaf (friends: ℕ) (total_loaves : ℕ) (slices_per_friend: ℕ) (total_slices: ℕ)
  (h1 : friends = 10) (h2 : total_loaves = 4) (h3 : slices_per_friend = 6) (h4 : total_slices = friends * slices_per_friend):
  total_slices / total_loaves = 15 :=
by
  sorry

end bread_slices_per_loaf_l151_151778


namespace matrix_satisfies_conditions_l151_151014

open Nat

def is_prime (n : ℕ) : Prop := Nat.Prime n

def matrix : List (List ℕ) :=
  [[6, 8, 9], [1, 7, 3], [4, 2, 5]]

noncomputable def sum_list (lst : List ℕ) : ℕ :=
  lst.foldl (· + ·) 0

def valid_matrix (matrix : List (List ℕ)) : Prop :=
  ∀ row_sum col_sum : ℕ, 
    (row_sum ∈ (matrix.map sum_list) ∧ is_prime row_sum) ∧
    (col_sum ∈ (List.transpose matrix).map sum_list ∧ is_prime col_sum)

theorem matrix_satisfies_conditions : valid_matrix matrix :=
by
  sorry

end matrix_satisfies_conditions_l151_151014


namespace power_rule_for_fractions_calculate_fraction_l151_151371

theorem power_rule_for_fractions (a b : ℚ) (n : ℕ) : (a / b)^n = (a^n) / (b^n) := 
by sorry

theorem calculate_fraction (a b n : ℕ) (h : a = 3 ∧ b = 5 ∧ n = 3) : (a / b)^n = 27 / 125 :=
by
  obtain ⟨ha, hb, hn⟩ := h
  simp [ha, hb, hn, power_rule_for_fractions (3 : ℚ) (5 : ℚ) 3]

end power_rule_for_fractions_calculate_fraction_l151_151371


namespace unripe_oranges_zero_l151_151187

def oranges_per_day (harvest_duration : ℕ) (ripe_oranges_per_day : ℕ) : ℕ :=
  harvest_duration * ripe_oranges_per_day

theorem unripe_oranges_zero
  (harvest_duration : ℕ)
  (ripe_oranges_per_day : ℕ)
  (total_ripe_oranges : ℕ)
  (h1 : harvest_duration = 25)
  (h2 : ripe_oranges_per_day = 82)
  (h3 : total_ripe_oranges = 2050)
  (h4 : oranges_per_day harvest_duration ripe_oranges_per_day = total_ripe_oranges) :
  ∀ unripe_oranges_per_day, unripe_oranges_per_day = 0 :=
by
  sorry

end unripe_oranges_zero_l151_151187


namespace abs_diff_between_sequences_l151_151788

def sequence_C (n : ℕ) : ℤ := 50 + 12 * (n - 1)
def sequence_D (n : ℕ) : ℤ := 50 + (-8) * (n - 1)

theorem abs_diff_between_sequences :
  |sequence_C 31 - sequence_D 31| = 600 :=
by
  sorry

end abs_diff_between_sequences_l151_151788


namespace moles_KOH_combined_l151_151560

-- Define the number of moles of KI produced
def moles_KI_produced : ℕ := 3

-- Define the molar ratio from the balanced chemical equation
def molar_ratio_KOH_NH4I_KI : ℕ := 1

-- The number of moles of KOH combined to produce the given moles of KI
theorem moles_KOH_combined (moles_KOH moles_NH4I : ℕ) (h : moles_NH4I = 3) 
  (h_produced : moles_KI_produced = 3) (ratio : molar_ratio_KOH_NH4I_KI = 1) :
  moles_KOH = 3 :=
by {
  -- Placeholder for proof, use sorry to skip proving
  sorry
}

end moles_KOH_combined_l151_151560


namespace intersection_points_l151_151022

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2
def parabola2 (x : ℝ) : ℝ := 9 * x^2 + 6 * x + 2

theorem intersection_points :
  {p : ℝ × ℝ | parabola1 p.1 = p.2 ∧ parabola2 p.1 = p.2} = 
  {(-5/3, 17), (0, 2)} :=
by
  sorry

end intersection_points_l151_151022


namespace fewest_keystrokes_One_to_410_l151_151429

noncomputable def fewest_keystrokes (start : ℕ) (target : ℕ) : ℕ :=
if target = 410 then 10 else sorry

theorem fewest_keystrokes_One_to_410 : fewest_keystrokes 1 410 = 10 :=
by
  sorry

end fewest_keystrokes_One_to_410_l151_151429


namespace max_xy_of_conditions_l151_151569

noncomputable def max_xy : ℝ := 37.5

theorem max_xy_of_conditions (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 10 * x + 15 * y = 150) (h4 : x^2 + y^2 ≤ 100) :
  xy ≤ max_xy :=
by sorry

end max_xy_of_conditions_l151_151569


namespace opposite_sign_pairs_l151_151198

theorem opposite_sign_pairs :
  ¬ ((- 2 ^ 3 < 0) ∧ (- (2 ^ 3) > 0)) ∧
  ¬ (|-4| < 0 ∧ -(-4) > 0) ∧
  ((- 3 ^ 4 < 0 ∧ (-(3 ^ 4)) = 81)) ∧
  ¬ (10 ^ 2 < 0 ∧ 2 ^ 10 > 0) :=
by
  sorry

end opposite_sign_pairs_l151_151198


namespace exists_nat_sum_of_squares_two_ways_l151_151411

theorem exists_nat_sum_of_squares_two_ways :
  ∃ n : ℕ, n < 100 ∧ ∃ a b c d : ℕ, a ≠ b ∧ c ≠ d ∧ n = a^2 + b^2 ∧ n = c^2 + d^2 :=
by {
  sorry
}

end exists_nat_sum_of_squares_two_ways_l151_151411


namespace f_2016_eq_one_third_l151_151736

noncomputable def f (x : ℕ) : ℝ := sorry

axiom f_one : f 1 = 2
axiom f_recurrence : ∀ x : ℕ, f (x + 1) = (1 + f x) / (1 - f x)

theorem f_2016_eq_one_third : f 2016 = 1 / 3 := sorry

end f_2016_eq_one_third_l151_151736


namespace boxes_per_day_l151_151380

theorem boxes_per_day (apples_per_box fewer_apples_per_day total_apples_two_weeks : ℕ)
  (h1 : apples_per_box = 40)
  (h2 : fewer_apples_per_day = 500)
  (h3 : total_apples_two_weeks = 24500) :
  (∃ x : ℕ, (7 * apples_per_box * x + 7 * (apples_per_box * x - fewer_apples_per_day) = total_apples_two_weeks) ∧ x = 50) := 
sorry

end boxes_per_day_l151_151380


namespace radius_large_circle_l151_151011

-- Definitions for the conditions
def radius_small_circle : ℝ := 2

def is_tangent_externally (r1 r2 : ℝ) : Prop := -- Definition of external tangency
  r1 + r2 = 4

def is_tangent_internally (R r : ℝ) : Prop := -- Definition of internal tangency
  R - r = 4

-- Setting up the property we need to prove: large circle radius
theorem radius_large_circle
  (R r : ℝ)
  (h1 : r = radius_small_circle)
  (h2 : is_tangent_externally r r)
  (h3 : is_tangent_externally r r)
  (h4 : is_tangent_externally r r)
  (h5 : is_tangent_externally r r)
  (h6 : is_tangent_internally R r) :
  R = 4 :=
by sorry

end radius_large_circle_l151_151011


namespace solve_eqs_l151_151556

theorem solve_eqs (x y : ℝ) 
  (h1 : x^2 + y^2 = 2)
  (h2 : x^2 / (2 - y) + y^2 / (2 - x) = 2) :
  x = 1 ∧ y = 1 :=
by
  sorry

end solve_eqs_l151_151556


namespace allan_balloons_l151_151069

theorem allan_balloons (x : ℕ) : 
  (2 + x) + 1 = 6 → x = 3 :=
by
  intro h
  linarith

end allan_balloons_l151_151069


namespace find_a_l151_151688

theorem find_a (a : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f x = abs (2 * x - a) + a)
  (h2 : ∀ x : ℝ, f x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) : 
  a = 1 := by
  sorry

end find_a_l151_151688


namespace lcm_gcd_product_l151_151378

theorem lcm_gcd_product (a b : ℕ) (ha : a = 36) (hb : b = 60) : 
  Nat.lcm a b * Nat.gcd a b = 2160 :=
by
  rw [ha, hb]
  sorry

end lcm_gcd_product_l151_151378


namespace standard_eq_circle_l151_151449

noncomputable def circle_eq (x y : ℝ) (r : ℝ) : Prop :=
  (x - 4)^2 + (y - 4)^2 = 16 ∨ (x - 1)^2 + (y + 1)^2 = 1

theorem standard_eq_circle {x y : ℝ}
  (h1 : 5 * x - 3 * y = 8)
  (h2 : abs x = abs y) :
  ∃ r : ℝ, circle_eq x y r :=
by {
  sorry
}

end standard_eq_circle_l151_151449


namespace increase_average_by_3_l151_151452

theorem increase_average_by_3 (x : ℕ) (average_initial : ℕ := 32) (matches_initial : ℕ := 10) (score_11th_match : ℕ := 65) :
  (matches_initial * average_initial + score_11th_match = 11 * (average_initial + x)) → x = 3 := 
sorry

end increase_average_by_3_l151_151452


namespace remainder_of_sum_mod_18_l151_151144

theorem remainder_of_sum_mod_18 :
  let nums := [85, 86, 87, 88, 89, 90, 91, 92, 93]
  let sum_nums := nums.sum
  let product := 90 * sum_nums
  product % 18 = 10 :=
by
  sorry

end remainder_of_sum_mod_18_l151_151144


namespace find_x_l151_151955

theorem find_x (x : ℝ) : 
  (∀ (y : ℝ), 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0) ↔ x = 3 / 2 := sorry

end find_x_l151_151955


namespace candy_problem_l151_151526

theorem candy_problem (n : ℕ) (h : n ∈ [2, 5, 9, 11, 14]) : ¬(23 - n) % 3 ≠ 0 → n = 9 := by
  sorry

end candy_problem_l151_151526


namespace exchange_rate_l151_151504

def jackPounds : ℕ := 42
def jackEuros : ℕ := 11
def jackYen : ℕ := 3000
def poundsPerYen : ℕ := 100
def totalYen : ℕ := 9400

theorem exchange_rate :
  ∃ (x : ℕ), 100 * jackPounds + 100 * jackEuros * x + jackYen = totalYen ∧ x = 2 :=
by
  sorry

end exchange_rate_l151_151504


namespace perpendicular_line_through_point_l151_151365

theorem perpendicular_line_through_point (x y : ℝ) : (x, y) = (0, -3) ∧ (∀ x y : ℝ, 2 * x + 3 * y - 6 = 0) → 3 * x - 2 * y - 6 = 0 :=
by
  sorry

end perpendicular_line_through_point_l151_151365


namespace harriet_ran_48_miles_l151_151277

def total_distance : ℕ := 195
def katarina_distance : ℕ := 51
def equal_distance (n : ℕ) : Prop := (total_distance - katarina_distance) = 3 * n
def harriet_distance : ℕ := 48

theorem harriet_ran_48_miles
  (total_eq : total_distance = 195)
  (kat_eq : katarina_distance = 51)
  (equal_dist_eq : equal_distance harriet_distance) :
  harriet_distance = 48 :=
by
  sorry

end harriet_ran_48_miles_l151_151277


namespace find_smallest_number_l151_151756

theorem find_smallest_number
  (a1 a2 a3 a4 : ℕ)
  (h1 : (a1 + a2 + a3 + a4) / 4 = 30)
  (h2 : a2 = 28)
  (h3 : a2 = 35 - 7) :
  a1 = 27 :=
sorry

end find_smallest_number_l151_151756


namespace equivalent_statement_l151_151379

theorem equivalent_statement (x y z w : ℝ)
  (h : (2 * x + y) / (y + z) = (z + w) / (w + 2 * x)) :
  (x = z / 2 ∨ 2 * x + y + z + w = 0) :=
sorry

end equivalent_statement_l151_151379


namespace remi_water_intake_l151_151683

def bottle_capacity := 20
def daily_refills := 3
def num_days := 7
def spill1 := 5
def spill2 := 8

def daily_intake := daily_refills * bottle_capacity
def total_intake_without_spill := daily_intake * num_days
def total_spill := spill1 + spill2
def total_intake_with_spill := total_intake_without_spill - total_spill

theorem remi_water_intake : total_intake_with_spill = 407 := 
by
  -- Provide proof here
  sorry

end remi_water_intake_l151_151683


namespace perfect_square_sum_l151_151275

-- Define the numbers based on the given conditions
def A (n : ℕ) : ℕ := 4 * (10^(2 * n) - 1) / 9
def B (n : ℕ) : ℕ := 2 * (10^(n + 1) - 1) / 9
def C (n : ℕ) : ℕ := 8 * (10^n - 1) / 9

-- Define the main theorem to be proved
theorem perfect_square_sum (n : ℕ) : 
  ∃ k, A n + B n + C n + 7 = k * k :=
sorry

end perfect_square_sum_l151_151275


namespace trigonometric_identity_l151_151244

theorem trigonometric_identity :
  let cos_30 := (Real.sqrt 3) / 2
  let sin_60 := (Real.sqrt 3) / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 1 := by
  sorry

end trigonometric_identity_l151_151244


namespace correct_calculation_l151_151057

-- Definitions for conditions
def cond_A (x y : ℝ) : Prop := 3 * x + 4 * y = 7 * x * y
def cond_B (x : ℝ) : Prop := 5 * x - 2 * x = 3 * x ^ 2
def cond_C (y : ℝ) : Prop := 7 * y ^ 2 - 5 * y ^ 2 = 2
def cond_D (a b : ℝ) : Prop := 6 * a ^ 2 * b - b * a ^ 2 = 5 * a ^ 2 * b

-- Proof statement using conditions
theorem correct_calculation (a b : ℝ) : cond_D a b :=
by
  unfold cond_D
  sorry

end correct_calculation_l151_151057


namespace find_a_range_l151_151722

-- Definitions of sets A and B
def A (a x : ℝ) : Prop := a + 1 ≤ x ∧ x ≤ 2 * a - 1
def B (x : ℝ) : Prop := x ≤ 3 ∨ x > 5

-- Condition p: A ⊆ B
def p (a : ℝ) : Prop := ∀ x, A a x → B x

-- The function f(x) = x^2 - 2ax + 1
def f (a x : ℝ) : ℝ := x^2 - 2 * a * x + 1

-- Condition q: f(x) is increasing on (1/2, +∞)
def q (a : ℝ) : Prop := ∀ x y, 1/2 < x → x < y → f a x ≤ f a y

-- The given propositions
def prop1 (a : ℝ) : Prop := p a
def prop2 (a : ℝ) : Prop := q a

-- Given conditions
def given_conditions (a : ℝ) : Prop := ¬ (prop1 a ∧ prop2 a) ∧ (prop1 a ∨ prop2 a)

-- Proof statement: Find the range of values for 'a' according to the given conditions
theorem find_a_range (a : ℝ) :
  given_conditions a →
  (1/2 < a ∧ a ≤ 2) ∨ (4 < a) :=
sorry

end find_a_range_l151_151722


namespace billed_minutes_l151_151507

noncomputable def John_bill (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) : ℝ :=
  (total_bill - monthly_fee) / cost_per_minute

theorem billed_minutes : ∀ (monthly_fee cost_per_minute total_bill : ℝ), 
  monthly_fee = 5 → 
  cost_per_minute = 0.25 → 
  total_bill = 12.02 → 
  John_bill monthly_fee cost_per_minute total_bill = 28 :=
by
  intros monthly_fee cost_per_minute total_bill hf hm hb
  rw [hf, hm, hb, John_bill]
  norm_num
  sorry

end billed_minutes_l151_151507


namespace no_negative_roots_l151_151008

theorem no_negative_roots (x : ℝ) : 4 * x^4 - 7 * x^3 - 20 * x^2 - 13 * x + 25 ≠ 0 ∨ x ≥ 0 := 
sorry

end no_negative_roots_l151_151008


namespace ratio_cubed_eq_27_l151_151822

theorem ratio_cubed_eq_27 : (81000^3) / (27000^3) = 27 := 
by
  sorry

end ratio_cubed_eq_27_l151_151822


namespace bacteria_growth_l151_151503

-- Define the original and current number of bacteria
def original_bacteria := 600
def current_bacteria := 8917

-- Define the increase in bacteria count
def additional_bacteria := 8317

-- Prove the statement
theorem bacteria_growth : current_bacteria - original_bacteria = additional_bacteria :=
by {
    -- Lean will require the proof here, so we use sorry for now 
    sorry
}

end bacteria_growth_l151_151503


namespace marble_prob_l151_151738

theorem marble_prob (a c x y p q : ℕ) (h1 : 2 * a + c = 36) 
    (h2 : (x / a) * (x / a) * (y / c) = 1 / 3) 
    (h3 : (a - x) / a * (a - x) / a * (c - y) / c = p / q) 
    (hpq_rel_prime : Nat.gcd p q = 1) : p + q = 65 := by
  sorry

end marble_prob_l151_151738


namespace exists_geometric_weak_arithmetic_l151_151302

theorem exists_geometric_weak_arithmetic (m : ℕ) (hm : 3 ≤ m) :
  ∃ (k : ℕ) (a : ℕ → ℕ), 
    (∀ i, 1 ≤ i → i ≤ m → a i = k^(m - i)*(k + 1)^(i - 1)) ∧
    ((∀ i, 1 ≤ i → i < m → a i < a (i + 1)) ∧ 
    ∃ (x : ℕ → ℕ) (d : ℕ), 
      (x 0 ≤ a 1 ∧ 
      ∀ i, 1 ≤ i → i < m → (x i ≤ a (i + 1) ∧ a (i + 1) < x (i + 1)) ∧ 
      ∀ i, 0 ≤ i → i < m - 1 → x (i + 1) - x i = d)) :=
by
  sorry

end exists_geometric_weak_arithmetic_l151_151302


namespace cody_ate_dumplings_l151_151050

theorem cody_ate_dumplings (initial_dumplings remaining_dumplings : ℕ) (h1 : initial_dumplings = 14) (h2 : remaining_dumplings = 7) : initial_dumplings - remaining_dumplings = 7 :=
by
  sorry

end cody_ate_dumplings_l151_151050


namespace geometric_series_sum_l151_151580

theorem geometric_series_sum {a r : ℚ} (n : ℕ) (h_a : a = 3/4) (h_r : r = 3/4) (h_n : n = 8) : 
       a * (1 - r^n) / (1 - r) = 176925 / 65536 :=
by
  -- Utilizing the provided conditions
  have h_a := h_a
  have h_r := h_r
  have h_n := h_n
  -- Proving the theorem using sorry as a placeholder for the detailed steps
  sorry

end geometric_series_sum_l151_151580


namespace find_f_expression_l151_151143

theorem find_f_expression (f : ℝ → ℝ) (h : ∀ x, f (x - 1) = x^2) : 
  ∀ x, f x = x^2 + 2 * x + 1 :=
by
  sorry

end find_f_expression_l151_151143


namespace lineup_possibilities_l151_151932

theorem lineup_possibilities (total_players : ℕ) (all_stars_in_lineup : ℕ) (injured_player : ℕ) :
  total_players = 15 ∧ all_stars_in_lineup = 2 ∧ injured_player = 1 →
  Nat.choose 12 4 = 495 :=
by
  intro h
  sorry

end lineup_possibilities_l151_151932


namespace model_tower_height_l151_151441

theorem model_tower_height (real_height : ℝ) (real_volume : ℝ) (model_volume : ℝ) (h_real : real_height = 60) (v_real : real_volume = 200000) (v_model : model_volume = 0.2) :
  real_height / (real_volume / model_volume)^(1/3) = 0.6 :=
by
  rw [h_real, v_real, v_model]
  norm_num
  sorry

end model_tower_height_l151_151441


namespace triangle_area_l151_151401

theorem triangle_area (A B C : ℝ) (AB AC : ℝ) (A_angle : ℝ) (h1 : A_angle = π / 6)
  (h2 : AB * AC * Real.cos A_angle = Real.tan A_angle) :
  1 / 2 * AB * AC * Real.sin A_angle = 1 / 6 :=
by
  sorry

end triangle_area_l151_151401


namespace curved_surface_area_of_cone_l151_151966

noncomputable def slant_height : ℝ := 22
noncomputable def radius : ℝ := 7
noncomputable def pi : ℝ := Real.pi

theorem curved_surface_area_of_cone :
  abs (pi * radius * slant_height - 483.22) < 0.01 := 
by
  sorry

end curved_surface_area_of_cone_l151_151966


namespace cube_face_sum_l151_151408

theorem cube_face_sum (a b c d e f : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : e > 0) (h6 : f > 0) :
  (a * b * c + a * e * c + a * b * f + a * e * f + d * b * c + d * e * c + d * b * f + d * e * f = 1287) →
  (a + d + b + e + c + f = 33) :=
by
  sorry

end cube_face_sum_l151_151408


namespace distance_between_foci_correct_l151_151148

/-- Define the given conditions for the ellipse -/
def ellipse_center : ℝ × ℝ := (3, -2)
def semi_major_axis : ℝ := 7
def semi_minor_axis : ℝ := 3

/-- Define the distance between the foci of the ellipse -/
noncomputable def distance_between_foci : ℝ :=
  2 * Real.sqrt (semi_major_axis ^ 2 - semi_minor_axis ^ 2)

theorem distance_between_foci_correct :
  distance_between_foci = 4 * Real.sqrt 10 := by
  sorry

end distance_between_foci_correct_l151_151148


namespace stuffed_animal_cost_l151_151667

variables 
  (M S A A_single C : ℝ)
  (Coupon_discount : ℝ)
  (Maximum_budget : ℝ)

noncomputable def conditions : Prop :=
  M = 6 ∧
  M = 3 * S ∧
  M = A / 4 ∧
  A_single = A / 2 ∧
  C = A_single / 2 ∧
  C = 2 * S ∧
  Coupon_discount = 0.10 ∧
  Maximum_budget = 30

theorem stuffed_animal_cost (h : conditions M S A A_single C Coupon_discount Maximum_budget) :
  A_single = 12 :=
sorry

end stuffed_animal_cost_l151_151667


namespace game_returns_to_A_after_three_rolls_l151_151770

theorem game_returns_to_A_after_three_rolls :
  (∃ i j k : ℕ, 1 ≤ i ∧ i ≤ 6 ∧ 1 ≤ j ∧ j ≤ 6 ∧ 1 ≤ k ∧ k ≤ 6 ∧ (i + j + k) % 12 = 0) → 
  true :=
by
  sorry

end game_returns_to_A_after_three_rolls_l151_151770


namespace positive_difference_is_correct_l151_151627

/-- Angela's compounded interest parameters -/
def angela_initial_deposit : ℝ := 9000
def angela_interest_rate : ℝ := 0.08
def years : ℕ := 25

/-- Bob's simple interest parameters -/
def bob_initial_deposit : ℝ := 11000
def bob_interest_rate : ℝ := 0.09

/-- Compound interest calculation for Angela -/
def angela_balance : ℝ := angela_initial_deposit * (1 + angela_interest_rate) ^ years

/-- Simple interest calculation for Bob -/
def bob_balance : ℝ := bob_initial_deposit * (1 + bob_interest_rate * years)

/-- Difference calculation -/
def balance_difference : ℝ := angela_balance - bob_balance

/-- The positive difference between their balances to the nearest dollar -/
theorem positive_difference_is_correct :
  abs (round balance_difference) = 25890 :=
by
  sorry

end positive_difference_is_correct_l151_151627


namespace valid_documents_count_l151_151552

-- Definitions based on the conditions
def total_papers : ℕ := 400
def invalid_percentage : ℝ := 0.40
def valid_percentage : ℝ := 1.0 - invalid_percentage

-- Question and answer formalized as a theorem
theorem valid_documents_count : total_papers * valid_percentage = 240 := by
  sorry

end valid_documents_count_l151_151552


namespace solve_equation_l151_151689

theorem solve_equation (x : ℝ) (h : (x^2 - x + 2) / (x - 1) = x + 3) : x = 5 / 3 := 
by sorry

end solve_equation_l151_151689


namespace isosceles_base_length_l151_151026

theorem isosceles_base_length (b : ℝ) (h1 : 7 + 7 + b = 23) : b = 9 :=
sorry

end isosceles_base_length_l151_151026


namespace calc_addition_even_odd_probability_calc_addition_multiplication_even_probability_l151_151869

-- Define the necessary probability events and conditions.
variable {p : ℝ} (calc_action : ℕ → ℝ)

-- Condition: initially, the display shows 0.
def initial_display : ℕ := 0

-- Events for part (a): addition only, randomly chosen numbers from 0 to 9.
def random_addition_event (n : ℕ) : Prop := n % 2 = 0 ∨ n % 2 = 1

-- Events for part (b): both addition and multiplication allowed.
def random_operation_event (n : ℕ) : Prop := (n % 2 = 0 ∧ n % 2 = 1) ∨ -- addition
                                               (n ≠ 0 ∧ n % 2 = 1 ∧ (n/2) % 2 = 1) -- multiplication

-- Statements to be proved based on above definitions.
theorem calc_addition_even_odd_probability :
  calc_action 0 = 1 / 2 → random_addition_event initial_display := sorry

theorem calc_addition_multiplication_even_probability :
  calc_action (initial_display + 1) > 1 / 2 → random_operation_event (initial_display + 1) := sorry

end calc_addition_even_odd_probability_calc_addition_multiplication_even_probability_l151_151869


namespace cars_gain_one_passenger_each_l151_151021

-- Conditions
def initial_people_per_car : ℕ := 3 -- 2 passengers + 1 driver
def total_cars : ℕ := 20
def total_people_at_end : ℕ := 80

-- Question (equivalent to "answer")
theorem cars_gain_one_passenger_each :
  (total_people_at_end = total_cars * initial_people_per_car + total_cars) →
  total_people_at_end - total_cars * initial_people_per_car = total_cars :=
by sorry

end cars_gain_one_passenger_each_l151_151021


namespace fraction_ordering_l151_151373

theorem fraction_ordering :
  (4 / 13) < (12 / 37) ∧ (12 / 37) < (15 / 31) ∧ (4 / 13) < (15 / 31) :=
by sorry

end fraction_ordering_l151_151373


namespace sqrt_square_l151_151002

theorem sqrt_square (x : ℝ) (h_nonneg : 0 ≤ x) : (Real.sqrt x)^2 = x :=
by
  sorry

example : (Real.sqrt 25)^2 = 25 :=
by
  exact sqrt_square 25 (by norm_num)

end sqrt_square_l151_151002


namespace truck_loading_time_l151_151752

theorem truck_loading_time :
  let worker1_rate := (1:ℝ) / 6
  let worker2_rate := (1:ℝ) / 5
  let combined_rate := worker1_rate + worker2_rate
  (combined_rate != 0) → 
  (1 / combined_rate = (30:ℝ) / 11) :=
by
  sorry

end truck_loading_time_l151_151752


namespace find_a_if_f_is_odd_l151_151890

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(-x) * (1 - a^x)

theorem find_a_if_f_is_odd (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  (∀ x : ℝ, f a (-x) = - f a x) → a = 4 :=
by
  sorry

end find_a_if_f_is_odd_l151_151890


namespace roots_are_positive_integers_implies_r_values_l151_151861

theorem roots_are_positive_integers_implies_r_values (r x : ℕ) (h : (r * x^2 - (2 * r + 7) * x + (r + 7) = 0) ∧ (x > 0)) :
  r = 7 ∨ r = 0 ∨ r = 1 :=
by
  sorry

end roots_are_positive_integers_implies_r_values_l151_151861


namespace sum_of_ages_l151_151857

theorem sum_of_ages (a b c d e : ℕ) 
  (h1 : 1 ≤ a ∧ a ≤ 9) 
  (h2 : 1 ≤ b ∧ b ≤ 9) 
  (h3 : 1 ≤ c ∧ c ≤ 9) 
  (h4 : 1 ≤ d ∧ d ≤ 9) 
  (h5 : 1 ≤ e ∧ e ≤ 9) 
  (h6 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h7 : a * b = 28 ∨ a * c = 28 ∨ a * d = 28 ∨ a * e = 28 ∨ b * c = 28 ∨ b * d = 28 ∨ b * e = 28 ∨ c * d = 28 ∨ c * e = 28 ∨ d * e = 28)
  (h8 : a * b = 20 ∨ a * c = 20 ∨ a * d = 20 ∨ a * e = 20 ∨ b * c = 20 ∨ b * d = 20 ∨ b * e = 20 ∨ c * d = 20 ∨ c * e = 20 ∨ d * e = 20)
  (h9 : a + b = 14 ∨ a + c = 14 ∨ a + d = 14 ∨ a + e = 14 ∨ b + c = 14 ∨ b + d = 14 ∨ b + e = 14 ∨ c + d = 14 ∨ c + e = 14 ∨ d + e = 14) 
  : a + b + c + d + e = 25 :=
by
  sorry

end sum_of_ages_l151_151857


namespace suitable_bases_for_346_l151_151440

theorem suitable_bases_for_346 (b : ℕ) (hb : b^3 ≤ 346 ∧ 346 < b^4 ∧ (346 % b) % 2 = 0) : b = 6 ∨ b = 7 :=
sorry

end suitable_bases_for_346_l151_151440


namespace julia_tuesday_kids_l151_151985

theorem julia_tuesday_kids :
  ∃ x : ℕ, (∃ y : ℕ, y = 6 ∧ y = x + 1) → x = 5 := 
by
  sorry

end julia_tuesday_kids_l151_151985


namespace total_cost_l151_151735

-- Defining the prices based on the given conditions
def price_smartphone : ℕ := 300
def price_pc : ℕ := price_smartphone + 500
def price_tablet : ℕ := price_smartphone + price_pc

-- The theorem to prove the total cost of buying one of each product
theorem total_cost : price_smartphone + price_pc + price_tablet = 2200 :=
by
  sorry

end total_cost_l151_151735


namespace solve_sine_equation_l151_151357

theorem solve_sine_equation (x : ℝ) (k : ℤ) (h : |Real.sin x| ≠ 1) :
  (8.477 * ((∑' n, Real.sin x ^ n) / (∑' n, ((-1 : ℝ) * Real.sin x) ^ n)) = 4 / (1 + Real.tan x ^ 2)) 
  ↔ (x = (-1)^k * (Real.pi / 6) + k * Real.pi) :=
by
  sorry

end solve_sine_equation_l151_151357


namespace chicken_nuggets_cost_l151_151352

theorem chicken_nuggets_cost :
  ∀ (nuggets_ordered boxes_cost : ℕ) (nuggets_per_box : ℕ),
  nuggets_ordered = 100 →
  nuggets_per_box = 20 →
  boxes_cost = 4 →
  (nuggets_ordered / nuggets_per_box) * boxes_cost = 20 :=
by
  intros nuggets_ordered boxes_cost nuggets_per_box h1 h2 h3
  sorry

end chicken_nuggets_cost_l151_151352


namespace LaKeisha_needs_to_mow_more_sqft_l151_151639

noncomputable def LaKeisha_price_per_sqft : ℝ := 0.10
noncomputable def LaKeisha_book_cost : ℝ := 150
noncomputable def LaKeisha_mowed_sqft : ℕ := 3 * 20 * 15
noncomputable def LaKeisha_earnings_so_far : ℝ := LaKeisha_mowed_sqft * LaKeisha_price_per_sqft

theorem LaKeisha_needs_to_mow_more_sqft (additional_sqft_needed : ℝ) :
  additional_sqft_needed = (LaKeisha_book_cost - LaKeisha_earnings_so_far) / LaKeisha_price_per_sqft → 
  additional_sqft_needed = 600 :=
by
  sorry

end LaKeisha_needs_to_mow_more_sqft_l151_151639


namespace find_d_l151_151250

theorem find_d (d : ℝ) (h₁ : ∃ x, x = ⌊d⌋ ∧ 3 * x^2 + 19 * x - 84 = 0)
                (h₂ : ∃ y, y = d - ⌊d⌋ ∧ 5 * y^2 - 28 * y + 12 = 0 ∧ 0 ≤ y ∧ y < 1) :
  d = 3.2 :=
by
  sorry

end find_d_l151_151250


namespace solve_for_x_l151_151224

theorem solve_for_x (x : ℝ) (h : 5 + 3.5 * x = 2.5 * x - 25) : x = -30 :=
sorry

end solve_for_x_l151_151224


namespace range_of_a_l151_151257

-- Define the inequality condition
def inequality (a x : ℝ) : Prop := (a-2)*x^2 + 2*(a-2)*x < 4

-- The main theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, inequality a x) ↔ (-2 : ℝ) < a ∧ a ≤ 2 := by
  sorry

end range_of_a_l151_151257


namespace smallest_m_for_no_real_solution_l151_151383

theorem smallest_m_for_no_real_solution : 
  (∀ x : ℝ, ∀ m : ℝ, (m * x^2 - 3 * x + 1 = 0) → false) ↔ (m ≥ 3) :=
by
  sorry

end smallest_m_for_no_real_solution_l151_151383


namespace problem_solution_l151_151346

theorem problem_solution (y : Fin 8 → ℝ)
  (h1 : y 0 + 4 * y 1 + 9 * y 2 + 16 * y 3 + 25 * y 4 + 36 * y 5 + 49 * y 6 + 64 * y 7 = 2)
  (h2 : 4 * y 0 + 9 * y 1 + 16 * y 2 + 25 * y 3 + 36 * y 4 + 49 * y 5 + 64 * y 6 + 81 * y 7 = 15)
  (h3 : 9 * y 0 + 16 * y 1 + 25 * y 2 + 36 * y 3 + 49 * y 4 + 64 * y 5 + 81 * y 6 + 100 * y 7 = 156)
  (h4 : 16 * y 0 + 25 * y 1 + 36 * y 2 + 49 * y 3 + 64 * y 4 + 81 * y 5 + 100 * y 6 + 121 * y 7 = 1305) :
  25 * y 0 + 36 * y 1 + 49 * y 2 + 64 * y 3 + 81 * y 4 + 100 * y 5 + 121 * y 6 + 144 * y 7 = 4360 :=
sorry

end problem_solution_l151_151346


namespace pizzas_difference_l151_151803

def pizzas (craig_first_day craig_second_day heather_first_day heather_second_day total_pizzas: ℕ) :=
  heather_first_day = 4 * craig_first_day ∧
  heather_second_day = craig_second_day - 20 ∧
  craig_first_day = 40 ∧
  craig_first_day + heather_first_day + craig_second_day + heather_second_day = total_pizzas

theorem pizzas_difference :
  ∀ (craig_first_day craig_second_day heather_first_day heather_second_day : ℕ),
  pizzas craig_first_day craig_second_day heather_first_day heather_second_day 380 →
  craig_second_day - craig_first_day = 60 :=
by
  intros craig_first_day craig_second_day heather_first_day heather_second_day h
  sorry

end pizzas_difference_l151_151803


namespace Jacob_age_is_3_l151_151598

def Phoebe_age : ℕ := sorry
def Rehana_age : ℕ := 25
def Jacob_age (P : ℕ) : ℕ := 3 * P / 5

theorem Jacob_age_is_3 (P : ℕ) (h1 : Rehana_age + 5 = 3 * (P + 5)) (h2 : Rehana_age = 25) (h3 : Jacob_age P = 3) : Jacob_age P = 3 := by {
  sorry
}

end Jacob_age_is_3_l151_151598


namespace trigonometric_identity_l151_151690

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 1) : 
  1 - 2 * Real.sin α * Real.cos α - 3 * (Real.cos α)^2 = -3 / 2 :=
sorry

end trigonometric_identity_l151_151690


namespace number_of_ways_to_choose_officers_l151_151851

-- Define the number of boys and girls.
def num_boys : ℕ := 12
def num_girls : ℕ := 13

-- Define the total number of boys and girls.
def num_members : ℕ := num_boys + num_girls

-- Calculate the number of ways to choose the president, vice-president, and secretary with given conditions.
theorem number_of_ways_to_choose_officers : 
  (num_boys * num_girls * (num_boys - 1)) + (num_girls * num_boys * (num_girls - 1)) = 3588 :=
by
  -- The first part calculates the ways when the president is a boy.
  -- The second part calculates the ways when the president is a girl.
  sorry

end number_of_ways_to_choose_officers_l151_151851


namespace equation1_solution_equation2_solution_l151_151813

theorem equation1_solution (x : ℝ) (h : 3 * x - 1 = x + 7) : x = 4 := by
  sorry

theorem equation2_solution (x : ℝ) (h : (x + 1) / 2 - 1 = (1 - 2 * x) / 3) : x = 5 / 7 := by
  sorry

end equation1_solution_equation2_solution_l151_151813


namespace mode_of_dataset_l151_151375

def dataset : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

def frequency (n : ℕ) (l : List ℕ) : ℕ :=
  l.count n

theorem mode_of_dataset :
  (∀ n ≠ 3, frequency n dataset ≤ 3) ∧ frequency 3 dataset = 3 :=
by
  sorry

end mode_of_dataset_l151_151375


namespace no_contradiction_to_thermodynamics_l151_151533

variables (T_handle T_environment : ℝ) (cold_water : Prop)
noncomputable def increased_grip_increases_heat_transfer (A1 A2 : ℝ) (k : ℝ) (dT dx : ℝ) : Prop :=
  A2 > A1 ∧ k * (A2 - A1) * (dT / dx) > 0

theorem no_contradiction_to_thermodynamics (T_handle T_environment : ℝ) (cold_water : Prop) :
  T_handle > T_environment ∧ cold_water →
  ∃ A1 A2 k dT dx, T_handle > T_environment ∧ k > 0 ∧ dT > 0 ∧ dx > 0 → increased_grip_increases_heat_transfer A1 A2 k dT dx :=
sorry

end no_contradiction_to_thermodynamics_l151_151533


namespace ratio_planes_bisect_volume_l151_151079

-- Definitions
def n : ℕ := 6
def m : ℕ := 20

-- Statement to prove
theorem ratio_planes_bisect_volume : (n / m : ℚ) = 3 / 10 := by
  sorry

end ratio_planes_bisect_volume_l151_151079


namespace jung_kook_blue_balls_l151_151399

def num_boxes := 2
def blue_balls_per_box := 5
def total_blue_balls := num_boxes * blue_balls_per_box

theorem jung_kook_blue_balls : total_blue_balls = 10 :=
by
  sorry

end jung_kook_blue_balls_l151_151399


namespace arithmetic_sequence_is_a_l151_151182

theorem arithmetic_sequence_is_a
  (a : ℚ) (d : ℚ)
  (h1 : 140 + d = a)
  (h2 : a + d = 45 / 28)
  (h3 : a > 0) :
  a = 3965 / 56 :=
by
  sorry

end arithmetic_sequence_is_a_l151_151182


namespace curve_defined_by_r_eq_4_is_circle_l151_151048

theorem curve_defined_by_r_eq_4_is_circle : ∀ θ : ℝ, ∃ r : ℝ, r = 4 → ∀ θ : ℝ, r = 4 :=
by
  sorry

end curve_defined_by_r_eq_4_is_circle_l151_151048


namespace emails_in_morning_and_afternoon_l151_151481

-- Conditions
def morning_emails : Nat := 5
def afternoon_emails : Nat := 8

-- Theorem statement
theorem emails_in_morning_and_afternoon : morning_emails + afternoon_emails = 13 := by
  -- Proof goes here, but adding sorry for now
  sorry

end emails_in_morning_and_afternoon_l151_151481


namespace sequence_an_value_l151_151642

theorem sequence_an_value (a : ℕ → ℕ) (S : ℕ → ℕ)
  (hS : ∀ n, 4 * S n = (a n - 1) * (a n + 3))
  (h_pos : ∀ n, 0 < a n)
  (n_nondec : ∀ n, a (n + 1) - a n = 2) :
  a 1005 = 2011 := 
sorry

end sequence_an_value_l151_151642


namespace find_m_l151_151004

theorem find_m (θ₁ θ₂ : ℝ) (l : ℝ → ℝ) (m : ℕ) 
  (hθ₁ : θ₁ = Real.pi / 100) 
  (hθ₂ : θ₂ = Real.pi / 75)
  (hl : ∀ x, l x = x / 4) 
  (R : ((ℝ → ℝ) → (ℝ → ℝ)))
  (H_R : ∀ l, R l = (sorry : ℝ → ℝ)) 
  (R_n : ℕ → (ℝ → ℝ) → (ℝ → ℝ)) 
  (H_R1 : R_n 1 l = R l) 
  (H_Rn : ∀ n, R_n (n + 1) l = R (R_n n l)) :
  m = 1500 :=
sorry

end find_m_l151_151004


namespace largest_area_of_rotating_triangle_l151_151943

def Point := (ℝ × ℝ)

def A : Point := (0, 0)
def B : Point := (13, 0)
def C : Point := (21, 0)

def line (P : Point) (slope : ℝ) (x : ℝ) : ℝ := P.2 + slope * (x - P.1)

def l_A (x : ℝ) : ℝ := line A 1 x
def l_B (x : ℝ) : ℝ := x
def l_C (x : ℝ) : ℝ := line C (-1) x

def rotating_triangle_max_area (l_A l_B l_C : ℝ → ℝ) : ℝ := 116.5

theorem largest_area_of_rotating_triangle :
  rotating_triangle_max_area l_A l_B l_C = 116.5 :=
sorry

end largest_area_of_rotating_triangle_l151_151943


namespace compare_fractions_l151_151303

theorem compare_fractions {x : ℝ} (h : 3 < x ∧ x < 4) : 
  (2 / 3) > ((5 - x) / 3) :=
by sorry

end compare_fractions_l151_151303


namespace total_amount_l151_151462

noncomputable def mark_amount : ℝ := 5 / 8

noncomputable def carolyn_amount : ℝ := 7 / 20

theorem total_amount : mark_amount + carolyn_amount = 0.975 := by
  sorry

end total_amount_l151_151462


namespace mangoes_in_basket_B_l151_151444

theorem mangoes_in_basket_B :
  ∀ (A C D E B : ℕ), 
    (A = 15) →
    (C = 20) →
    (D = 25) →
    (E = 35) →
    (5 * 25 = A + C + D + E + B) →
    (B = 30) :=
by
  intros A C D E B hA hC hD hE hSum
  sorry

end mangoes_in_basket_B_l151_151444


namespace product_of_possible_x_values_l151_151731

theorem product_of_possible_x_values : 
  (∃ x1 x2 : ℚ, 
    (|15 / x1 + 4| = 3 ∧ |15 / x2 + 4| = 3) ∧
    -15 * -(15 / 7) = (225 / 7)) :=
sorry

end product_of_possible_x_values_l151_151731


namespace quadrilateral_circumscribed_circle_l151_151583

theorem quadrilateral_circumscribed_circle (a : ℝ) :
  ((a + 2) * x + (1 - a) * y - 3 = 0) ∧ ((a - 1) * x + (2 * a + 3) * y + 2 = 0) →
  ( a = 1 ∨ a = -1 ) :=
by
  intro h
  sorry

end quadrilateral_circumscribed_circle_l151_151583


namespace carl_wins_in_4950_configurations_l151_151534

noncomputable def num_distinct_configurations_at_Carl_win : ℕ :=
  sorry
  
theorem carl_wins_in_4950_configurations :
  num_distinct_configurations_at_Carl_win = 4950 :=
sorry

end carl_wins_in_4950_configurations_l151_151534


namespace a14_eq_33_l151_151201

variable {a : ℕ → ℝ}
variables (d : ℝ) (a1 : ℝ)

-- Defining the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℝ := a1 + n * d

-- Given conditions
axiom a5_eq_6 : arithmetic_sequence 4 = 6
axiom a8_eq_15 : arithmetic_sequence 7 = 15

-- Theorem statement
theorem a14_eq_33 : arithmetic_sequence 13 = 33 :=
by
  -- Proof skipped
  sorry

end a14_eq_33_l151_151201


namespace brain_can_always_open_door_l151_151530

noncomputable def can_open_door (a b c n m k : ℕ) : Prop :=
∃ x y z : ℕ, a^n = x^3 ∧ b^m = y^3 ∧ c^k = z^3

theorem brain_can_always_open_door :
  ∀ (a b c n m k : ℕ), 
  ∃ x y z : ℕ, a^n = x^3 ∧ b^m = y^3 ∧ c^k = z^3 :=
by sorry

end brain_can_always_open_door_l151_151530


namespace three_letter_words_with_A_at_least_once_l151_151991

theorem three_letter_words_with_A_at_least_once :
  let total_words := 4^3
  let words_without_A := 3^3
  total_words - words_without_A = 37 :=
by
  let total_words := 4^3
  let words_without_A := 3^3
  sorry

end three_letter_words_with_A_at_least_once_l151_151991


namespace area_of_enclosed_shape_l151_151982

noncomputable def enclosed_area : ℝ := 
∫ x in (0 : ℝ)..(2/3 : ℝ), (2 * x - 3 * x^2)

theorem area_of_enclosed_shape : enclosed_area = 4 / 27 := by
  sorry

end area_of_enclosed_shape_l151_151982


namespace battery_current_l151_151191

theorem battery_current (V R : ℝ) (R_val : R = 12) (hV : V = 48) (hI : I = 48 / R) : I = 4 :=
by
  sorry

end battery_current_l151_151191


namespace sum_of_legs_eq_40_l151_151734

theorem sum_of_legs_eq_40
  (x : ℝ)
  (h1 : x > 0)
  (h2 : x^2 + (x + 2)^2 = 29^2) :
  x + (x + 2) = 40 :=
by
  sorry

end sum_of_legs_eq_40_l151_151734


namespace min_value_when_a_is_half_range_of_a_for_positivity_l151_151941

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + 2*x + a) / x

theorem min_value_when_a_is_half : 
  ∀ x ∈ Set.Ici (1 : ℝ), f x (1/2) ≥ (7 / 2) := 
by 
  sorry

theorem range_of_a_for_positivity :
  ∀ x ∈ Set.Ici (1 : ℝ), f x a > 0 ↔ a ∈ Set.Ioc (-3 : ℝ) 1 :=
by 
  sorry

end min_value_when_a_is_half_range_of_a_for_positivity_l151_151941


namespace find_numbers_l151_151757

theorem find_numbers (x y : ℝ) (h₁ : x + y = x * y) (h₂ : x * y = x / y) :
  (x = 1 / 2) ∧ (y = -1) := by
  sorry

end find_numbers_l151_151757


namespace at_least_one_does_not_land_l151_151019

/-- Proposition stating "A lands within the designated area". -/
def p : Prop := sorry

/-- Proposition stating "B lands within the designated area". -/
def q : Prop := sorry

/-- Negation of proposition p, stating "A does not land within the designated area". -/
def not_p : Prop := ¬p

/-- Negation of proposition q, stating "B does not land within the designated area". -/
def not_q : Prop := ¬q

/-- The proposition "At least one trainee does not land within the designated area" can be expressed as (¬p) ∨ (¬q). -/
theorem at_least_one_does_not_land : (¬p ∨ ¬q) := sorry

end at_least_one_does_not_land_l151_151019


namespace gcd_b2_add_11b_add_28_b_add_6_eq_2_l151_151307

theorem gcd_b2_add_11b_add_28_b_add_6_eq_2 {b : ℤ} (h : ∃ k : ℤ, b = 1836 * k) : 
  Int.gcd (b^2 + 11 * b + 28) (b + 6) = 2 := 
by
  sorry

end gcd_b2_add_11b_add_28_b_add_6_eq_2_l151_151307


namespace isosceles_triangle_perimeter_l151_151749

theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 4) (h2 : b = 9) (h3 : c = 9) 
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) : a + b + c = 22 := 
by 
  sorry

end isosceles_triangle_perimeter_l151_151749


namespace scientific_notation_correct_l151_151894

/-- Define the number 42.39 million as 42.39 * 10^6 and prove that it is equivalent to 4.239 * 10^7 -/
def scientific_notation_of_42_39_million : Prop :=
  (42.39 * 10^6 = 4.239 * 10^7)

theorem scientific_notation_correct : scientific_notation_of_42_39_million :=
by 
  sorry

end scientific_notation_correct_l151_151894


namespace extra_people_needed_l151_151479

theorem extra_people_needed 
  (initial_people : ℕ) 
  (initial_time : ℕ) 
  (final_time : ℕ) 
  (work_done : ℕ) 
  (all_paint_same_rate : initial_people * initial_time = work_done) :
  initial_people = 8 →
  initial_time = 3 →
  final_time = 2 →
  work_done = 24 →
  ∃ extra_people : ℕ, extra_people = 4 :=
by
  sorry

end extra_people_needed_l151_151479


namespace counterexample_exists_l151_151100

theorem counterexample_exists : ∃ n : ℕ, n ≥ 2 ∧ ¬ ∃ k : ℕ, 2 ^ 2 ^ n % (2 ^ n - 1) = 4 ^ k := 
by
  sorry

end counterexample_exists_l151_151100


namespace books_jerry_added_l151_151071

def initial_action_figures : ℕ := 7
def initial_books : ℕ := 2

theorem books_jerry_added (B : ℕ) (h : initial_action_figures = initial_books + B + 1) : B = 4 :=
by
  sorry

end books_jerry_added_l151_151071


namespace goods_train_speed_l151_151296

theorem goods_train_speed (Vm : ℝ) (T : ℝ) (L : ℝ) (Vg : ℝ) :
  Vm = 50 → T = 9 → L = 280 →
  Vg = ((L / T) - (Vm * 1000 / 3600)) * 3600 / 1000 →
  Vg = 62 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end goods_train_speed_l151_151296


namespace smallest_possible_value_of_other_integer_l151_151610

theorem smallest_possible_value_of_other_integer (x : ℕ) (x_pos : 0 < x) (a b : ℕ) (h1 : a = 77) 
    (h2 : gcd a b = x + 7) (h3 : lcm a b = x * (x + 7)) : b = 22 :=
sorry

end smallest_possible_value_of_other_integer_l151_151610


namespace roots_sum_of_squares_l151_151106

theorem roots_sum_of_squares
  (p q r : ℝ)
  (h_roots : ∀ x, (3 * x^3 - 4 * x^2 + 3 * x + 7 = 0) → (x = p ∨ x = q ∨ x = r))
  (h_sum : p + q + r = 4 / 3)
  (h_prod_sum : p * q + q * r + r * p = 1)
  (h_prod : p * q * r = -7 / 3) :
  p^2 + q^2 + r^2 = -2 / 9 := 
sorry

end roots_sum_of_squares_l151_151106


namespace find_first_number_l151_151395

theorem find_first_number (x : ℕ) (h : x + 15 = 20) : x = 5 :=
by
  sorry

end find_first_number_l151_151395


namespace solve_inequality_l151_151033

theorem solve_inequality (a : ℝ) (x : ℝ) :
  (a = 0 → x > 1 → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (0 < a → a < 2 → 1 < x → x < 2 / a → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (a = 2 → False → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (a > 2 → 2 / a < x → x < 1 → (ax^2 - (a + 2) * x + 2 < 0)) ∧
  (a < 0 → ((x < 2 / a ∨ x > 1) → (ax^2 - (a + 2) * x + 2 < 0))) := sorry

end solve_inequality_l151_151033


namespace solve_system_l151_151649

theorem solve_system :
  ∃ x y : ℝ, (x + y = 5) ∧ (x + 2 * y = 8) ∧ (x = 2) ∧ (y = 3) :=
by
  sorry

end solve_system_l151_151649


namespace max_discount_rate_l151_151073

-- Define the constants used in the problem
def costPrice : ℝ := 4
def sellingPrice : ℝ := 5
def minProfitMarginRate : ℝ := 0.1
def minProfit : ℝ := costPrice * minProfitMarginRate -- which is 0.4

-- The maximum discount rate that preserves the required profit margin
theorem max_discount_rate : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 100) ∧ (sellingPrice * (1 - x / 100) - costPrice ≥ minProfit) ∧ (x = 12) :=
by
  sorry

end max_discount_rate_l151_151073


namespace tan_double_angle_cos_beta_l151_151276

theorem tan_double_angle (α β : ℝ) (h1 : Real.sin α = 4 * Real.sqrt 3 / 7) 
  (h2 : Real.cos (β - α) = 13 / 14) (h3 : 0 < β ∧ β < α ∧ α < Real.pi / 2) : 
  Real.tan (2 * α) = -(8 * Real.sqrt 3) / 47 :=
  sorry

theorem cos_beta (α β : ℝ) (h1 : Real.sin α = 4 * Real.sqrt 3 / 7) 
  (h2 : Real.cos (β - α) = 13 / 14) (h3 : 0 < β ∧ β < α ∧ α < Real.pi / 2) : 
  Real.cos β = 1 / 2 :=
  sorry

end tan_double_angle_cos_beta_l151_151276


namespace slips_with_3_l151_151585

-- Definitions of the conditions
def num_slips : ℕ := 15
def expected_value : ℚ := 5.4

-- Theorem statement
theorem slips_with_3 (y : ℕ) (t : ℕ := num_slips) (E : ℚ := expected_value) :
  E = (3 * y + 8 * (t - y)) / t → y = 8 :=
by
  sorry

end slips_with_3_l151_151585


namespace largest_divisor_same_remainder_l151_151743

theorem largest_divisor_same_remainder 
  (d : ℕ) (r : ℕ)
  (a b c : ℕ) 
  (h13511 : 13511 = a * d + r) 
  (h13903 : 13903 = b * d + r)
  (h14589 : 14589 = c * d + r) :
  d = 98 :=
by 
  sorry

end largest_divisor_same_remainder_l151_151743


namespace quadratic_real_roots_l151_151471

theorem quadratic_real_roots (a: ℝ) :
  ∀ x: ℝ, (a-6) * x^2 - 8 * x + 9 = 0 ↔ (a ≤ 70/9 ∧ a ≠ 6) :=
  sorry

end quadratic_real_roots_l151_151471


namespace symmetric_point_of_A_l151_151935

theorem symmetric_point_of_A (a b : ℝ) 
  (h1 : 2 * a - 4 * b + 9 = 0) 
  (h2 : ∃ t : ℝ, (a, b) = (1 - 4 * t, 4 + 2 * t)) : 
  (a, b) = (1, 4) :=
sorry

end symmetric_point_of_A_l151_151935


namespace exist_consecutive_days_20_games_l151_151572

theorem exist_consecutive_days_20_games 
  (a : ℕ → ℕ)
  (h_daily : ∀ n, a (n + 1) - a n ≥ 1)
  (h_weekly : ∀ n, a (n + 7) - a n ≤ 12) :
  ∃ i j, i < j ∧ a j - a i = 20 := by 
  sorry

end exist_consecutive_days_20_games_l151_151572


namespace distance_between_parallel_lines_l151_151261

-- Definition of the first line l1
def line1 (x y : ℝ) (c1 : ℝ) : Prop := 3 * x + 4 * y + c1 = 0

-- Definition of the second line l2
def line2 (x y : ℝ) (c2 : ℝ) : Prop := 6 * x + 8 * y + c2 = 0

-- The problem statement in Lean:
theorem distance_between_parallel_lines (c1 c2 : ℝ) :
  ∃ d : ℝ, d = |2 * c1 - c2| / 10 :=
sorry

end distance_between_parallel_lines_l151_151261


namespace arrangement_problem_l151_151762

def numWaysToArrangeParticipants : ℕ := 90

theorem arrangement_problem :
  ∃ (boys : ℕ) (girls : ℕ) (select_boys : ℕ → ℕ) (select_girls : ℕ → ℕ)
    (arrange : ℕ × ℕ × ℕ → ℕ),
  boys = 3 ∧ girls = 5 ∧
  select_boys boys = 3 ∧ select_girls girls = 5 ∧ 
  arrange (select_boys boys, select_girls girls, 2) = numWaysToArrangeParticipants :=
by
  sorry

end arrangement_problem_l151_151762


namespace daleyza_contracted_units_l151_151668

variable (units_building1 : ℕ)
variable (units_building2 : ℕ)
variable (units_building3 : ℕ)

def total_units (units_building1 units_building2 units_building3 : ℕ) : ℕ :=
  units_building1 + units_building2 + units_building3

theorem daleyza_contracted_units :
  units_building1 = 4000 →
  units_building2 = 2 * units_building1 / 5 →
  units_building3 = 120 * units_building2 / 100 →
  total_units units_building1 units_building2 units_building3 = 7520 :=
by
  intros h1 h2 h3
  unfold total_units
  rw [h1, h2, h3]
  sorry

end daleyza_contracted_units_l151_151668


namespace todd_runs_faster_l151_151924

-- Define the times taken by Brian and Todd
def brian_time : ℕ := 96
def todd_time : ℕ := 88

-- The theorem stating the problem
theorem todd_runs_faster : brian_time - todd_time = 8 :=
by
  -- Solution here
  sorry

end todd_runs_faster_l151_151924


namespace total_interest_obtained_l151_151577

-- Define the interest rates and face values
def interest_16 := 0.16 * 100
def interest_12 := 0.12 * 100
def interest_20 := 0.20 * 100

-- State the theorem to be proved
theorem total_interest_obtained : 
  interest_16 + interest_12 + interest_20 = 48 :=
by
  sorry

end total_interest_obtained_l151_151577


namespace quadratic_minimum_value_l151_151268

theorem quadratic_minimum_value (p q : ℝ) (h_min_value : ∀ x : ℝ, 2 * x^2 + p * x + q ≥ 10) :
  q = 10 + p^2 / 8 :=
by
  sorry

end quadratic_minimum_value_l151_151268


namespace discount_rate_on_pony_jeans_l151_151409

theorem discount_rate_on_pony_jeans (F P : ℝ) 
  (h1 : F + P = 25)
  (h2 : 45 * F + 36 * P = 900) :
  P = 25 :=
by
  sorry

end discount_rate_on_pony_jeans_l151_151409


namespace fraction_multiplication_simplifies_l151_151235

theorem fraction_multiplication_simplifies :
  (3 : ℚ) / 4 * (4 / 5) * (2 / 3) = 2 / 5 := 
by 
  -- Prove the equality step-by-step
  sorry

end fraction_multiplication_simplifies_l151_151235


namespace leos_current_weight_l151_151811

theorem leos_current_weight (L K : ℝ) 
  (h1 : L + 10 = 1.5 * K) 
  (h2 : L + K = 180) : L = 104 := 
by 
  sorry

end leos_current_weight_l151_151811


namespace order_of_four_l151_151886

theorem order_of_four {m n p q : ℝ} (hmn : m < n) (hpq : p < q) (h1 : (p - m) * (p - n) < 0) (h2 : (q - m) * (q - n) < 0) : m < p ∧ p < q ∧ q < n :=
by
  sorry

end order_of_four_l151_151886


namespace lcm_18_24_30_eq_360_l151_151750

-- Define the three numbers in the condition
def a : ℕ := 18
def b : ℕ := 24
def c : ℕ := 30

-- State the theorem to prove
theorem lcm_18_24_30_eq_360 : Nat.lcm a (Nat.lcm b c) = 360 :=
by 
  sorry -- Proof is omitted as per instructions

end lcm_18_24_30_eq_360_l151_151750


namespace lending_rate_l151_151150

noncomputable def principal: ℝ := 5000
noncomputable def rate_borrowed: ℝ := 4
noncomputable def time_years: ℝ := 2
noncomputable def gain_per_year: ℝ := 100

theorem lending_rate :
  ∃ (rate_lent: ℝ), 
  (principal * rate_lent * time_years / 100) - (principal * rate_borrowed * time_years / 100) / time_years = gain_per_year ∧
  rate_lent = 6 :=
by
  sorry

end lending_rate_l151_151150


namespace min_elements_l151_151386

-- Definitions for conditions in part b
def num_elements (n : ℕ) : ℕ := 2 * n + 1
def sum_upper_bound (n : ℕ) : ℕ := 15 * n + 2
def sum_arithmetic_mean (n : ℕ) : ℕ := 14 * n + 7

-- Prove that for conditions, the number of elements should be at least 11
theorem min_elements (n : ℕ) (h : 14 * n + 7 ≤ 15 * n + 2) : 2 * n + 1 ≥ 11 :=
by {
  sorry
}

end min_elements_l151_151386


namespace remaining_slices_after_weekend_l151_151708

theorem remaining_slices_after_weekend 
  (initial_pies : ℕ) (slices_per_pie : ℕ) (rebecca_initial_slices : ℕ) 
  (family_fraction : ℚ) (sunday_evening_slices : ℕ) : 
  initial_pies = 2 → 
  slices_per_pie = 8 → 
  rebecca_initial_slices = 2 → 
  family_fraction = 0.5 → 
  sunday_evening_slices = 2 → 
  (initial_pies * slices_per_pie 
   - rebecca_initial_slices 
   - family_fraction * (initial_pies * slices_per_pie - rebecca_initial_slices) 
   - sunday_evening_slices) = 5 :=
by 
  intros initial_pies_eq slices_per_pie_eq rebecca_initial_slices_eq family_fraction_eq sunday_evening_slices_eq
  sorry

end remaining_slices_after_weekend_l151_151708


namespace unique_natural_in_sequences_l151_151652

def seq_x (n : ℕ) : ℤ := if n = 0 then 10 else if n = 1 then 10 else seq_x (n - 2) * (seq_x (n - 1) + 1) + 1
def seq_y (n : ℕ) : ℤ := if n = 0 then -10 else if n = 1 then -10 else (seq_y (n - 1) + 1) * seq_y (n - 2) + 1

theorem unique_natural_in_sequences (k : ℕ) (i j : ℕ) :
  seq_x i = k → seq_y j ≠ k :=
by
  sorry

end unique_natural_in_sequences_l151_151652


namespace crayons_ratio_l151_151864

theorem crayons_ratio (K B G J : ℕ) 
  (h1 : K = 2 * B)
  (h2 : B = 2 * G)
  (h3 : G = J)
  (h4 : K = 128)
  (h5 : J = 8) : 
  G / J = 4 :=
by
  sorry

end crayons_ratio_l151_151864


namespace tetrahedron_edge_length_of_tangent_spheres_l151_151629

theorem tetrahedron_edge_length_of_tangent_spheres (r : ℝ) (h₁ : r = 2) :
  ∃ s : ℝ, s = 4 :=
by
  sorry

end tetrahedron_edge_length_of_tangent_spheres_l151_151629


namespace solution_set_inequality_l151_151363

variable (f : ℝ → ℝ)

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def derivative (f : ℝ → ℝ) (f' : ℝ → ℝ) : Prop :=
  ∀ x, HasDerivAt f (f' x) x

def condition_x_f_prime (f f' : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x^2 * f' x > 2 * x * f (-x)

-- Main theorem to prove the solution set of inequality
theorem solution_set_inequality (f' : ℝ → ℝ) :
  is_odd_function f →
  derivative f f' →
  condition_x_f_prime f f' →
  ∀ x : ℝ, x^2 * f x < (3 * x - 1)^2 * f (1 - 3 * x) → x < (1 / 4) := 
  by
    intros h_odd h_deriv h_cond x h_ineq
    sorry

end solution_set_inequality_l151_151363


namespace range_of_a_l151_151322

open Real

theorem range_of_a (a : ℝ) :
  (∀ x > 0, ae^x + x + x * log x ≥ x^2) → a ≥ 1 / exp 2 :=
sorry

end range_of_a_l151_151322


namespace problem_correct_choice_l151_151489

-- Definitions of the propositions
def p : Prop := ∃ n : ℕ, 3 = 2 * n + 1
def q : Prop := ∃ n : ℕ, 5 = 2 * n

-- The problem statement
theorem problem_correct_choice : p ∨ q :=
sorry

end problem_correct_choice_l151_151489


namespace number_of_ways_to_choose_4_captains_from_15_l151_151986

def choose_captains (n r : ℕ) : ℕ :=
  Nat.choose n r

theorem number_of_ways_to_choose_4_captains_from_15 :
  choose_captains 15 4 = 1365 := by
  sorry

end number_of_ways_to_choose_4_captains_from_15_l151_151986


namespace no_solution_exists_l151_151882

theorem no_solution_exists (x y : ℝ) : 9^(y + 1) / (1 + 4 / x^2) ≠ 1 :=
by
  sorry

end no_solution_exists_l151_151882


namespace ratio_perimeters_of_squares_l151_151415

theorem ratio_perimeters_of_squares 
  (s₁ s₂ : ℝ)
  (h : (s₁ ^ 2) / (s₂ ^ 2) = 25 / 36) :
  (4 * s₁) / (4 * s₂) = 5 / 6 :=
by
  sorry

end ratio_perimeters_of_squares_l151_151415


namespace min_value_x1x2_squared_inequality_ab_l151_151671

def D : Set (ℝ × ℝ) := 
  { p | ∃ x1 x2, p = (x1, x2) ∧ x1 + x2 = 2 ∧ x1 > 0 ∧ x2 > 0 }

-- Part 1: Proving the minimum value of x1^2 + x2^2 in set D is 2
theorem min_value_x1x2_squared (x1 x2 : ℝ) (h : (x1, x2) ∈ D) : 
  x1^2 + x2^2 ≥ 2 := 
sorry

-- Part 2: Proving the inequality for any (a, b) in set D
theorem inequality_ab (a b : ℝ) (h : (a, b) ∈ D) : 
  (1 / (a + 2 * b) + 1 / (2 * a + b)) ≥ (2 / 3) := 
sorry

end min_value_x1x2_squared_inequality_ab_l151_151671


namespace values_of_x_l151_151295

theorem values_of_x (x : ℕ) (h : Nat.choose 18 x = Nat.choose 18 (3 * x - 6)) : x = 3 ∨ x = 6 :=
by
  sorry

end values_of_x_l151_151295


namespace largest_in_eight_consecutive_integers_l151_151301

theorem largest_in_eight_consecutive_integers (n : ℕ) (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) + (n + 7) = 4304) :
  n + 7 = 544 :=
by
  sorry

end largest_in_eight_consecutive_integers_l151_151301


namespace inequality_equivalence_l151_151212

theorem inequality_equivalence (a : ℝ) :
  (∀ (x : ℝ), |x + 1| + |x - 1| ≥ a) ↔ (a ≤ 2) :=
sorry

end inequality_equivalence_l151_151212


namespace product_of_remaining_numbers_is_12_l151_151162

noncomputable def final_numbers_product : ℕ := 
  12

theorem product_of_remaining_numbers_is_12 :
  ∀ (initial_ones initial_twos initial_threes initial_fours : ℕ)
  (erase_add_op : Π (a b c : ℕ), Prop),
  initial_ones = 11 ∧ initial_twos = 22 ∧ initial_threes = 33 ∧ initial_fours = 44 ∧
  (∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c → erase_add_op a b c) →
  (∃ (final1 final2 final3 : ℕ), erase_add_op 11 22 33 → final1 * final2 * final3 = final_numbers_product) :=
sorry

end product_of_remaining_numbers_is_12_l151_151162


namespace largest_common_divisor_l151_151581

theorem largest_common_divisor (a b : ℕ) (h1 : a = 360) (h2 : b = 315) : 
  ∃ d : ℕ, d ∣ a ∧ d ∣ b ∧ ∀ e : ℕ, (e ∣ a ∧ e ∣ b) → e ≤ d ∧ d = 45 :=
by
  sorry

end largest_common_divisor_l151_151581


namespace maximum_time_for_3_digit_combination_lock_l151_151421

def max_time_to_open_briefcase : ℕ :=
  let num_combinations := 9 * 9 * 9
  let time_per_trial := 3
  num_combinations * time_per_trial

theorem maximum_time_for_3_digit_combination_lock :
  max_time_to_open_briefcase = 2187 :=
by
  sorry

end maximum_time_for_3_digit_combination_lock_l151_151421


namespace truck_travel_l151_151336

/-- If a truck travels 150 miles using 5 gallons of diesel, then it will travel 210 miles using 7 gallons of diesel. -/
theorem truck_travel (d1 d2 g1 g2 : ℕ) (h1 : d1 = 150) (h2 : g1 = 5) (h3 : g2 = 7) (h4 : d2 = d1 * g2 / g1) : d2 = 210 := by
  sorry

end truck_travel_l151_151336


namespace initial_tax_rate_l151_151979

variable (R : ℝ)

theorem initial_tax_rate
  (income : ℝ := 48000)
  (new_rate : ℝ := 0.30)
  (savings : ℝ := 7200)
  (tax_savings : income * (R / 100) - income * new_rate = savings) :
  R = 45 := by
  sorry

end initial_tax_rate_l151_151979


namespace axis_of_symmetry_parabola_l151_151084

/-- If a parabola passes through points A(-2,0) and B(4,0), then the axis of symmetry of the parabola is the line x = 1. -/
theorem axis_of_symmetry_parabola (x : ℝ → ℝ) (hA : x (-2) = 0) (hB : x 4 = 0) : 
  ∃ c : ℝ, c = 1 ∧ ∀ y : ℝ, x y = x (2 * c - y) :=
sorry

end axis_of_symmetry_parabola_l151_151084


namespace union_of_sets_l151_151265

def M : Set ℕ := {2, 3, 5}
def N : Set ℕ := {3, 4, 5}

theorem union_of_sets : M ∪ N = {2, 3, 4, 5} := by
  sorry

end union_of_sets_l151_151265


namespace abs_ab_eq_2128_l151_151347

theorem abs_ab_eq_2128 (a b : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : ∃ r s : ℤ, r ≠ s ∧ ∃ r' : ℤ, r' = r ∧ 
          (x^3 + a * x^2 + b * x + 16 * a = (x - r)^2 * (x - s) ∧ r * r * s = -16 * a)) :
  |a * b| = 2128 :=
sorry

end abs_ab_eq_2128_l151_151347


namespace eq_1_solution_eq_2_solution_eq_3_solution_eq_4_solution_l151_151163

-- Equation (1): 2x^2 + 2x - 1 = 0
theorem eq_1_solution (x : ℝ) :
  2 * x^2 + 2 * x - 1 = 0 ↔ (x = (-1 + Real.sqrt 3) / 2 ∨ x = (-1 - Real.sqrt 3) / 2) := by
  sorry

-- Equation (2): x(x-1) = 2(x-1)
theorem eq_2_solution (x : ℝ) :
  x * (x - 1) = 2 * (x - 1) ↔ (x = 1 ∨ x = 2) := by
  sorry

-- Equation (3): 4(x-2)^2 = 9(2x+1)^2
theorem eq_3_solution (x : ℝ) :
  4 * (x - 2)^2 = 9 * (2 * x + 1)^2 ↔ (x = -7 / 4 ∨ x = 1 / 8) := by
  sorry

-- Equation (4): (2x-1)^2 - 3(2x-1) = 4
theorem eq_4_solution (x : ℝ) :
  (2 * x - 1)^2 - 3 * (2 * x - 1) = 4 ↔ (x = 5 / 2 ∨ x = 0) := by
  sorry

end eq_1_solution_eq_2_solution_eq_3_solution_eq_4_solution_l151_151163
