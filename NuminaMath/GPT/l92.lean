import Mathlib

namespace pyramid_surface_area_l92_9223

theorem pyramid_surface_area
  (base_side_length : ℝ)
  (peak_height : ℝ)
  (base_area : ℝ)
  (slant_height : ℝ)
  (triangular_face_area : ℝ)
  (total_surface_area : ℝ)
  (h1 : base_side_length = 10)
  (h2 : peak_height = 12)
  (h3 : base_area = base_side_length ^ 2)
  (h4 : slant_height = Real.sqrt (peak_height ^ 2 + (base_side_length / 2) ^ 2))
  (h5 : triangular_face_area = 0.5 * base_side_length * slant_height)
  (h6 : total_surface_area = base_area + 4 * triangular_face_area)
  : total_surface_area = 360 := 
sorry

end pyramid_surface_area_l92_9223


namespace doughnuts_per_person_l92_9288

theorem doughnuts_per_person :
  let samuel_doughnuts := 24
  let cathy_doughnuts := 36
  let total_doughnuts := samuel_doughnuts + cathy_doughnuts
  let total_people := 10
  total_doughnuts / total_people = 6 := 
by
  -- Definitions and conditions from the problem
  let samuel_doughnuts := 24
  let cathy_doughnuts := 36
  let total_doughnuts := samuel_doughnuts + cathy_doughnuts
  let total_people := 10
  -- Goal to prove
  show total_doughnuts / total_people = 6
  sorry

end doughnuts_per_person_l92_9288


namespace partial_fraction_decomposition_l92_9247

noncomputable def partial_fraction_product (A B C : ℤ) : ℤ :=
  A * B * C

theorem partial_fraction_decomposition:
  ∃ A B C : ℤ, 
  (∀ x : ℤ, (x^2 - 19 = A * (x + 2) * (x - 3) 
                    + B * (x - 1) * (x - 3) 
                    + C * (x - 1) * (x + 2) )) 
  → partial_fraction_product A B C = 3 :=
by
  sorry

end partial_fraction_decomposition_l92_9247


namespace quadratic_has_distinct_real_roots_l92_9236

theorem quadratic_has_distinct_real_roots :
  let a := 5
  let b := 14
  let c := 5
  let discriminant := b^2 - 4 * a * c
  discriminant > 0 := 
by
  sorry

end quadratic_has_distinct_real_roots_l92_9236


namespace total_amount_is_correct_l92_9275

variable (w x y z R : ℝ)
variable (hx : x = 0.345 * w)
variable (hy : y = 0.45625 * w)
variable (hz : z = 0.61875 * w)
variable (hy_value : y = 112.50)

theorem total_amount_is_correct :
  R = w + x + y + z → R = 596.8150684931507 := by
  sorry

end total_amount_is_correct_l92_9275


namespace average_of_b_and_c_l92_9215

theorem average_of_b_and_c (a b c : ℝ) 
  (h₁ : (a + b) / 2 = 50) 
  (h₂ : c - a = 40) : 
  (b + c) / 2 = 70 := 
by
  sorry

end average_of_b_and_c_l92_9215


namespace part1_solution_set_eq_part2_a_range_l92_9265

theorem part1_solution_set_eq : {x : ℝ | |2 * x + 1| + |2 * x - 3| ≤ 6} = Set.Icc (-1) 2 :=
by sorry

theorem part2_a_range (a : ℝ) (h : a > 0) : 
  (∃ x : ℝ, |2 * x + 1| + |2 * x - 3| < |a - 2|) → 6 < a :=
by sorry

end part1_solution_set_eq_part2_a_range_l92_9265


namespace quadratic_form_completion_l92_9259

theorem quadratic_form_completion (b c : ℤ)
  (h : ∀ x:ℂ, x^2 + 520*x + 600 = (x+b)^2 + c) :
  c / b = -258 :=
by sorry

end quadratic_form_completion_l92_9259


namespace sum_of_fractions_l92_9294

theorem sum_of_fractions :
  (3 / 12 : Real) + (6 / 120) + (9 / 1200) = 0.3075 :=
by
  sorry

end sum_of_fractions_l92_9294


namespace simplify_and_evaluate_l92_9298

theorem simplify_and_evaluate (a : ℝ) (h : a^2 + 2 * a - 1 = 0) :
  ((a - 2) / (a^2 + 2 * a) - (a - 1) / (a^2 + 4 * a + 4)) / ((a - 4) / (a + 2)) = 1 / 3 :=
by sorry

end simplify_and_evaluate_l92_9298


namespace dress_design_count_l92_9271

-- Definitions of the given conditions
def number_of_colors : Nat := 4
def number_of_patterns : Nat := 5

-- Statement to prove the total number of unique dress designs
theorem dress_design_count :
  number_of_colors * number_of_patterns = 20 := by
  sorry

end dress_design_count_l92_9271


namespace units_digit_of_6_pow_5_l92_9254

theorem units_digit_of_6_pow_5 : (6^5 % 10) = 6 := 
by sorry

end units_digit_of_6_pow_5_l92_9254


namespace cube_union_volume_is_correct_cube_union_surface_area_is_correct_l92_9219

noncomputable def cubeUnionVolume : ℝ :=
  let cubeVolume := 1
  let intersectionVolume := 1 / 4
  cubeVolume * 2 - intersectionVolume

theorem cube_union_volume_is_correct :
  cubeUnionVolume = 5 / 4 := sorry

noncomputable def cubeUnionSurfaceArea : ℝ :=
  2 * (6 * (1 / 4) + 6 * (1 / 4 / 4))

theorem cube_union_surface_area_is_correct :
  cubeUnionSurfaceArea = 15 / 2 := sorry

end cube_union_volume_is_correct_cube_union_surface_area_is_correct_l92_9219


namespace symmetric_angles_l92_9243

theorem symmetric_angles (α β : ℝ) (k : ℤ) (h : α + β = 2 * k * Real.pi) : α = 2 * k * Real.pi - β :=
by
  sorry

end symmetric_angles_l92_9243


namespace cost_of_tax_free_items_l92_9225

/-- 
Daniel went to a shop and bought items worth Rs 25, including a 30 paise sales tax on taxable items
with a tax rate of 10%. Prove that the cost of tax-free items is Rs 22.
-/
theorem cost_of_tax_free_items (total_spent taxable_amount sales_tax rate : ℝ)
  (h1 : total_spent = 25)
  (h2 : sales_tax = 0.3)
  (h3 : rate = 0.1)
  (h4 : taxable_amount = sales_tax / rate) :
  (total_spent - taxable_amount = 22) :=
by
  sorry

end cost_of_tax_free_items_l92_9225


namespace capsule_depth_equation_l92_9260

theorem capsule_depth_equation (x y z : ℝ) (h : y = 4 * x + z) : y = 4 * x + z := 
by 
  exact h

end capsule_depth_equation_l92_9260


namespace student_D_most_stable_l92_9249

-- Define the variances for students A, B, C, and D
def SA_squared : ℝ := 2.1
def SB_squared : ℝ := 3.5
def SC_squared : ℝ := 9
def SD_squared : ℝ := 0.7

-- Theorem stating that student D has the most stable performance
theorem student_D_most_stable :
  SD_squared < SA_squared ∧ SD_squared < SB_squared ∧ SD_squared < SC_squared := by
  sorry

end student_D_most_stable_l92_9249


namespace james_problem_l92_9267

def probability_at_least_two_green_apples (total: ℕ) (red: ℕ) (green: ℕ) (yellow: ℕ) (choices: ℕ) : ℚ :=
  let favorable_outcomes := (Nat.choose green 2) * (Nat.choose (total - green) 1) + (Nat.choose green 3)
  let total_outcomes := Nat.choose total choices
  favorable_outcomes / total_outcomes

theorem james_problem : probability_at_least_two_green_apples 10 5 3 2 3 = 11 / 60 :=
by sorry

end james_problem_l92_9267


namespace find_n_l92_9227

theorem find_n (n : ℕ) (hnpos : 0 < n)
  (hsquare : ∃ k : ℕ, k^2 = n^4 + 2*n^3 + 5*n^2 + 12*n + 5) :
  n = 1 ∨ n = 2 := 
sorry

end find_n_l92_9227


namespace brass_selling_price_l92_9263

noncomputable def copper_price : ℝ := 0.65
noncomputable def zinc_price : ℝ := 0.30
noncomputable def total_weight_brass : ℝ := 70
noncomputable def weight_copper : ℝ := 30
noncomputable def weight_zinc := total_weight_brass - weight_copper
noncomputable def cost_copper := weight_copper * copper_price
noncomputable def cost_zinc := weight_zinc * zinc_price
noncomputable def total_cost := cost_copper + cost_zinc
noncomputable def selling_price_per_pound := total_cost / total_weight_brass

theorem brass_selling_price :
  selling_price_per_pound = 0.45 :=
by
  sorry

end brass_selling_price_l92_9263


namespace preferred_pets_combination_l92_9285

-- Define the number of puppies, kittens, and hamsters
def num_puppies : ℕ := 20
def num_kittens : ℕ := 10
def num_hamsters : ℕ := 12

-- State the main theorem to prove, that the number of ways Alice, Bob, and Charlie 
-- can buy their preferred pets is 2400
theorem preferred_pets_combination : num_puppies * num_kittens * num_hamsters = 2400 :=
by
  sorry

end preferred_pets_combination_l92_9285


namespace min_sticks_cover_200cm_l92_9224

def length_covered (n6 n7 : ℕ) : ℕ :=
  6 * n6 + 7 * n7

theorem min_sticks_cover_200cm :
  ∃ (n6 n7 : ℕ), length_covered n6 n7 = 200 ∧ (∀ (m6 m7 : ℕ), (length_covered m6 m7 = 200 → m6 + m7 ≥ n6 + n7)) ∧ (n6 + n7 = 29) :=
sorry

end min_sticks_cover_200cm_l92_9224


namespace log_inequalities_l92_9255

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := Real.log (Real.log 2 / Real.log 3) / Real.log 2

theorem log_inequalities : c < b ∧ b < a :=
  sorry

end log_inequalities_l92_9255


namespace find_n_l92_9214

noncomputable def n (n : ℕ) : Prop :=
  lcm n 12 = 42 ∧ gcd n 12 = 6

theorem find_n (n : ℕ) (h : lcm n 12 = 42) (h1 : gcd n 12 = 6) : n = 21 :=
by sorry

end find_n_l92_9214


namespace fill_cistern_time_l92_9296

theorem fill_cistern_time (fill_ratio : ℚ) (time_for_fill_ratio : ℚ) :
  fill_ratio = 1/11 ∧ time_for_fill_ratio = 4 → (11 * time_for_fill_ratio) = 44 :=
by
  sorry

end fill_cistern_time_l92_9296


namespace polynomial_factorization_l92_9201

-- Definitions used in the conditions
def given_polynomial (a b c : ℝ) : ℝ :=
  a^3 * (b^2 - c^2) + b^3 * (c^2 - a^2) + c^3 * (a^2 - b^2)

def p (a b c : ℝ) : ℝ := -(a * b + a * c + b * c)

-- The Lean 4 statement to be proved
theorem polynomial_factorization (a b c : ℝ) :
  given_polynomial a b c = (a - b) * (b - c) * (c - a) * p a b c :=
by
  sorry

end polynomial_factorization_l92_9201


namespace required_tents_l92_9276

def numberOfPeopleInMattFamily : ℕ := 1 + 2
def numberOfPeopleInBrotherFamily : ℕ := 1 + 1 + 4
def numberOfPeopleInUncleJoeFamily : ℕ := 1 + 1 + 3
def totalNumberOfPeople : ℕ := numberOfPeopleInMattFamily + numberOfPeopleInBrotherFamily + numberOfPeopleInUncleJoeFamily
def numberOfPeopleSleepingInHouse : ℕ := 4
def numberOfPeopleSleepingInTents : ℕ := totalNumberOfPeople - numberOfPeopleSleepingInHouse
def peoplePerTent : ℕ := 2

def numberOfTentsNeeded : ℕ :=
  numberOfPeopleSleepingInTents / peoplePerTent

theorem required_tents : numberOfTentsNeeded = 5 := by
  sorry

end required_tents_l92_9276


namespace negation_of_statement_l92_9218

theorem negation_of_statement :
  ¬(∀ x : ℝ, ∃ n : ℕ, 0 < n ∧ n > x^2) ↔ (∃ x : ℝ, ∀ n : ℕ, 0 < n → n < x^2) := by
sorry

end negation_of_statement_l92_9218


namespace heejin_most_balls_is_volleyballs_l92_9250

def heejin_basketballs : ℕ := 3
def heejin_volleyballs : ℕ := 5
def heejin_baseballs : ℕ := 1

theorem heejin_most_balls_is_volleyballs :
  heejin_volleyballs > heejin_basketballs ∧ heejin_volleyballs > heejin_baseballs :=
by
  sorry

end heejin_most_balls_is_volleyballs_l92_9250


namespace smallest_a_condition_l92_9210

theorem smallest_a_condition:
  ∃ a: ℝ, (∀ x y z: ℝ, (0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x + y + z = 1) → a * (x^2 + y^2 + z^2) + x * y * z ≥ 10 / 27) ∧ a = 2 / 9 :=
sorry

end smallest_a_condition_l92_9210


namespace range_of_x_plus_y_l92_9211

open Real

theorem range_of_x_plus_y (x y : ℝ) (h : x - sqrt (x + 1) = sqrt (y + 1) - y) :
  -sqrt 5 + 1 ≤ x + y ∧ x + y ≤ sqrt 5 + 1 :=
by sorry

end range_of_x_plus_y_l92_9211


namespace perpendicular_lines_l92_9234

def line1 (a : ℝ) (x y : ℝ) := a * x + 2 * y + 6 = 0
def line2 (a : ℝ) (x y : ℝ) := x + (a - 1) * y + a^2 - 1 = 0

theorem perpendicular_lines (a : ℝ) : 
  (∀ x y : ℝ, line1 a x y) ∧ (∀ x y : ℝ, line2 a x y) ∧ 
  (∀ x1 y1 x2 y2 : ℝ, 
    (line1 a x1 y1) ∧ (line2 a x2 y2) → 
    (-a / 2) * (-1 / (a - 1)) = -1) → a = 2 / 3 :=
sorry

end perpendicular_lines_l92_9234


namespace exceeds_threshold_at_8_l92_9277

def geometric_sum (a r n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

def exceeds_threshold (n : ℕ) : Prop :=
  geometric_sum 2 2 n ≥ 500

theorem exceeds_threshold_at_8 :
  ∀ n < 8, ¬exceeds_threshold n ∧ exceeds_threshold 8 :=
by
  sorry

end exceeds_threshold_at_8_l92_9277


namespace jana_winning_strategy_l92_9232

theorem jana_winning_strategy (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  (m + n) % 2 = 1 ∨ m = 1 ∨ n = 1 := sorry

end jana_winning_strategy_l92_9232


namespace amusement_park_weekly_revenue_l92_9251

def ticket_price : ℕ := 3
def visitors_mon_to_fri_per_day : ℕ := 100
def visitors_saturday : ℕ := 200
def visitors_sunday : ℕ := 300

theorem amusement_park_weekly_revenue : 
  let total_visitors_weekdays := visitors_mon_to_fri_per_day * 5
  let total_visitors_weekend := visitors_saturday + visitors_sunday
  let total_visitors := total_visitors_weekdays + total_visitors_weekend
  let total_revenue := total_visitors * ticket_price
  total_revenue = 3000 := by
  sorry

end amusement_park_weekly_revenue_l92_9251


namespace polynomial_three_positive_roots_inequality_polynomial_three_positive_roots_equality_condition_l92_9292

theorem polynomial_three_positive_roots_inequality
  (a b c : ℝ)
  (x1 x2 x3 : ℝ)
  (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0)
  (h_poly : ∀ x, x^3 + a * x^2 + b * x + c = 0) :
  2 * a^3 + 9 * c ≤ 7 * a * b :=
sorry

theorem polynomial_three_positive_roots_equality_condition
  (a b c : ℝ)
  (x1 x2 x3 : ℝ)
  (h1 : x1 > 0) (h2 : x2 > 0) (h3 : x3 > 0)
  (h_poly : ∀ x, x^3 + a * x^2 + b * x + c = 0) :
  (2 * a^3 + 9 * c = 7 * a * b) ↔ (x1 = x2 ∧ x2 = x3) :=
sorry

end polynomial_three_positive_roots_inequality_polynomial_three_positive_roots_equality_condition_l92_9292


namespace solve_real_solution_l92_9231

theorem solve_real_solution:
  ∀ x : ℝ, (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 8) ↔
           (x = 4 + Real.sqrt 57) ∨ (x = 4 - Real.sqrt 57) :=
by
  sorry

end solve_real_solution_l92_9231


namespace cos_double_angle_l92_9204

theorem cos_double_angle (α : ℝ) (h : Real.sin (π + α) = 2 / 3) : Real.cos (2 * α) = 1 / 9 := 
sorry

end cos_double_angle_l92_9204


namespace mass_percentage_Al_in_mixture_l92_9270

/-- Define molar masses for the respective compounds -/
def molar_mass_AlCl3 : ℝ := 133.33
def molar_mass_Al2SO4_3 : ℝ := 342.17
def molar_mass_AlOH3 : ℝ := 78.01

/-- Define masses of respective compounds given in grams -/
def mass_AlCl3 : ℝ := 50
def mass_Al2SO4_3 : ℝ := 70
def mass_AlOH3 : ℝ := 40

/-- Define molar mass of Al -/
def molar_mass_Al : ℝ := 26.98

theorem mass_percentage_Al_in_mixture :
  (mass_AlCl3 / molar_mass_AlCl3 * molar_mass_Al +
   mass_Al2SO4_3 / molar_mass_Al2SO4_3 * (2 * molar_mass_Al) +
   mass_AlOH3 / molar_mass_AlOH3 * molar_mass_Al) / 
  (mass_AlCl3 + mass_Al2SO4_3 + mass_AlOH3) * 100 
  = 21.87 := by
  sorry

end mass_percentage_Al_in_mixture_l92_9270


namespace smallest_x_integer_value_l92_9206

theorem smallest_x_integer_value (x : ℤ) (h : (x - 5) ∣ 58) : x = -53 :=
by
  sorry

end smallest_x_integer_value_l92_9206


namespace triangle_area_correct_l92_9248

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem triangle_area_correct : 
  area_of_triangle (0, 0) (2, 0) (2, 3) = 3 :=
by
  sorry

end triangle_area_correct_l92_9248


namespace inscribed_circle_radius_l92_9272

theorem inscribed_circle_radius (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 30) : 
  ∃ r : ℝ, r = (105 * Real.sqrt 274) / 274 := 
by 
  sorry

end inscribed_circle_radius_l92_9272


namespace model_y_completion_time_l92_9282

theorem model_y_completion_time
  (rate_model_x : ℕ → ℝ)
  (rate_model_y : ℕ → ℝ)
  (num_model_x : ℕ)
  (num_model_y : ℕ)
  (time_model_x : ℝ)
  (combined_rate : ℝ)
  (same_number : num_model_y = num_model_x)
  (task_completion_x : ∀ x, rate_model_x x = 1 / time_model_x)
  (total_model_x : num_model_x = 24)
  (task_completion_y : ∀ y, rate_model_y y = 1 / y)
  (one_minute_completion : num_model_x * rate_model_x 1 + num_model_y * rate_model_y 36 = combined_rate)
  : 36 = time_model_x * 2 :=
by
  sorry

end model_y_completion_time_l92_9282


namespace age_of_James_when_Thomas_reaches_current_age_l92_9240
    
theorem age_of_James_when_Thomas_reaches_current_age
  (T S J : ℕ)
  (h1 : T = 6)
  (h2 : S = T + 13)
  (h3 : S = J - 5) :
  J + (S - T) = 37 := 
by
  sorry

end age_of_James_when_Thomas_reaches_current_age_l92_9240


namespace largest_digit_divisible_by_6_l92_9273

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

theorem largest_digit_divisible_by_6 : ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 9 ∧ is_even N ∧ is_divisible_by_3 (26 + N) ∧ 
  (∀ (N' : ℕ), 0 ≤ N' ∧ N' ≤ 9 ∧ is_even N' ∧ is_divisible_by_3 (26 + N') → N' ≤ N) :=
sorry

end largest_digit_divisible_by_6_l92_9273


namespace root_calculation_l92_9202

theorem root_calculation :
  (Real.sqrt (Real.sqrt (0.00032 ^ (1 / 5)) ^ (1 / 4))) = 0.6687 :=
by
  sorry

end root_calculation_l92_9202


namespace math_problem_l92_9229

noncomputable def proof_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  a^2 + 4 * b^2 + 1 / (a * b) ≥ 4

theorem math_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : proof_problem a b ha hb :=
by
  sorry

end math_problem_l92_9229


namespace usual_time_72_l92_9222

namespace TypicalTimeProof

variables (S T : ℝ) 

theorem usual_time_72 (h : T ≠ 0) (h2 : 0.75 * S ≠ 0) (h3 : 4 * T = 3 * (T + 24)) : T = 72 := by
  sorry

end TypicalTimeProof

end usual_time_72_l92_9222


namespace kennedy_distance_to_school_l92_9252

def miles_per_gallon : ℕ := 19
def initial_gallons : ℕ := 2
def distance_softball_park : ℕ := 6
def distance_burger_restaurant : ℕ := 2
def distance_friends_house : ℕ := 4
def distance_home : ℕ := 11

def total_distance_possible : ℕ := miles_per_gallon * initial_gallons
def distance_after_school : ℕ := distance_softball_park + distance_burger_restaurant + distance_friends_house + distance_home
def distance_to_school : ℕ := total_distance_possible - distance_after_school

theorem kennedy_distance_to_school :
  distance_to_school = 15 :=
by
  sorry

end kennedy_distance_to_school_l92_9252


namespace find_ax5_by5_l92_9299

variables (a b x y: ℝ)

theorem find_ax5_by5 (h1 : a * x + b * y = 5)
                      (h2 : a * x^2 + b * y^2 = 11)
                      (h3 : a * x^3 + b * y^3 = 24)
                      (h4 : a * x^4 + b * y^4 = 56) :
                      a * x^5 + b * y^5 = 180.36 :=
sorry

end find_ax5_by5_l92_9299


namespace alyssa_puppies_left_l92_9246

def initial_puppies : Nat := 7
def puppies_per_puppy : Nat := 4
def given_away : Nat := 15

theorem alyssa_puppies_left :
  (initial_puppies + initial_puppies * puppies_per_puppy) - given_away = 20 := 
  by
    sorry

end alyssa_puppies_left_l92_9246


namespace gcd_324_243_135_l92_9266

theorem gcd_324_243_135 : Nat.gcd (Nat.gcd 324 243) 135 = 27 := by
  sorry

end gcd_324_243_135_l92_9266


namespace reciprocal_of_neg_two_l92_9237

theorem reciprocal_of_neg_two : ∀ x : ℝ, x = -2 → (1 / x) = -1 / 2 :=
by
  intro x h
  rw [h]
  norm_num

end reciprocal_of_neg_two_l92_9237


namespace find_x_intercept_of_perpendicular_line_l92_9238

noncomputable def line_y_intercept : ℝ × ℝ := (0, 3)
noncomputable def given_line (x y : ℝ) : Prop := 2 * x + y = 3
noncomputable def x_intercept_of_perpendicular_line : ℝ × ℝ := (-6, 0)

theorem find_x_intercept_of_perpendicular_line :
  (∀ (x y : ℝ), given_line x y → (slope_of_perpendicular_line : ℝ) = 1/2 ∧ 
  ∀ (b : ℝ), line_y_intercept = (0, b) → ∀ (y : ℝ), y = 1/2 * x + b → (x, 0) = x_intercept_of_perpendicular_line) :=
sorry

end find_x_intercept_of_perpendicular_line_l92_9238


namespace certain_event_l92_9233

theorem certain_event (a : ℝ) : a^2 ≥ 0 := 
sorry

end certain_event_l92_9233


namespace c_finish_work_in_6_days_l92_9295

theorem c_finish_work_in_6_days (a b c : ℝ) (ha : a = 1/36) (hb : b = 1/18) (habc : a + b + c = 1/4) : c = 1/6 :=
by
  sorry

end c_finish_work_in_6_days_l92_9295


namespace least_number_divisible_by_38_and_3_remainder_1_exists_l92_9200

theorem least_number_divisible_by_38_and_3_remainder_1_exists :
  ∃ n, n % 38 = 1 ∧ n % 3 = 1 ∧ ∀ m, m % 38 = 1 ∧ m % 3 = 1 → n ≤ m :=
sorry

end least_number_divisible_by_38_and_3_remainder_1_exists_l92_9200


namespace vertex_coordinates_quadratic_through_point_a_less_than_neg_2_fifth_l92_9290

noncomputable def quadratic_function (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (a * x + 2 * a + 2)

theorem vertex_coordinates (a : ℝ) (H : a = 1) : 
    (∃ v_x v_y : ℝ, quadratic_function a v_x = v_y ∧ v_x = -5 / 2 ∧ v_y = -9 / 4) := 
by {
    sorry
}

theorem quadratic_through_point : 
    (∃ a : ℝ, (quadratic_function a 0 = -2) ∧ (∀ x, quadratic_function a x = -2 * (x + 1)^2)) := 
by {
    sorry
}

theorem a_less_than_neg_2_fifth 
  (x1 x2 y1 y2 a : ℝ) (H1 : x1 + x2 = 2) (H2 : x1 < x2) (H3 : y1 > y2) 
  (Hfunc : ∀ x, quadratic_function (a * x + 2 * a + 2) (x + 1) = quadratic_function x y) :
    a < -2 / 5 := 
by {
    sorry
}

end vertex_coordinates_quadratic_through_point_a_less_than_neg_2_fifth_l92_9290


namespace sum_infinite_geometric_l92_9280

theorem sum_infinite_geometric (a r : ℝ) (ha : a = 2) (hr : r = 1/3) : 
  ∑' n : ℕ, a * r^n = 3 := by
  sorry

end sum_infinite_geometric_l92_9280


namespace corrected_mean_l92_9287

theorem corrected_mean (n : ℕ) (mean incorrect_observation correct_observation : ℝ) (h_n : n = 50) (h_mean : mean = 32) (h_incorrect : incorrect_observation = 23) (h_correct : correct_observation = 48) : 
  (mean * n + (correct_observation - incorrect_observation)) / n = 32.5 := 
by 
  sorry

end corrected_mean_l92_9287


namespace proof_problem_l92_9281

variables (p q : Prop)

theorem proof_problem (h₁ : p) (h₂ : ¬ q) : ¬ p ∨ ¬ q :=
by
  sorry

end proof_problem_l92_9281


namespace age_of_new_teacher_l92_9261

-- Definitions of conditions
def avg_age_20_teachers (sum_of_ages : ℕ) : Prop :=
  sum_of_ages = 49 * 20

def avg_age_21_teachers (sum_of_ages : ℕ) : Prop :=
  sum_of_ages = 48 * 21

-- The proof goal
theorem age_of_new_teacher (sum_age_20 : ℕ) (sum_age_21 : ℕ) (h1 : avg_age_20_teachers sum_age_20) (h2 : avg_age_21_teachers sum_age_21) : 
  sum_age_21 - sum_age_20 = 28 :=
sorry

end age_of_new_teacher_l92_9261


namespace math_problem_l92_9244

theorem math_problem 
  (a b c : ℝ)
  (h1 : a < b)
  (h2 : ∀ x, (x < -2 ∨ |x - 30| ≤ 2) ↔ ( (x - a) * (x - b) / (x - c) ≤ 0 )) :
  a + 2 * b + 3 * c = 86 :=
sorry

end math_problem_l92_9244


namespace sum_of_coordinates_of_intersection_l92_9230

theorem sum_of_coordinates_of_intersection :
  let A := (0, 4)
  let B := (6, 0)
  let C := (9, 3)
  let D := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let E := ((B.1 + C.1) / 2, (B.2 + C.2) / 2)
  let line_AE := (fun x : ℚ => (-1/3) * x + 4)
  let line_CD := (fun x : ℚ => (1/6) * x + 1/2)
  let F_x := (21 : ℚ) / 3
  let F_y := line_AE F_x
  F_x + F_y = 26 / 3 := sorry

end sum_of_coordinates_of_intersection_l92_9230


namespace next_hexagon_dots_l92_9269

theorem next_hexagon_dots (base_dots : ℕ) (increment : ℕ) : base_dots = 2 → increment = 2 → 
  (2 + 6*2) + 6*(2*2) + 6*(3*2) + 6*(4*2) = 122 := 
by
  intros hbd hi
  sorry

end next_hexagon_dots_l92_9269


namespace cassidy_grounded_days_l92_9209

-- Definitions for the conditions
def days_for_lying : Nat := 14
def extra_days_per_grade : Nat := 3
def grades_below_B : Nat := 4

-- Definition for the total days grounded
def total_days_grounded : Nat :=
  days_for_lying + extra_days_per_grade * grades_below_B

-- The theorem statement
theorem cassidy_grounded_days :
  total_days_grounded = 26 := by
  sorry

end cassidy_grounded_days_l92_9209


namespace find_n_l92_9228

theorem find_n
  (c d : ℝ)
  (H1 : 450 * c + 300 * d = 300 * c + 375 * d)
  (H2 : ∃ t1 t2 t3 : ℝ, t1 = 4 ∧ t2 = 1 ∧ t3 = n ∧ 75 * 4 * (c + d) = 900 * c + t3 * d)
  : n = 600 / 7 :=
by
  sorry

end find_n_l92_9228


namespace prove_frac_addition_l92_9235

def frac_addition_correct : Prop :=
  (3 / 8 + 9 / 12 = 9 / 8)

theorem prove_frac_addition : frac_addition_correct :=
  by
  -- We assume the necessary fractions and their properties.
  sorry

end prove_frac_addition_l92_9235


namespace incorrect_residual_plot_statement_l92_9205

theorem incorrect_residual_plot_statement :
  ∀ (vertical_only_residual : Prop)
    (horizontal_any_of : Prop)
    (narrower_band_smaller_ssr : Prop)
    (narrower_band_smaller_corr : Prop)
    ,
    narrower_band_smaller_corr → False :=
  by intros vertical_only_residual horizontal_any_of narrower_band_smaller_ssr narrower_band_smaller_corr
     sorry

end incorrect_residual_plot_statement_l92_9205


namespace eric_has_correct_green_marbles_l92_9242

def total_marbles : ℕ := 20
def white_marbles : ℕ := 12
def blue_marbles : ℕ := 6
def green_marbles : ℕ := total_marbles - (white_marbles + blue_marbles)

theorem eric_has_correct_green_marbles : green_marbles = 2 :=
by
  sorry

end eric_has_correct_green_marbles_l92_9242


namespace unit_digit_seven_consecutive_l92_9278

theorem unit_digit_seven_consecutive (n : ℕ) : 
  (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) * (n + 5) * (n + 6)) % 10 = 0 := 
by
  sorry

end unit_digit_seven_consecutive_l92_9278


namespace number_of_routes_jack_to_jill_l92_9221

def num_routes_avoiding (start goal avoid : ℕ × ℕ) : ℕ := sorry

theorem number_of_routes_jack_to_jill : 
  num_routes_avoiding (0,0) (3,2) (1,1) = 4 :=
sorry

end number_of_routes_jack_to_jill_l92_9221


namespace count_distinct_rat_k_l92_9297

theorem count_distinct_rat_k : 
  (∃ N : ℕ, N = 108 ∧ ∀ k : ℚ, abs k < 300 → (∃ x : ℤ, 3 * x^2 + k * x + 20 = 0) →
  (∃! k, abs k < 300 ∧ (∃ x : ℤ, 3 * x^2 + k * x + 20 = 0))) :=
sorry

end count_distinct_rat_k_l92_9297


namespace candy_bar_cost_l92_9268

def initial_amount : ℕ := 4
def remaining_amount : ℕ := 3
def cost_of_candy_bar : ℕ := initial_amount - remaining_amount

theorem candy_bar_cost : cost_of_candy_bar = 1 := by
  sorry

end candy_bar_cost_l92_9268


namespace highest_car_color_is_blue_l92_9253

def total_cars : ℕ := 24
def red_cars : ℕ := total_cars / 4
def blue_cars : ℕ := red_cars + 6
def yellow_cars : ℕ := total_cars - (red_cars + blue_cars)

theorem highest_car_color_is_blue :
  blue_cars > red_cars ∧ blue_cars > yellow_cars :=
by sorry

end highest_car_color_is_blue_l92_9253


namespace alice_wins_with_optimal_strategy_l92_9274

theorem alice_wins_with_optimal_strategy :
  (∀ (N : ℕ) (X Y : ℕ), N = 270000 → N = X * Y → gcd X Y ≠ 1 → 
    (∃ (alice : ℕ → ℕ → Prop), ∀ N, ∃ (X Y : ℕ), alice N (X * Y) → gcd X Y ≠ 1) ∧
    (∀ (bob : ℕ → ℕ → ℕ → Prop), ∀ N X Y, bob N X Y → gcd X Y ≠ 1)) →
  (N : ℕ) → N = 270000 → gcd N 1 ≠ 1 :=
by
  sorry

end alice_wins_with_optimal_strategy_l92_9274


namespace height_difference_is_9_l92_9212

-- Definitions of the height of Petronas Towers and Empire State Building.
def height_Petronas : ℕ := 452
def height_EmpireState : ℕ := 443

-- Definition stating the height difference.
def height_difference := height_Petronas - height_EmpireState

-- Proving the height difference is 9 meters.
theorem height_difference_is_9 : height_difference = 9 :=
by
  -- the proof goes here
  sorry

end height_difference_is_9_l92_9212


namespace option_b_correct_l92_9284

theorem option_b_correct (a b : ℝ) (h : a ≠ b) : (1 / (a - b) + 1 / (b - a) = 0) :=
by
  sorry

end option_b_correct_l92_9284


namespace boys_from_other_communities_l92_9279

theorem boys_from_other_communities (total_boys : ℕ) (percent_muslims percent_hindus percent_sikhs : ℕ) 
    (h_total_boys : total_boys = 300)
    (h_percent_muslims : percent_muslims = 44)
    (h_percent_hindus : percent_hindus = 28)
    (h_percent_sikhs : percent_sikhs = 10) :
  ∃ (percent_others : ℕ), percent_others = 100 - (percent_muslims + percent_hindus + percent_sikhs) ∧ 
                             (percent_others * total_boys / 100) = 54 := 
by 
  sorry

end boys_from_other_communities_l92_9279


namespace apple_juice_less_than_cherry_punch_l92_9239

def orange_punch : ℝ := 4.5
def total_punch : ℝ := 21
def cherry_punch : ℝ := 2 * orange_punch
def combined_punch : ℝ := orange_punch + cherry_punch
def apple_juice : ℝ := total_punch - combined_punch

theorem apple_juice_less_than_cherry_punch : cherry_punch - apple_juice = 1.5 := by
  sorry

end apple_juice_less_than_cherry_punch_l92_9239


namespace value_of_y_l92_9262

theorem value_of_y (x y : ℝ) (h1 : x ^ (2 * y) = 81) (h2 : x = 9) : y = 1 :=
sorry

end value_of_y_l92_9262


namespace tens_digit_of_3_pow_2010_l92_9291

theorem tens_digit_of_3_pow_2010 : (3^2010 / 10) % 10 = 4 := by
  sorry

end tens_digit_of_3_pow_2010_l92_9291


namespace product_value_l92_9245

-- Definitions of each term
def term (n : Nat) : Rat :=
  1 + 1 / (n^2 : ℚ)

-- Define the product of these terms
def product : Rat :=
  term 1 * term 2 * term 3 * term 4 * term 5 * term 6

-- The proof problem statement that needs to be verified
theorem product_value :
  product = 16661 / 3240 :=
sorry

end product_value_l92_9245


namespace average_num_divisors_2019_l92_9257

def num_divisors (n : ℕ) : ℕ :=
  (n.divisors).card

theorem average_num_divisors_2019 :
  1 / 2019 * (Finset.sum (Finset.range 2020) num_divisors) = 15682 / 2019 :=
by
  sorry

end average_num_divisors_2019_l92_9257


namespace scientific_notation_of_114_trillion_l92_9258

theorem scientific_notation_of_114_trillion :
  (114 : ℝ) * 10^12 = (1.14 : ℝ) * 10^14 :=
by
  sorry

end scientific_notation_of_114_trillion_l92_9258


namespace line_equation_minimized_area_l92_9283

theorem line_equation_minimized_area :
  ∀ (l_1 l_2 l_3 : ℝ × ℝ → Prop) (l : ℝ × ℝ → Prop),
    (∀ x y : ℝ, l_1 (x, y) ↔ 3 * x + 2 * y - 1 = 0) ∧
    (∀ x y : ℝ, l_2 (x, y) ↔ 5 * x + 2 * y + 1 = 0) ∧
    (∀ x y : ℝ, l_3 (x, y) ↔ 3 * x - 5 * y + 6 = 0) →
    (∃ c : ℝ, ∀ x y : ℝ, l (x, y) ↔ 3 * x - 5 * y + c = 0) →
    (∃ x y : ℝ, l_1 (x, y) ∧ l_2 (x, y) ∧ l (x, y)) →
    (∀ a : ℝ, ∀ x y : ℝ, l (x, y) ↔ x + y = a) →
    (∃ k : ℝ, k > 0 ∧ ∀ x y : ℝ, l (x, y) ↔ 2 * x - y + 4 = 0) → 
    sorry :=
sorry

end line_equation_minimized_area_l92_9283


namespace AY_is_2_sqrt_55_l92_9213

noncomputable def AY_length : ℝ :=
  let rA := 10
  let rB := 3
  let AB := rA + rB
  let AD := rA - rB
  let BD := Real.sqrt (AB^2 - AD^2)
  2 * Real.sqrt (rA^2 + BD^2)

theorem AY_is_2_sqrt_55 :
  AY_length = 2 * Real.sqrt 55 :=
by
  -- Assuming the given problem's conditions.
  let rA := 10
  let rB := 3
  let AB := rA + rB
  let AD := rA - rB
  let BD := Real.sqrt (AB^2 - AD^2)
  show AY_length = 2 * Real.sqrt 55
  sorry

end AY_is_2_sqrt_55_l92_9213


namespace sum_of_nonzero_perfect_squares_l92_9203

theorem sum_of_nonzero_perfect_squares (p n : ℕ) (hp_prime : Nat.Prime p) 
    (hn_ge_p : n ≥ p) (h_perfect_square : ∃ k : ℕ, 1 + n * p = k^2) :
    ∃ (a : ℕ) (f : Fin p → ℕ), (∀ i, 0 < f i ∧ ∃ m, f i = m^2) ∧ (n + 1 = a + (Finset.univ.sum f)) :=
sorry

end sum_of_nonzero_perfect_squares_l92_9203


namespace emma_garden_area_l92_9289

-- Define the given conditions
def EmmaGarden (total_posts : ℕ) (posts_on_shorter_side : ℕ) (posts_on_longer_side : ℕ) (distance_between_posts : ℕ) : Prop :=
  total_posts = 24 ∧
  distance_between_posts = 6 ∧
  (posts_on_longer_side + 1) = 3 * (posts_on_shorter_side + 1) ∧
  2 * (posts_on_shorter_side + 1 + posts_on_longer_side + 1) = 24

-- The theorem to prove
theorem emma_garden_area : ∃ (length width : ℕ), EmmaGarden 24 2 8 6 ∧ (length = 6 * (2) ∧ width = 6 * (8 - 1)) ∧ (length * width = 576) :=
by
  -- proof goes here
  sorry

end emma_garden_area_l92_9289


namespace wall_length_l92_9264

theorem wall_length
    (brick_length brick_width brick_height : ℝ)
    (wall_height wall_width : ℝ)
    (num_bricks : ℕ)
    (wall_length_cm : ℝ)
    (h_brick_volume : brick_length * brick_width * brick_height = 1687.5)
    (h_wall_volume :
        wall_length_cm * wall_height * wall_width
        = (brick_length * brick_width * brick_height) * num_bricks)
    (h_wall_height : wall_height = 600)
    (h_wall_width : wall_width = 22.5)
    (h_num_bricks : num_bricks = 7200) :
    wall_length_cm / 100 = 9 := 
by
  sorry

end wall_length_l92_9264


namespace infinite_solutions_for_equation_l92_9207

theorem infinite_solutions_for_equation :
  ∃ (x y z : ℤ), x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ ∀ (k : ℤ), (x^2 + y^5 = z^3) :=
sorry

end infinite_solutions_for_equation_l92_9207


namespace angle_measure_is_fifty_l92_9216

theorem angle_measure_is_fifty (x : ℝ) :
  (90 - x = (1 / 2) * (180 - x) - 25) → x = 50 := by
  intro h
  sorry

end angle_measure_is_fifty_l92_9216


namespace matches_in_round1_group1_matches_in_round1_group2_matches_in_round2_l92_9208

def num_teams_group1 : ℕ := 3
def num_teams_group2 : ℕ := 4

def num_matches_round1_group1 (n : ℕ) : ℕ := n * (n - 1) / 2
def num_matches_round1_group2 (n : ℕ) : ℕ := n * (n - 1) / 2

def num_matches_round2 (n1 n2 : ℕ) : ℕ := n1 * n2

theorem matches_in_round1_group1 : num_matches_round1_group1 num_teams_group1 = 3 := 
by
  -- Exact proof steps should be filled in here.
  sorry

theorem matches_in_round1_group2 : num_matches_round1_group2 num_teams_group2 = 6 := 
by
  -- Exact proof steps should be filled in here.
  sorry

theorem matches_in_round2 : num_matches_round2 num_teams_group1 num_teams_group2 = 12 := 
by
  -- Exact proof steps should be filled in here.
  sorry

end matches_in_round1_group1_matches_in_round1_group2_matches_in_round2_l92_9208


namespace integer_to_sixth_power_l92_9293

theorem integer_to_sixth_power (a b : ℕ) (h : 3^a * 3^b = 3^(a + b)) (ha : a = 12) (hb : b = 18) : 
  ∃ x : ℕ, x = 243 ∧ x^6 = 3^(a + b) :=
by
  sorry

end integer_to_sixth_power_l92_9293


namespace inequality_holds_for_all_real_numbers_l92_9226

theorem inequality_holds_for_all_real_numbers :
  ∀ x y z : ℝ, 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_holds_for_all_real_numbers_l92_9226


namespace find_number_l92_9286

def incorrect_multiplication (x : ℕ) : ℕ := 394 * x
def correct_multiplication (x : ℕ) : ℕ := 493 * x
def difference (x : ℕ) : ℕ := correct_multiplication x - incorrect_multiplication x
def expected_difference : ℕ := 78426

theorem find_number (x : ℕ) (h : difference x = expected_difference) : x = 792 := by
  sorry

end find_number_l92_9286


namespace probability_two_red_balls_l92_9256

open Nat

theorem probability_two_red_balls (total_balls red_balls blue_balls green_balls balls_picked : Nat) 
  (total_eq : total_balls = red_balls + blue_balls + green_balls) 
  (red_eq : red_balls = 7) 
  (blue_eq : blue_balls = 5) 
  (green_eq : green_balls = 4) 
  (picked_eq : balls_picked = 2) :
  (choose red_balls balls_picked) / (choose total_balls balls_picked) = 7 / 40 :=
by
  sorry

end probability_two_red_balls_l92_9256


namespace weight_difference_calc_l92_9220

-- Define the weights in pounds
def Anne_weight : ℕ := 67
def Douglas_weight : ℕ := 52
def Maria_weight : ℕ := 48

-- Define the combined weight of Douglas and Maria
def combined_weight_DM : ℕ := Douglas_weight + Maria_weight

-- Define the weight difference
def weight_difference : ℤ := Anne_weight - combined_weight_DM

-- The theorem stating the difference
theorem weight_difference_calc : weight_difference = -33 := by
  -- The proof will go here
  sorry

end weight_difference_calc_l92_9220


namespace jebb_expense_l92_9241

-- Define the costs
def seafood_platter := 45.0
def rib_eye_steak := 38.0
def vintage_wine_glass := 18.0
def chocolate_dessert := 12.0

-- Define the rules and discounts
def discount_percentage := 0.10
def service_fee_12 := 0.12
def service_fee_15 := 0.15
def tip_percentage := 0.20

-- Total food and wine cost
def total_food_and_wine_cost := 
  seafood_platter + rib_eye_steak + (2 * vintage_wine_glass) + chocolate_dessert

-- Total food cost excluding wine
def food_cost_excluding_wine := 
  seafood_platter + rib_eye_steak + chocolate_dessert

-- 10% discount on food cost excluding wine
def discount_amount := discount_percentage * food_cost_excluding_wine
def reduced_food_cost := food_cost_excluding_wine - discount_amount

-- New total cost before applying the service fee
def total_cost_before_service_fee := reduced_food_cost + (2 * vintage_wine_glass)

-- Determine the service fee based on cost
def service_fee := 
  if total_cost_before_service_fee > 80.0 then 
    service_fee_15 * total_cost_before_service_fee 
  else if total_cost_before_service_fee >= 50.0 then 
    service_fee_12 * total_cost_before_service_fee 
  else 
    0.0

-- Total cost after discount and service fee
def total_cost_after_service_fee := total_cost_before_service_fee + service_fee

-- Tip amount (20% of total cost after discount and service fee)
def tip_amount := tip_percentage * total_cost_after_service_fee

-- Total amount Jebb spent
def total_amount_spent := total_cost_after_service_fee + tip_amount

-- Lean theorem statement
theorem jebb_expense :
  total_amount_spent = 167.67 :=
by
  -- prove the theorem here
  sorry

end jebb_expense_l92_9241


namespace volume_at_10_l92_9217

noncomputable def gas_volume (T : ℝ) : ℝ :=
  if T = 30 then 40 else 40 - (30 - T) / 5 * 5

theorem volume_at_10 :
  gas_volume 10 = 20 :=
by
  simp [gas_volume]
  sorry

end volume_at_10_l92_9217
