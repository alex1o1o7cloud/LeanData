import Mathlib

namespace NUMINAMATH_GPT_percentage_increase_twice_l739_73900

theorem percentage_increase_twice (P : ℝ) (x : ℝ) :
  P * (1 + x)^2 = P * 1.3225 → x = 0.15 :=
by
  intro h
  have h1 : (1 + x)^2 = 1.3225 := by sorry
  have h2 : x^2 + 2 * x = 0.3225 := by sorry
  have h3 : x = (-2 + Real.sqrt 5.29) / 2 := by sorry
  have h4 : x = -2 / 2 + Real.sqrt 5.29 / 2 := by sorry
  have h5 : x = 0.15 := by sorry
  exact h5

end NUMINAMATH_GPT_percentage_increase_twice_l739_73900


namespace NUMINAMATH_GPT_triangle_angles_correct_l739_73953

open Real

noncomputable def angle_triple (A B C : ℝ) : Prop :=
  ∃ (a b c : ℝ),
    a = 2 * b * cos C ∧ 
    sin A * sin (B / 2 + C) = sin C * (sin (B / 2) + sin A)

theorem triangle_angles_correct (A B C : ℝ) (h : angle_triple A B C) :
  A = 5 * π / 9 ∧ B = 2 * π / 9 ∧ C = 2 * π / 9 := 
sorry

end NUMINAMATH_GPT_triangle_angles_correct_l739_73953


namespace NUMINAMATH_GPT_hiking_hours_l739_73986

theorem hiking_hours
  (violet_water_per_hour : ℕ := 800)
  (dog_water_per_hour : ℕ := 400)
  (total_water : ℕ := 4800) :
  (total_water / (violet_water_per_hour + dog_water_per_hour) = 4) :=
by
  sorry

end NUMINAMATH_GPT_hiking_hours_l739_73986


namespace NUMINAMATH_GPT_disease_cases_linear_decrease_l739_73951

theorem disease_cases_linear_decrease (cases_1970 cases_2010 cases_1995 cases_2005 : ℕ)
  (year_1970 year_2010 year_1995 year_2005 : ℕ)
  (h_cases_1970 : cases_1970 = 800000)
  (h_cases_2010 : cases_2010 = 200)
  (h_year_1970 : year_1970 = 1970)
  (h_year_2010 : year_2010 = 2010)
  (h_year_1995 : year_1995 = 1995)
  (h_year_2005 : year_2005 = 2005)
  (linear_decrease : ∀ t, cases_1970 - (cases_1970 - cases_2010) * (t - year_1970) / (year_2010 - year_1970) = cases_1970 - t * (cases_1970 - cases_2010) / (year_2010 - year_1970))
  : cases_1995 = 300125 ∧ cases_2005 = 100175 := sorry

end NUMINAMATH_GPT_disease_cases_linear_decrease_l739_73951


namespace NUMINAMATH_GPT_intersection_with_complement_l739_73926

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {0, 2, 4}

theorem intersection_with_complement (hU : U = {0, 1, 2, 3, 4})
                                     (hA : A = {0, 1, 2, 3})
                                     (hB : B = {0, 2, 4}) :
  A ∩ (U \ B) = {1, 3} :=
by sorry

end NUMINAMATH_GPT_intersection_with_complement_l739_73926


namespace NUMINAMATH_GPT_range_of_a_l739_73904

-- Define the inequality problem
def inequality_always_true (a : ℝ) : Prop :=
  ∀ x, a * x^2 + 3 * a * x + a - 2 < 0

-- Define the range condition for "a"
def range_condition (a : ℝ) : Prop :=
  (a = 0 ∧ (-2 < 0)) ∨
  (a ≠ 0 ∧ a < 0 ∧ a * (5 * a + 8) < 0)

-- The main theorem stating the equivalence
theorem range_of_a (a : ℝ) : inequality_always_true a ↔ a ∈ Set.Icc (- (8 / 5)) 0 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l739_73904


namespace NUMINAMATH_GPT_S8_is_255_l739_73985

-- Definitions and hypotheses
def geometric_sequence_sum (a : ℕ → ℚ) (q : ℚ) (n : ℕ) : ℚ :=
  a 0 * (1 - q^n) / (1 - q)

variables (a : ℕ → ℚ) (q : ℚ)
variable (h_geo_seq : ∀ n, a (n + 1) = a n * q)
variable (h_S2 : geometric_sequence_sum a q 2 = 3)
variable (h_S4 : geometric_sequence_sum a q 4 = 15)

-- Goal
theorem S8_is_255 : geometric_sequence_sum a q 8 = 255 := 
by {
  -- skipping the proof
  sorry
}

end NUMINAMATH_GPT_S8_is_255_l739_73985


namespace NUMINAMATH_GPT_karen_nuts_l739_73992

/-- Karen added 0.25 cup of walnuts to a batch of trail mix.
Later, she added 0.25 cup of almonds.
In all, Karen put 0.5 cups of nuts in the trail mix. -/
theorem karen_nuts (walnuts almonds : ℝ) 
  (h_walnuts : walnuts = 0.25) 
  (h_almonds : almonds = 0.25) : 
  walnuts + almonds = 0.5 := 
by
  sorry

end NUMINAMATH_GPT_karen_nuts_l739_73992


namespace NUMINAMATH_GPT_kareem_largest_l739_73952

def jose_final : ℕ :=
  let start := 15
  let minus_two := start - 2
  let triple := minus_two * 3
  triple + 5

def thuy_final : ℕ :=
  let start := 15
  let triple := start * 3
  let minus_two := triple - 2
  minus_two + 5

def kareem_final : ℕ :=
  let start := 15
  let minus_two := start - 2
  let add_five := minus_two + 5
  add_five * 3

theorem kareem_largest : kareem_final > jose_final ∧ kareem_final > thuy_final := by
  sorry

end NUMINAMATH_GPT_kareem_largest_l739_73952


namespace NUMINAMATH_GPT_find_smallest_integer_l739_73927

/-- There exists an integer n such that:
   n ≡ 1 [MOD 3],
   n ≡ 2 [MOD 4],
   n ≡ 3 [MOD 5],
   and the smallest such n is 58. -/
theorem find_smallest_integer :
  ∃ n : ℕ, n % 3 = 1 ∧ n % 4 = 2 ∧ n % 5 = 3 ∧ n = 58 :=
by
  -- Proof goes here (not provided as per the instructions)
  sorry

end NUMINAMATH_GPT_find_smallest_integer_l739_73927


namespace NUMINAMATH_GPT_Willey_Farm_Available_Capital_l739_73970

theorem Willey_Farm_Available_Capital 
  (total_acres : ℕ)
  (cost_per_acre_corn : ℕ)
  (cost_per_acre_wheat : ℕ)
  (acres_wheat : ℕ)
  (available_capital : ℕ) :
  total_acres = 4500 →
  cost_per_acre_corn = 42 →
  cost_per_acre_wheat = 35 →
  acres_wheat = 3400 →
  available_capital = (acres_wheat * cost_per_acre_wheat) + 
                      ((total_acres - acres_wheat) * cost_per_acre_corn) →
  available_capital = 165200 := sorry

end NUMINAMATH_GPT_Willey_Farm_Available_Capital_l739_73970


namespace NUMINAMATH_GPT_children_got_off_bus_l739_73941

theorem children_got_off_bus :
  ∀ (initial_children final_children new_children off_children : ℕ),
    initial_children = 21 → final_children = 16 → new_children = 5 →
    initial_children - off_children + new_children = final_children →
    off_children = 10 :=
by
  intro initial_children final_children new_children off_children
  intros h_init h_final h_new h_eq
  sorry

end NUMINAMATH_GPT_children_got_off_bus_l739_73941


namespace NUMINAMATH_GPT_y_intercepts_parabola_l739_73998

theorem y_intercepts_parabola : 
  ∀ (y : ℝ), ¬(0 = 3 * y^2 - 5 * y + 12) :=
by 
  -- Given x = 0, we have the equation 3 * y^2 - 5 * y + 12 = 0.
  -- The discriminant ∆ = b^2 - 4ac = (-5)^2 - 4 * 3 * 12 = 25 - 144 = -119 which is less than 0.
  -- Since the discriminant is negative, the quadratic equation has no real roots.
  sorry

end NUMINAMATH_GPT_y_intercepts_parabola_l739_73998


namespace NUMINAMATH_GPT_seeds_in_fourth_pot_l739_73942

-- Define the conditions as variables
def total_seeds : ℕ := 10
def number_of_pots : ℕ := 4
def seeds_per_pot : ℕ := 3

-- Define the theorem to prove the quantity of seeds planted in the fourth pot
theorem seeds_in_fourth_pot :
  (total_seeds - (seeds_per_pot * (number_of_pots - 1))) = 1 := by
  sorry

end NUMINAMATH_GPT_seeds_in_fourth_pot_l739_73942


namespace NUMINAMATH_GPT_quadratic_roots_expression_l739_73976

theorem quadratic_roots_expression {m n : ℝ}
  (h₁ : m^2 + m - 12 = 0)
  (h₂ : n^2 + n - 12 = 0)
  (h₃ : m + n = -1) :
  m^2 + 2 * m + n = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_roots_expression_l739_73976


namespace NUMINAMATH_GPT_multiple_of_remainder_l739_73963

theorem multiple_of_remainder (R V D Q k : ℤ) (h1 : R = 6) (h2 : V = 86) (h3 : D = 5 * Q) 
  (h4 : D = k * R + 2) (h5 : V = D * Q + R) : k = 3 := by
  sorry

end NUMINAMATH_GPT_multiple_of_remainder_l739_73963


namespace NUMINAMATH_GPT_correct_operation_l739_73946

-- Definitions based on conditions
def exprA (a b : ℤ) : ℤ := 3 * a * b - a * b
def exprB (a : ℤ) : ℤ := -3 * a^2 - 5 * a^2
def exprC (x : ℤ) : ℤ := -3 * x - 2 * x

-- Statement to prove that exprB is correct
theorem correct_operation (a : ℤ) : exprB a = -8 * a^2 := by
  sorry

end NUMINAMATH_GPT_correct_operation_l739_73946


namespace NUMINAMATH_GPT_nearest_integer_is_11304_l739_73964

def nearest_integer_to_a_plus_b_pow_six (a b : ℝ) (h : b = Real.sqrt 5) : ℝ :=
  (a + b) ^ 6

theorem nearest_integer_is_11304 : nearest_integer_to_a_plus_b_pow_six 3 (Real.sqrt 5) rfl = 11304 := 
  sorry

end NUMINAMATH_GPT_nearest_integer_is_11304_l739_73964


namespace NUMINAMATH_GPT_total_bill_for_group_is_129_l739_73972

theorem total_bill_for_group_is_129 :
  let num_adults := 6
  let num_teenagers := 3
  let num_children := 1
  let cost_adult_meal := 9
  let cost_teenager_meal := 7
  let cost_child_meal := 5
  let cost_soda := 2.50
  let num_sodas := 10
  let cost_dessert := 4
  let num_desserts := 3
  let cost_appetizer := 6
  let num_appetizers := 2
  let total_bill := 
    (num_adults * cost_adult_meal) +
    (num_teenagers * cost_teenager_meal) +
    (num_children * cost_child_meal) +
    (num_sodas * cost_soda) +
    (num_desserts * cost_dessert) +
    (num_appetizers * cost_appetizer)
  total_bill = 129 := by
sorry

end NUMINAMATH_GPT_total_bill_for_group_is_129_l739_73972


namespace NUMINAMATH_GPT_jessica_attended_games_l739_73916

/-- 
Let total_games be the total number of soccer games.
Let initially_planned be the number of games Jessica initially planned to attend.
Let commitments_skipped be the number of games skipped due to other commitments.
Let rescheduled_games be the rescheduled games during the season.
Let additional_missed be the additional games missed due to rescheduling.
-/
theorem jessica_attended_games
    (total_games initially_planned commitments_skipped rescheduled_games additional_missed : ℕ)
    (h1 : total_games = 12)
    (h2 : initially_planned = 8)
    (h3 : commitments_skipped = 3)
    (h4 : rescheduled_games = 2)
    (h5 : additional_missed = 4) :
    (initially_planned - commitments_skipped) - additional_missed = 1 := by
  sorry

end NUMINAMATH_GPT_jessica_attended_games_l739_73916


namespace NUMINAMATH_GPT_convert_octal_127_to_binary_l739_73965

def octal_to_binary (n : ℕ) : ℕ :=
  match n with
  | 1 => 3  -- 001 in binary
  | 2 => 2  -- 010 in binary
  | 7 => 7  -- 111 in binary
  | _ => 0  -- No other digits are used in this example

theorem convert_octal_127_to_binary :
  octal_to_binary 1 * 1000000 + octal_to_binary 2 * 1000 + octal_to_binary 7 = 1010111 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_convert_octal_127_to_binary_l739_73965


namespace NUMINAMATH_GPT_point_D_coordinates_l739_73958

theorem point_D_coordinates 
  (F : (ℕ × ℕ)) 
  (coords_F : F = (5,5)) 
  (D : (ℕ × ℕ)) 
  (coords_D : D = (2,4)) :
  (D = (2,4)) :=
by 
  sorry

end NUMINAMATH_GPT_point_D_coordinates_l739_73958


namespace NUMINAMATH_GPT_min_value_expression_l739_73908

theorem min_value_expression (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 2 * b = 1) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt 10 + 6 ∧ (∀ x y : ℝ, (x > 0) → (y > 0) → (a + 2 * y = 1) → ( (y^2 + a + 1) / (a * y)  ≥  c )) :=
sorry

end NUMINAMATH_GPT_min_value_expression_l739_73908


namespace NUMINAMATH_GPT_geometric_sequence_sum_l739_73973

variable {α : Type*} 
variable [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) : Prop :=
  ∃ q : α, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → α) (h : is_geometric_sequence a) 
  (h1 : a 0 + a 1 = 20) 
  (h2 : a 2 + a 3 = 40) : 
  a 4 + a 5 = 80 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l739_73973


namespace NUMINAMATH_GPT_final_price_is_correct_l739_73921

-- Define the original price
def original_price : ℝ := 10

-- Define the first reduction percentage
def first_reduction_percentage : ℝ := 0.30

-- Define the second reduction percentage
def second_reduction_percentage : ℝ := 0.50

-- Define the price after the first reduction
def price_after_first_reduction : ℝ := original_price * (1 - first_reduction_percentage)

-- Define the final price after the second reduction
def final_price : ℝ := price_after_first_reduction * (1 - second_reduction_percentage)

-- Theorem to prove the final price is $3.50
theorem final_price_is_correct : final_price = 3.50 := by
  sorry

end NUMINAMATH_GPT_final_price_is_correct_l739_73921


namespace NUMINAMATH_GPT_eastville_to_westpath_travel_time_l739_73948

theorem eastville_to_westpath_travel_time :
  ∀ (d t₁ t₂ : ℝ) (s₁ s₂ : ℝ), 
  t₁ = 6 → s₁ = 80 → s₂ = 50 → d = s₁ * t₁ → t₂ = d / s₂ → t₂ = 9.6 := 
by
  intros d t₁ t₂ s₁ s₂ ht₁ hs₁ hs₂ hd ht₂
  sorry

end NUMINAMATH_GPT_eastville_to_westpath_travel_time_l739_73948


namespace NUMINAMATH_GPT_find_budget_l739_73983

variable (B : ℝ)

-- Conditions provided
axiom cond1 : 0.30 * B = 300

theorem find_budget : B = 1000 :=
by
  -- Notes:
  -- The proof will go here.
  sorry

end NUMINAMATH_GPT_find_budget_l739_73983


namespace NUMINAMATH_GPT_mass_percentage_of_calcium_in_calcium_oxide_l739_73979

theorem mass_percentage_of_calcium_in_calcium_oxide
  (Ca_molar_mass : ℝ)
  (O_molar_mass : ℝ)
  (Ca_mass : Ca_molar_mass = 40.08)
  (O_mass : O_molar_mass = 16.00) :
  ((Ca_molar_mass / (Ca_molar_mass + O_molar_mass)) * 100) = 71.45 :=
by
  sorry

end NUMINAMATH_GPT_mass_percentage_of_calcium_in_calcium_oxide_l739_73979


namespace NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l739_73943

variables (x : ℝ)

theorem equation_one_solution (h : 2 * (x + 3) = 5 * x) : x = 2 :=
sorry

theorem equation_two_solution (h : (x - 3) / 0.5 - (x + 4) / 0.2 = 1.6) : x = -9.2 :=
sorry

end NUMINAMATH_GPT_equation_one_solution_equation_two_solution_l739_73943


namespace NUMINAMATH_GPT_opposite_of_negative_rational_l739_73907

theorem opposite_of_negative_rational : - (-(4/3)) = (4/3) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_negative_rational_l739_73907


namespace NUMINAMATH_GPT_pencil_distribution_l739_73915

theorem pencil_distribution (n : ℕ) (friends : ℕ): 
  (friends = 4) → (n = 8) → 
  (∃ A B C D : ℕ, A ≥ 2 ∧ B ≥ 1 ∧ C ≥ 1 ∧ D ≥ 1 ∧ A + B + C + D = n) →
  (∃! k : ℕ, k = 20) :=
by
  intros friends_eq n_eq h
  use 20
  sorry

end NUMINAMATH_GPT_pencil_distribution_l739_73915


namespace NUMINAMATH_GPT_Sandy_tokens_more_than_siblings_l739_73956

theorem Sandy_tokens_more_than_siblings :
  let total_tokens := 1000000
  let siblings := 4
  let Sandy_tokens := total_tokens / 2
  let sibling_tokens := (total_tokens - Sandy_tokens) / siblings
  Sandy_tokens - sibling_tokens = 375000 :=
by
  -- Definitions as per conditions
  let total_tokens := 1000000
  let siblings := 4
  let Sandy_tokens := total_tokens / 2
  let sibling_tokens := (total_tokens - Sandy_tokens) / siblings
  -- Conclusion
  show Sandy_tokens - sibling_tokens = 375000
  sorry

end NUMINAMATH_GPT_Sandy_tokens_more_than_siblings_l739_73956


namespace NUMINAMATH_GPT_minimum_value_of_function_l739_73902

theorem minimum_value_of_function : ∀ x : ℝ, (x^2 + 9) / Real.sqrt (x^2 + 3) ≥ 2 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_function_l739_73902


namespace NUMINAMATH_GPT_find_a_l739_73905

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  (x^2 / 4) - y^2 = 1 ∧ x ≥ 2

theorem find_a (a : ℝ) : 
  (∃ x y : ℝ, point_on_hyperbola x y ∧ (min ((x - a)^2 + y^2) = 3)) → 
  (a = -1 ∨ a = 2 * Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_find_a_l739_73905


namespace NUMINAMATH_GPT_cannot_achieve_61_cents_with_six_coins_l739_73937

theorem cannot_achieve_61_cents_with_six_coins :
  ¬ ∃ (p n d q : ℕ), 
      p + n + d + q = 6 ∧ 
      p + 5 * n + 10 * d + 25 * q = 61 :=
by
  sorry

end NUMINAMATH_GPT_cannot_achieve_61_cents_with_six_coins_l739_73937


namespace NUMINAMATH_GPT_plane_difference_correct_l739_73933

noncomputable def max_planes : ℕ := 27
noncomputable def min_planes : ℕ := 7
noncomputable def diff_planes : ℕ := max_planes - min_planes

theorem plane_difference_correct : diff_planes = 20 := by
  sorry

end NUMINAMATH_GPT_plane_difference_correct_l739_73933


namespace NUMINAMATH_GPT_dhoni_remaining_earnings_l739_73936

theorem dhoni_remaining_earnings :
  let rent := 0.20
  let dishwasher := 0.15
  let bills := 0.10
  let car := 0.08
  let grocery := 0.12
  let tax := 0.05
  let expenses := rent + dishwasher + bills + car + grocery + tax
  let remaining_after_expenses := 1.0 - expenses
  let savings := 0.40 * remaining_after_expenses
  let remaining_after_savings := remaining_after_expenses - savings
  remaining_after_savings = 0.18 := by
sorry

end NUMINAMATH_GPT_dhoni_remaining_earnings_l739_73936


namespace NUMINAMATH_GPT_find_x_l739_73968

theorem find_x (x : ℝ) :
  let P1 := (2, 10)
  let P2 := (6, 2)
  
  -- Slope of the line joining (2, 10) and (6, 2)
  let slope12 := (P2.2 - P1.2) / (P2.1 - P1.1)
  
  -- Slope of the line joining (2, 10) and (x, -3)
  let P3 := (x, -3)
  let slope13 := (P3.2 - P1.2) / (P3.1 - P1.1)
  
  -- Condition that both slopes are equal
  slope12 = slope13
  
  -- To Prove: x must be 8.5
  → x = 8.5 :=
sorry

end NUMINAMATH_GPT_find_x_l739_73968


namespace NUMINAMATH_GPT_islander_distances_l739_73994

theorem islander_distances (A B C D : ℕ) (k1 : A = 1 ∨ A = 2)
  (k2 : B = 2)
  (C_liar : C = 1) (is_knight : C ≠ 1) :
  C = 1 ∨ C = 3 ∨ C = 4 ∧ D = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_islander_distances_l739_73994


namespace NUMINAMATH_GPT_total_students_are_45_l739_73957

theorem total_students_are_45 (burgers hot_dogs students : ℕ)
  (h1 : burgers = 30)
  (h2 : burgers = 2 * hot_dogs)
  (h3 : students = burgers + hot_dogs) : students = 45 :=
sorry

end NUMINAMATH_GPT_total_students_are_45_l739_73957


namespace NUMINAMATH_GPT_isosceles_triangle_l739_73912

theorem isosceles_triangle (a b c : ℝ) (h : a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b) = 0) : 
  a = b ∨ b = c ∨ c = a :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_l739_73912


namespace NUMINAMATH_GPT_division_and_multiplication_l739_73901

theorem division_and_multiplication (x : ℝ) (h : x = 9) : (x / 6 * 12) = 18 := by
  sorry

end NUMINAMATH_GPT_division_and_multiplication_l739_73901


namespace NUMINAMATH_GPT_percentage_markup_l739_73925

theorem percentage_markup 
  (selling_price : ℝ) 
  (cost_price : ℝ) 
  (h1 : selling_price = 8215)
  (h2 : cost_price = 6625)
  : ((selling_price - cost_price) / cost_price) * 100 = 24 := 
  by
    sorry

end NUMINAMATH_GPT_percentage_markup_l739_73925


namespace NUMINAMATH_GPT_abs_eq_sqrt_five_l739_73990

theorem abs_eq_sqrt_five (x : ℝ) (h : |x| = Real.sqrt 5) : x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := 
sorry

end NUMINAMATH_GPT_abs_eq_sqrt_five_l739_73990


namespace NUMINAMATH_GPT_find_a_l739_73982

theorem find_a (a : ℝ) (h : 2 * a + 3 = -3) : a = -3 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l739_73982


namespace NUMINAMATH_GPT_carbon_copies_after_folding_l739_73928

def initial_sheets : ℕ := 6
def initial_carbons (sheets : ℕ) : ℕ := sheets - 1
def final_copies (sheets : ℕ) : ℕ := sheets - 1

theorem carbon_copies_after_folding :
  (final_copies initial_sheets) =
  initial_carbons initial_sheets :=
by {
    -- sorry is a placeholder for the proof
    sorry
}

end NUMINAMATH_GPT_carbon_copies_after_folding_l739_73928


namespace NUMINAMATH_GPT_probability_of_X_l739_73922

variable (P : Prop → ℝ)
variable (event_X event_Y : Prop)

-- Defining the conditions
variable (hYP : P event_Y = 2 / 3)
variable (hXYP : P (event_X ∧ event_Y) = 0.13333333333333333)

-- Proving that the probability of selection of X is 0.2
theorem probability_of_X : P event_X = 0.2 := by
  sorry

end NUMINAMATH_GPT_probability_of_X_l739_73922


namespace NUMINAMATH_GPT_ducks_and_geese_meeting_l739_73929

theorem ducks_and_geese_meeting:
  ∀ x : ℕ, ( ∀ ducks_speed : ℚ, ducks_speed = (1/7) ) → 
         ( ∀ geese_speed : ℚ, geese_speed = (1/9) ) → 
         (ducks_speed * x + geese_speed * x = 1) :=
by
  sorry

end NUMINAMATH_GPT_ducks_and_geese_meeting_l739_73929


namespace NUMINAMATH_GPT_new_rate_ratio_l739_73932

/--
Hephaestus charged 3 golden apples for the first six months and raised his rate halfway through the year.
Apollo paid 54 golden apples in total for the entire year.
The ratio of the new rate to the old rate is 2.
-/
theorem new_rate_ratio
  (old_rate new_rate : ℕ)
  (total_payment : ℕ)
  (H1 : old_rate = 3)
  (H2 : total_payment = 54)
  (H3 : ∀ R : ℕ, new_rate = R * old_rate ∧ total_payment = 18 + 18 * R) :
  ∃ (R : ℕ), R = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_new_rate_ratio_l739_73932


namespace NUMINAMATH_GPT_inverse_is_correct_l739_73996

-- Definitions
def original_proposition (n : ℤ) : Prop := n < 0 → n ^ 2 > 0
def inverse_proposition (n : ℤ) : Prop := n ^ 2 > 0 → n < 0

-- Theorem stating the inverse
theorem inverse_is_correct : 
  (∀ n : ℤ, original_proposition n) → (∀ n : ℤ, inverse_proposition n) :=
by
  sorry

end NUMINAMATH_GPT_inverse_is_correct_l739_73996


namespace NUMINAMATH_GPT_tangent_lines_to_curve_at_l739_73995

noncomputable
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

noncomputable
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 2) * x

noncomputable
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + (a - 2)

theorem tangent_lines_to_curve_at (a : ℝ) :
  is_even_function (f' a) →
  (∀ x, f a x = - 2 → (2*x + (- f a x) = 0 ∨ 19*x - 4*(- f a x) - 27 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_tangent_lines_to_curve_at_l739_73995


namespace NUMINAMATH_GPT_carlos_cycles_more_than_diana_l739_73935

theorem carlos_cycles_more_than_diana :
  let slope_carlos := 1
  let slope_diana := 0.75
  let rate_carlos := slope_carlos * 20
  let rate_diana := slope_diana * 20
  let distance_carlos_after_3_hours := 3 * rate_carlos
  let distance_diana_after_3_hours := 3 * rate_diana
  distance_carlos_after_3_hours - distance_diana_after_3_hours = 15 :=
sorry

end NUMINAMATH_GPT_carlos_cycles_more_than_diana_l739_73935


namespace NUMINAMATH_GPT_calculate_expression_l739_73923

def smallest_positive_two_digit_multiple_of_7 : ℕ := 14
def smallest_positive_three_digit_multiple_of_5 : ℕ := 100

theorem calculate_expression : 
  let c := smallest_positive_two_digit_multiple_of_7
  let d := smallest_positive_three_digit_multiple_of_5
  (c * d) - 100 = 1300 :=
by 
  let c := smallest_positive_two_digit_multiple_of_7
  let d := smallest_positive_three_digit_multiple_of_5
  sorry

end NUMINAMATH_GPT_calculate_expression_l739_73923


namespace NUMINAMATH_GPT_gym_membership_count_l739_73961

theorem gym_membership_count :
  let charge_per_time := 18
  let times_per_month := 2
  let total_monthly_income := 10800
  let amount_per_member := charge_per_time * times_per_month
  let number_of_members := total_monthly_income / amount_per_member
  number_of_members = 300 :=
by
  let charge_per_time := 18
  let times_per_month := 2
  let total_monthly_income := 10800
  let amount_per_member := charge_per_time * times_per_month
  let number_of_members := total_monthly_income / amount_per_member
  sorry

end NUMINAMATH_GPT_gym_membership_count_l739_73961


namespace NUMINAMATH_GPT_Tim_weekly_water_intake_l739_73987

variable (daily_bottle_intake : ℚ)
variable (additional_intake : ℚ)
variable (quart_to_ounces : ℚ)
variable (days_in_week : ℕ := 7)

theorem Tim_weekly_water_intake (H1 : daily_bottle_intake = 2 * 1.5)
                              (H2 : additional_intake = 20)
                              (H3 : quart_to_ounces = 32) :
  (daily_bottle_intake * quart_to_ounces + additional_intake) * days_in_week = 812 := by
  sorry

end NUMINAMATH_GPT_Tim_weekly_water_intake_l739_73987


namespace NUMINAMATH_GPT_remaining_money_l739_73993

def potato_cost : ℕ := 6 * 2
def tomato_cost : ℕ := 9 * 3
def cucumber_cost : ℕ := 5 * 4
def banana_cost : ℕ := 3 * 5
def total_cost : ℕ := potato_cost + tomato_cost + cucumber_cost + banana_cost
def initial_money : ℕ := 500

theorem remaining_money : initial_money - total_cost = 426 :=
by
  sorry

end NUMINAMATH_GPT_remaining_money_l739_73993


namespace NUMINAMATH_GPT_emma_reaches_jack_after_33_minutes_l739_73949

-- Definitions from conditions
def distance_initial : ℝ := 30  -- 30 km apart initially
def combined_speed : ℝ := 2     -- combined speed is 2 km/min
def time_before_breakdown : ℝ := 6 -- Jack biked for 6 minutes before breaking down

-- Assume speeds
def v_J (v_E : ℝ) : ℝ := 2 * v_E  -- Jack's speed is twice Emma's speed

-- Assertion to prove
theorem emma_reaches_jack_after_33_minutes :
  ∀ v_E : ℝ, ((v_J v_E + v_E = combined_speed) → 
              (distance_initial - combined_speed * time_before_breakdown = 18) → 
              (v_E > 0) → 
              (time_before_breakdown + 18 / v_E = 33)) :=
by 
  intro v_E 
  intros h1 h2 h3 
  have h4 : v_J v_E = 2 * v_E := rfl
  sorry

end NUMINAMATH_GPT_emma_reaches_jack_after_33_minutes_l739_73949


namespace NUMINAMATH_GPT_factor_b_value_l739_73931

theorem factor_b_value (a b : ℤ) (h : ∀ x : ℂ, (x^2 - x - 1) ∣ (a*x^3 + b*x^2 + 1)) : b = -2 := 
sorry

end NUMINAMATH_GPT_factor_b_value_l739_73931


namespace NUMINAMATH_GPT_polynomial_multiplication_equiv_l739_73924

theorem polynomial_multiplication_equiv (x : ℝ) : 
  (x^4 + 50*x^2 + 625)*(x^2 - 25) = x^6 - 75*x^4 + 1875*x^2 - 15625 := 
by 
  sorry

end NUMINAMATH_GPT_polynomial_multiplication_equiv_l739_73924


namespace NUMINAMATH_GPT_harry_total_travel_time_l739_73974

def bus_time_already_sitting : Nat := 15
def bus_time_remaining : Nat := 25
def walk_fraction := 1 / 2

def bus_time_total : Nat := bus_time_already_sitting + bus_time_remaining
def walk_time : Nat := bus_time_total * walk_fraction

theorem harry_total_travel_time : bus_time_total + walk_time = 60 := by
  sorry

end NUMINAMATH_GPT_harry_total_travel_time_l739_73974


namespace NUMINAMATH_GPT_inequality_for_positive_reals_l739_73909

theorem inequality_for_positive_reals 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 / (a^3 + b^3 + a * b * c)) + (1 / (b^3 + c^3 + a * b * c)) + 
  (1 / (c^3 + a^3 + a * b * c)) ≤ 1 / (a * b * c) := 
sorry

end NUMINAMATH_GPT_inequality_for_positive_reals_l739_73909


namespace NUMINAMATH_GPT_determine_d_iff_l739_73975

theorem determine_d_iff (x : ℝ) : 
  (x ∈ Set.Ioo (-5/2) 3) ↔ (x * (2 * x + 3) < 15) :=
by
  sorry

end NUMINAMATH_GPT_determine_d_iff_l739_73975


namespace NUMINAMATH_GPT_algebra_1_algebra_2_l739_73914

variable (x1 x2 : ℝ)
variable (h_root1 : x1^2 - 2*x1 - 1 = 0)
variable (h_root2 : x2^2 - 2*x2 - 1 = 0)
variable (h_sum : x1 + x2 = 2)
variable (h_prod : x1 * x2 = -1)

theorem algebra_1 : (x1 + x2) * (x1 * x2) = -2 := by
  -- Proof here
  sorry

theorem algebra_2 : (x1 - x2)^2 = 8 := by
  -- Proof here
  sorry

end NUMINAMATH_GPT_algebra_1_algebra_2_l739_73914


namespace NUMINAMATH_GPT_negation_proposition_l739_73947

variable (a : ℝ)

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * a * x + a > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proposition_l739_73947


namespace NUMINAMATH_GPT_quadratic_real_roots_l739_73991

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_real_roots (k : ℝ) :
  discriminant (k - 1) 4 2 ≥ 0 ↔ k ≤ 3 ∧ k ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l739_73991


namespace NUMINAMATH_GPT_rihanna_money_left_l739_73939

-- Definitions of the item costs
def cost_of_mangoes : ℝ := 6 * 3
def cost_of_apple_juice : ℝ := 4 * 3.50
def cost_of_potato_chips : ℝ := 2 * 2.25
def cost_of_chocolate_bars : ℝ := 3 * 1.75

-- Total cost computation
def total_cost : ℝ := cost_of_mangoes + cost_of_apple_juice + cost_of_potato_chips + cost_of_chocolate_bars

-- Initial amount of money Rihanna has
def initial_money : ℝ := 50

-- Remaining money after the purchases
def remaining_money : ℝ := initial_money - total_cost

-- The theorem stating that the remaining money is $8.25
theorem rihanna_money_left : remaining_money = 8.25 := by
  -- Lean will require the proof here.
  sorry

end NUMINAMATH_GPT_rihanna_money_left_l739_73939


namespace NUMINAMATH_GPT_tetrahedron_sum_eq_14_l739_73918

theorem tetrahedron_sum_eq_14 :
  let edges := 6
  let corners := 4
  let faces := 4
  edges + corners + faces = 14 :=
by
  let edges := 6
  let corners := 4
  let faces := 4
  show edges + corners + faces = 14
  sorry

end NUMINAMATH_GPT_tetrahedron_sum_eq_14_l739_73918


namespace NUMINAMATH_GPT_product_modulo_7_l739_73960

theorem product_modulo_7 : 
  (2007 % 7 = 4) ∧ (2008 % 7 = 5) ∧ (2009 % 7 = 6) ∧ (2010 % 7 = 0) →
  (2007 * 2008 * 2009 * 2010) % 7 = 0 :=
by
  intros h
  rcases h with ⟨h1, h2, h3, h4⟩
  sorry

end NUMINAMATH_GPT_product_modulo_7_l739_73960


namespace NUMINAMATH_GPT_central_angle_of_sector_l739_73950

theorem central_angle_of_sector
  (r : ℝ) (S_sector : ℝ) (alpha : ℝ) (h₁ : r = 2) (h₂ : S_sector = (2 / 5) * Real.pi)
  (h₃ : S_sector = (1 / 2) * alpha * r^2) : alpha = Real.pi / 5 :=
by
  sorry

end NUMINAMATH_GPT_central_angle_of_sector_l739_73950


namespace NUMINAMATH_GPT_number_of_terminating_decimals_l739_73913

theorem number_of_terminating_decimals (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 150) :
  ∃ m, m = 50 ∧ 
  ∀ n, (1 ≤ n ∧ n ≤ 150) → (∃ k, n = 3 * k) →
  m = 50 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_terminating_decimals_l739_73913


namespace NUMINAMATH_GPT_extreme_values_range_of_a_inequality_of_zeros_l739_73978

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := -2 * (Real.log x) - a / (x ^ 2) + 1

theorem extreme_values (a : ℝ) (h : a = 1) :
  (∀ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a ≤ 0) ∧
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a = 0) ∧
  (∀ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a ≥ -3 + 2 * (Real.log 2)) ∧
  (∃ x ∈ Set.Icc (1 / 2 : ℝ) 2, f x a = -3 + 2 * (Real.log 2)) :=
sorry

theorem range_of_a :
  (∀ a : ℝ, (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ 0 < a ∧ a < 1) :=
sorry

theorem inequality_of_zeros (a : ℝ) (h : 0 < a) (h1 : a < 1) (x1 x2 : ℝ) (hx1 : f x1 a = 0) (hx2 : f x2 a = 0) (hx1x2 : x1 ≠ x2) :
  1 / (x1 ^ 2) + 1 / (x2 ^ 2) > 2 / a :=
sorry

end NUMINAMATH_GPT_extreme_values_range_of_a_inequality_of_zeros_l739_73978


namespace NUMINAMATH_GPT_fewest_students_possible_l739_73999

theorem fewest_students_possible :
  ∃ n : ℕ, n ≡ 2 [MOD 5] ∧ n ≡ 4 [MOD 6] ∧ n ≡ 6 [MOD 8] ∧ n = 22 :=
sorry

end NUMINAMATH_GPT_fewest_students_possible_l739_73999


namespace NUMINAMATH_GPT_number_of_blocks_needed_l739_73980

-- Define the dimensions of the fort
def fort_length : ℕ := 20
def fort_width : ℕ := 15
def fort_height : ℕ := 8

-- Define the thickness of the walls and the floor
def wall_thickness : ℕ := 2
def floor_thickness : ℕ := 1

-- Define the original volume of the fort
def V_original : ℕ := fort_length * fort_width * fort_height

-- Define the interior dimensions of the fort considering the thickness of the walls and floor
def interior_length : ℕ := fort_length - 2 * wall_thickness
def interior_width : ℕ := fort_width - 2 * wall_thickness
def interior_height : ℕ := fort_height - floor_thickness

-- Define the volume of the interior space
def V_interior : ℕ := interior_length * interior_width * interior_height

-- Statement to prove: number of blocks needed equals 1168
theorem number_of_blocks_needed : V_original - V_interior = 1168 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_blocks_needed_l739_73980


namespace NUMINAMATH_GPT_tagged_fish_in_second_catch_l739_73981

theorem tagged_fish_in_second_catch :
  ∀ (T : ℕ),
    (40 > 0) →
    (800 > 0) →
    (T / 40 = 40 / 800) →
    T = 2 := 
by
  intros T h1 h2 h3
  sorry

end NUMINAMATH_GPT_tagged_fish_in_second_catch_l739_73981


namespace NUMINAMATH_GPT_rate_of_descent_correct_l739_73930

def depth := 3500 -- in feet
def time := 100 -- in minutes

def rate_of_descent : ℕ := depth / time

theorem rate_of_descent_correct : rate_of_descent = 35 := by
  -- We intentionally skip the proof part as per the requirement
  sorry

end NUMINAMATH_GPT_rate_of_descent_correct_l739_73930


namespace NUMINAMATH_GPT_deg_to_rad_neg_630_l739_73910

theorem deg_to_rad_neg_630 :
  (-630 : ℝ) * (Real.pi / 180) = -7 * Real.pi / 2 := by
  sorry

end NUMINAMATH_GPT_deg_to_rad_neg_630_l739_73910


namespace NUMINAMATH_GPT_average_age_of_team_l739_73962

theorem average_age_of_team 
  (n : ℕ) 
  (captain_age : ℕ) 
  (wicket_keeper_age : ℕ) 
  (remaining_avg : ℕ → ℕ) 
  (h1 : n = 11)
  (h2 : captain_age = 27)
  (h3 : wicket_keeper_age = 28)
  (h4 : ∀ A, remaining_avg A = A - 1)
  (h5 : ∀ A, 11 * A = 9 * (remaining_avg A) + captain_age + wicket_keeper_age) : 
  ∃ A, A = 32 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_team_l739_73962


namespace NUMINAMATH_GPT_zilla_savings_l739_73903

theorem zilla_savings (earnings : ℝ) (rent : ℝ) (expenses : ℝ) (savings : ℝ) 
  (h1 : rent = 0.07 * earnings)
  (h2 : rent = 133)
  (h3 : expenses = earnings / 2)
  (h4 : savings = earnings - rent - expenses) :
  savings = 817 := 
sorry

end NUMINAMATH_GPT_zilla_savings_l739_73903


namespace NUMINAMATH_GPT_eden_stuffed_bears_l739_73934

theorem eden_stuffed_bears 
  (initial_bears : ℕ) 
  (percentage_kept : ℝ) 
  (sisters : ℕ) 
  (eden_initial_bears : ℕ)
  (h1 : initial_bears = 65) 
  (h2 : percentage_kept = 0.40) 
  (h3 : sisters = 4) 
  (h4 : eden_initial_bears = 20) :
  ∃ eden_bears : ℕ, eden_bears = 29 :=
by
  sorry

end NUMINAMATH_GPT_eden_stuffed_bears_l739_73934


namespace NUMINAMATH_GPT_area_isosceles_right_triangle_l739_73988

theorem area_isosceles_right_triangle 
( a : ℝ × ℝ )
( b : ℝ × ℝ )
( h_a : a = (Real.cos (2 / 3 * Real.pi), Real.sin (2 / 3 * Real.pi)) )
( is_isosceles_right_triangle : (a + b).fst * (a - b).fst + (a + b).snd * (a - b).snd = 0 
                                ∧ (a + b).fst * (a + b).fst + (a + b).snd * (a + b).snd 
                                = (a - b).fst * (a - b).fst + (a - b).snd * (a - b).snd ):
  1 / 2 * Real.sqrt ((1 - 1 / 2)^2 + (Real.sqrt 3 / 2 - -1 / 2)^2 )
 * Real.sqrt ((1 - -1 / 2)^2 + (Real.sqrt 3 / 2 - -1 / 2 )^2 ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_area_isosceles_right_triangle_l739_73988


namespace NUMINAMATH_GPT_percentage_of_bottle_danny_drank_l739_73940

theorem percentage_of_bottle_danny_drank
    (x : ℝ)  -- percentage of the first bottle Danny drinks, represented as a real number
    (b1 b2 b3 : ℝ)  -- volumes of the three bottles, represented as real numbers
    (h_b1 : b1 = 1)  -- first bottle is full (1 bottle)
    (h_b2 : b2 = 1)  -- second bottle is full (1 bottle)
    (h_b3 : b3 = 1)  -- third bottle is full (1 bottle)
    (h_given_away1 : b2 * 0.7 = 0.7)  -- gave away 70% of the second bottle
    (h_given_away2 : b3 * 0.7 = 0.7)  -- gave away 70% of the third bottle
    (h_soda_left : b1 * (1 - x) + b2 * 0.3 + b3 * 0.3 = 0.7)  -- 70% of bottle left
    : x = 0.9 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_bottle_danny_drank_l739_73940


namespace NUMINAMATH_GPT_triangle_ABC_is_acute_l739_73938

theorem triangle_ABC_is_acute (A B C : ℝ) (a b c : ℝ) 
  (h1: a^2 + b^2 >= c^2) (h2: b^2 + c^2 >= a^2) (h3: c^2 + a^2 >= b^2)
  (h4: (Real.sin A + Real.sin B) / (Real.sin B + Real.sin C) = 9 / 11)
  (h5: (Real.sin B + Real.sin C) / (Real.sin C + Real.sin A) = 11 / 10) : 
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2 :=
sorry

end NUMINAMATH_GPT_triangle_ABC_is_acute_l739_73938


namespace NUMINAMATH_GPT_fraction_finding_l739_73969

theorem fraction_finding (x : ℝ) (h : (3 / 4) * x * (2 / 3) = 0.4) : x = 0.8 :=
sorry

end NUMINAMATH_GPT_fraction_finding_l739_73969


namespace NUMINAMATH_GPT_ball_hits_ground_at_t_l739_73917

theorem ball_hits_ground_at_t (t : ℝ) : 
  (∃ t, -8 * t^2 - 12 * t + 64 = 0 ∧ 0 ≤ t) → t = 2 :=
by
  sorry

end NUMINAMATH_GPT_ball_hits_ground_at_t_l739_73917


namespace NUMINAMATH_GPT_loss_percentage_initially_l739_73997

theorem loss_percentage_initially 
  (SP : ℝ) 
  (CP : ℝ := 400) 
  (h1 : SP + 100 = 1.05 * CP) : 
  (1 - SP / CP) * 100 = 20 := 
by 
  sorry

end NUMINAMATH_GPT_loss_percentage_initially_l739_73997


namespace NUMINAMATH_GPT_rectangle_maximized_area_side_length_l739_73971

theorem rectangle_maximized_area_side_length
  (x y : ℝ)
  (h_perimeter : 2 * x + 2 * y = 40)
  (h_max_area : x * y = 100) :
  x = 10 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_maximized_area_side_length_l739_73971


namespace NUMINAMATH_GPT_ab_cd_divisible_eq_one_l739_73959

theorem ab_cd_divisible_eq_one (a b c d : ℕ) (h1 : ∃ e : ℕ, e = ab - cd ∧ (e ∣ a) ∧ (e ∣ b) ∧ (e ∣ c) ∧ (e ∣ d)) : ab - cd = 1 :=
sorry

end NUMINAMATH_GPT_ab_cd_divisible_eq_one_l739_73959


namespace NUMINAMATH_GPT_polynomial_evaluation_l739_73920

def polynomial_at (x : ℝ) : ℝ :=
  let f := (7 : ℝ) * x^5 + 12 * x^4 - 5 * x^3 - 6 * x^2 + 3 * x - 5
  f

theorem polynomial_evaluation : polynomial_at 3 = 2488 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l739_73920


namespace NUMINAMATH_GPT_find_principal_l739_73945

noncomputable def principal_amount (P : ℝ) (r : ℝ) : Prop :=
  (800 = (P * r * 2) / 100) ∧ (820 = P * (1 + r / 100)^2 - P)

theorem find_principal (P : ℝ) (r : ℝ) (h : principal_amount P r) : P = 8000 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_l739_73945


namespace NUMINAMATH_GPT_claudia_total_earnings_l739_73977

-- Definition of the problem conditions
def class_fee : ℕ := 10
def kids_saturday : ℕ := 20
def kids_sunday : ℕ := kids_saturday / 2

-- Theorem stating that Claudia makes $300.00 for the weekend
theorem claudia_total_earnings : (kids_saturday * class_fee) + (kids_sunday * class_fee) = 300 := 
by
  sorry

end NUMINAMATH_GPT_claudia_total_earnings_l739_73977


namespace NUMINAMATH_GPT_days_worked_together_l739_73967

theorem days_worked_together (W : ℝ) (h1 : ∀ (a b : ℝ), (a + b) * 40 = W) 
                             (h2 : ∀ a, a * 16 = W) 
                             (x : ℝ) 
                             (h3 : (x * (W / 40) + 12 * (W / 16)) = W) : 
                             x = 10 := 
by
  sorry

end NUMINAMATH_GPT_days_worked_together_l739_73967


namespace NUMINAMATH_GPT_remainder_is_five_l739_73966

theorem remainder_is_five (A : ℕ) (h : 17 = 6 * 2 + A) : A = 5 :=
sorry

end NUMINAMATH_GPT_remainder_is_five_l739_73966


namespace NUMINAMATH_GPT_percent_decrease_apr_to_may_l739_73911

theorem percent_decrease_apr_to_may (P : ℝ) 
  (h1 : ∀ P : ℝ, P > 0 → (1.35 * P = P + 0.35 * P))
  (h2 : ∀ x : ℝ, P * (1.35 * (1 - x / 100) * 1.5) = 1.62000000000000014 * P)
  (h3 : 0 < x ∧ x < 100)
  : x = 20 :=
  sorry

end NUMINAMATH_GPT_percent_decrease_apr_to_may_l739_73911


namespace NUMINAMATH_GPT_average_infection_per_round_l739_73954

theorem average_infection_per_round (x : ℝ) (h1 : 1 + x + x * (1 + x) = 100) : x = 9 :=
sorry

end NUMINAMATH_GPT_average_infection_per_round_l739_73954


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l739_73989

theorem simplify_and_evaluate_expression (a : ℕ) (h : a = 2023) :
  a * (1 - 2 * a) + 2 * (a + 1) * (a - 1) = 2021 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l739_73989


namespace NUMINAMATH_GPT_combination_30_2_l739_73984

theorem combination_30_2 : Nat.choose 30 2 = 435 := by
  sorry

end NUMINAMATH_GPT_combination_30_2_l739_73984


namespace NUMINAMATH_GPT_S_5_equals_31_l739_73955

-- Define the sequence sum function S
def S (n : Nat) : Nat := 2^n - 1

-- The theorem to prove that S(5) = 31
theorem S_5_equals_31 : S 5 = 31 :=
by
  rw [S]
  sorry

end NUMINAMATH_GPT_S_5_equals_31_l739_73955


namespace NUMINAMATH_GPT_number_of_correct_conclusions_l739_73906

-- Define the conditions given in the problem
def conclusion1 (x : ℝ) : Prop := x > 0 → x > Real.sin x
def conclusion2 (x : ℝ) : Prop := (x - Real.sin x = 0 → x = 0) → (x ≠ 0 → x - Real.sin x ≠ 0)
def conclusion3 (p q : Prop) : Prop := (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)
def conclusion4 : Prop := ¬(∀ x : ℝ, x - Real.log x > 0) = ∃ x : ℝ, x - Real.log x ≤ 0

-- Prove the number of correct conclusions is 3
theorem number_of_correct_conclusions : 
  (∃ x1 : ℝ, conclusion1 x1) ∧
  (∃ x1 : ℝ, conclusion2 x1) ∧
  (∃ p q : Prop, conclusion3 p q) ∧
  ¬conclusion4 →
  3 = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_correct_conclusions_l739_73906


namespace NUMINAMATH_GPT_sum_and_product_of_radical_l739_73944

theorem sum_and_product_of_radical (a b : ℝ) (h1 : 2 * a = -4) (h2 : a^2 - b = 1) :
  a + b = 1 :=
sorry

end NUMINAMATH_GPT_sum_and_product_of_radical_l739_73944


namespace NUMINAMATH_GPT_Ingrid_cookie_percentage_l739_73919

theorem Ingrid_cookie_percentage : 
  let irin_ratio := 9.18
  let ingrid_ratio := 5.17
  let nell_ratio := 2.05
  let kim_ratio := 3.45
  let linda_ratio := 4.56
  let total_cookies := 800
  let total_ratio := irin_ratio + ingrid_ratio + nell_ratio + kim_ratio + linda_ratio
  let ingrid_share := ingrid_ratio / total_ratio
  let ingrid_cookies := ingrid_share * total_cookies
  let ingrid_percentage := (ingrid_cookies / total_cookies) * 100
  ingrid_percentage = 21.25 :=
by
  sorry

end NUMINAMATH_GPT_Ingrid_cookie_percentage_l739_73919
