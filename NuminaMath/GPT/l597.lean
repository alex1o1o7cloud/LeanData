import Mathlib

namespace sequence_a7_l597_59728

theorem sequence_a7 (a b : ℕ) (h1 : a1 = a) (h2 : a2 = b) {a3 a4 a5 a6 a7 : ℕ}
  (h3 : a_3 = a + b)
  (h4 : a_4 = a + 2 * b)
  (h5 : a_5 = 2 * a + 3 * b)
  (h6 : a_6 = 3 * a + 5 * b)
  (h_a6 : a_6 = 50) :
  a_7 = 5 * a + 8 * b :=
by
  sorry

end sequence_a7_l597_59728


namespace min_max_sums_l597_59735

theorem min_max_sums (a b c d e f g : ℝ) 
    (h0 : 0 ≤ a) (h1 : 0 ≤ b) (h2 : 0 ≤ c)
    (h3 : 0 ≤ d) (h4 : 0 ≤ e) (h5 : 0 ≤ f) 
    (h6 : 0 ≤ g) (h_sum : a + b + c + d + e + f + g = 1) :
    (min (max (a + b + c) 
              (max (b + c + d) 
                   (max (c + d + e) 
                        (max (d + e + f) 
                             (e + f + g))))) = 1 / 3) :=
sorry

end min_max_sums_l597_59735


namespace water_speed_l597_59792

theorem water_speed (v : ℝ) 
  (still_water_speed : ℝ := 4)
  (distance : ℝ := 10)
  (time : ℝ := 5)
  (effective_speed : ℝ := distance / time) 
  (h : still_water_speed - v = effective_speed) :
  v = 2 :=
by
  sorry

end water_speed_l597_59792


namespace two_digit_numbers_l597_59744

def is_digit (n : ℕ) : Prop := n ≤ 9

theorem two_digit_numbers (a b : ℕ) (h1 : is_digit a) (h2 : is_digit b) 
  (h3 : a ≠ b) (h4 : (a + b) = 11) : 
  (∃ n m : ℕ, (n = 10 * a + b) ∧ (m = 10 * b + a) ∧ (∃ k : ℕ, (10 * a + b)^2 - (10 * b + a)^2 = k^2)) := 
sorry

end two_digit_numbers_l597_59744


namespace find_t_from_tan_conditions_l597_59795

theorem find_t_from_tan_conditions 
  (α t : ℝ)
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + Real.pi / 4) = 4 / t)
  (h3 : Real.tan (α + Real.pi / 4) = (Real.tan (Real.pi / 4) + Real.tan α) / (1 - Real.tan (Real.pi / 4) * Real.tan α)) :
  t = 2 := 
  by
  sorry

end find_t_from_tan_conditions_l597_59795


namespace find_common_ratio_l597_59781

variable {a : ℕ → ℝ}
variable (q : ℝ)

-- Definition of geometric sequence condition
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Given conditions
def conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 2 + a 4 = 20) ∧ (a 3 + a 5 = 40)

-- Proposition to be proved
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) (h_geo : is_geometric_sequence a q) (h_cond : conditions a q) : q = 2 :=
by 
  sorry

end find_common_ratio_l597_59781


namespace david_more_pushups_l597_59712

theorem david_more_pushups (d z : ℕ) (h1 : d = 51) (h2 : d + z = 53) : d - z = 49 := by
  sorry

end david_more_pushups_l597_59712


namespace can_cross_all_rivers_and_extra_material_l597_59703

-- Definitions for river widths, bridge length, and additional material.
def river1_width : ℕ := 487
def river2_width : ℕ := 621
def river3_width : ℕ := 376
def bridge_length : ℕ := 295
def additional_material : ℕ := 1020

-- Calculations for material needed for each river.
def material_needed_for_river1 : ℕ := river1_width - bridge_length
def material_needed_for_river2 : ℕ := river2_width - bridge_length
def material_needed_for_river3 : ℕ := river3_width - bridge_length

-- Total material needed to cross all three rivers.
def total_material_needed : ℕ := material_needed_for_river1 + material_needed_for_river2 + material_needed_for_river3

-- The main theorem statement to prove.
theorem can_cross_all_rivers_and_extra_material :
  total_material_needed <= additional_material ∧ (additional_material - total_material_needed = 421) := 
by 
  sorry

end can_cross_all_rivers_and_extra_material_l597_59703


namespace square_area_increase_l597_59790

theorem square_area_increase (s : ℝ) (h : s > 0) :
  ((1.15 * s) ^ 2 - s ^ 2) / s ^ 2 * 100 = 32.25 :=
by
  sorry

end square_area_increase_l597_59790


namespace integer_roots_k_values_y1_y2_squared_sum_k_0_y1_y2_squared_sum_k_neg1_l597_59727

theorem integer_roots_k_values (k : ℤ) :
  (∀ x : ℤ, k * x ^ 2 + (2 * k - 1) * x + k - 1 = 0) →
  k = 0 ∨ k = -1 :=
sorry

theorem y1_y2_squared_sum_k_0 (m y1 y2: ℝ) :
  (m > -2) →
  (k = 0) →
  ((k - 1) * y1 ^ 2 - 3 * y1 + m = 0) →
  ((k - 1) * y2 ^ 2 - 3 * y2 + m = 0) →
  y1^2 + y2^2 = 9 + 2 * m :=
sorry

theorem y1_y2_squared_sum_k_neg1 (m y1 y2: ℝ) :
  (m > -2) →
  (k = -1) →
  ((k - 1) * y1 ^ 2 - 3 * y1 + m = 0) →
  ((k - 1) * y2 ^ 2 - 3 * y2 + m = 0) →
  y1^2 + y2^2 = 9 / 4 + m :=
sorry

end integer_roots_k_values_y1_y2_squared_sum_k_0_y1_y2_squared_sum_k_neg1_l597_59727


namespace complement_P_l597_59774

def U : Set ℝ := Set.univ
def P : Set ℝ := {x | x^2 ≤ 1}

theorem complement_P :
  (U \ P) = Set.Iio (-1) ∪ Set.Ioi (1) :=
by
  sorry

end complement_P_l597_59774


namespace find_total_quantities_l597_59765

theorem find_total_quantities (n S S_3 S_2 : ℕ) (h1 : S = 8 * n) (h2 : S_3 = 4 * 3) (h3 : S_2 = 14 * 2) (h4 : S = S_3 + S_2) : n = 5 :=
by
  sorry

end find_total_quantities_l597_59765


namespace find_five_digit_number_l597_59759

theorem find_five_digit_number (x : ℕ) (hx : 10000 ≤ x ∧ x < 100000)
  (h : 10 * x + 1 = 3 * (100000 + x) ∨ 3 * (10 * x + 1) = 100000 + x) :
  x = 42857 :=
sorry

end find_five_digit_number_l597_59759


namespace proposition_holds_n_2019_l597_59716

theorem proposition_holds_n_2019 (P: ℕ → Prop) 
  (H1: ∀ k : ℕ, k > 0 → ¬ P (k + 1) → ¬ P k) 
  (H2: P 2018) : 
  P 2019 :=
by 
  sorry

end proposition_holds_n_2019_l597_59716


namespace find_three_numbers_l597_59773

theorem find_three_numbers (x : ℤ) (a b c : ℤ) :
  a + b + c = (x + 1)^2 ∧ a + b = x^2 ∧ b + c = (x - 1)^2 ∧
  a = 80 ∧ b = 320 ∧ c = 41 :=
by {
  sorry
}

end find_three_numbers_l597_59773


namespace fraction_is_five_over_nine_l597_59702

theorem fraction_is_five_over_nine (f k t : ℝ) (h1 : t = f * (k - 32)) (h2 : t = 50) (h3 : k = 122) : f = 5 / 9 :=
by
  sorry

end fraction_is_five_over_nine_l597_59702


namespace complex_division_l597_59767

def i : ℂ := Complex.I

theorem complex_division :
  (i^3 / (1 + i)) = -1/2 - 1/2 * i := 
by sorry

end complex_division_l597_59767


namespace prove_2x_plus_y_le_sqrt_11_l597_59796

variable (x y : ℝ)
variable (h : 3 * x^2 + 2 * y^2 ≤ 6)

theorem prove_2x_plus_y_le_sqrt_11 : 2 * x + y ≤ Real.sqrt 11 := by
  sorry

end prove_2x_plus_y_le_sqrt_11_l597_59796


namespace probability_of_rolling_5_is_1_over_9_l597_59714

def num_sides_dice : ℕ := 6

def favorable_combinations : List (ℕ × ℕ) :=
[(1, 4), (2, 3), (3, 2), (4, 1)]

def total_combinations : ℕ :=
num_sides_dice * num_sides_dice

def favorable_count : ℕ := favorable_combinations.length

def probability_rolling_5 : ℚ :=
favorable_count / total_combinations

theorem probability_of_rolling_5_is_1_over_9 :
  probability_rolling_5 = 1 / 9 :=
sorry

end probability_of_rolling_5_is_1_over_9_l597_59714


namespace B_values_for_divisibility_l597_59777

theorem B_values_for_divisibility (B : ℕ) (h : 4 + B + B + B + 2 ≡ 0 [MOD 9]) : B = 1 ∨ B = 4 ∨ B = 7 :=
by sorry

end B_values_for_divisibility_l597_59777


namespace solve_equation_l597_59725

-- Define the equation to be solved
def equation (x : ℝ) : Prop := (x + 2)^4 + (x - 4)^4 = 272

-- State the theorem we want to prove
theorem solve_equation : ∃ x : ℝ, equation x :=
  sorry

end solve_equation_l597_59725


namespace initial_workers_l597_59708

/--
In a factory, some workers were employed, and then 25% more workers have just been hired.
There are now 1065 employees in the factory. Prove that the number of workers initially employed is 852.
-/
theorem initial_workers (x : ℝ) (h1 : x + 0.25 * x = 1065) : x = 852 :=
sorry

end initial_workers_l597_59708


namespace exists_pair_sum_ends_with_last_digit_l597_59705

theorem exists_pair_sum_ends_with_last_digit (a : ℕ → ℕ) (h_distinct: ∀ i j, (i ≠ j) → a i ≠ a j) (h_range: ∀ i, a i < 10) : ∀ (n : ℕ), n < 10 → ∃ i j, (i ≠ j) ∧ (a i + a j) % 10 = n % 10 :=
by sorry

end exists_pair_sum_ends_with_last_digit_l597_59705


namespace general_term_of_sequence_l597_59756

theorem general_term_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hSn : ∀ n, S n = 3 * n^2 - n + 1) :
  (∀ n, a n = if n = 1 then 3 else 6 * n - 4) :=
by
  sorry

end general_term_of_sequence_l597_59756


namespace evaluate_polynomial_l597_59766

noncomputable def polynomial_evaluation : Prop :=
∀ (x : ℝ), x^2 - 3*x - 9 = 0 ∧ 0 < x → (x^4 - 3*x^3 - 9*x^2 + 27*x - 8) = (65 + 81*(Real.sqrt 5))/2

theorem evaluate_polynomial : polynomial_evaluation :=
sorry

end evaluate_polynomial_l597_59766


namespace cubed_identity_l597_59745

theorem cubed_identity (x : ℝ) (h : x + 1/x = -7) : x^3 + 1/x^3 = -322 := 
by 
  sorry

end cubed_identity_l597_59745


namespace inequality_solution_l597_59713

theorem inequality_solution (x : ℝ) : 3 * x^2 - x > 9 ↔ x < -3 ∨ x > 1 := by
  sorry

end inequality_solution_l597_59713


namespace smaller_number_between_5_and_8_l597_59746

theorem smaller_number_between_5_and_8 :
  min 5 8 = 5 :=
by
  sorry

end smaller_number_between_5_and_8_l597_59746


namespace quadratic_range_l597_59775

theorem quadratic_range (x y : ℝ) 
    (h1 : y = (x - 1)^2 + 1)
    (h2 : 2 ≤ y ∧ y < 5) : 
    (-1 < x ∧ x ≤ 0) ∨ (2 ≤ x ∧ x < 3) :=
by
  sorry

end quadratic_range_l597_59775


namespace unique_xy_exists_l597_59726

theorem unique_xy_exists (n : ℕ) : 
  ∃! (x y : ℕ), n = ((x + y) ^ 2 + 3 * x + y) / 2 := 
sorry

end unique_xy_exists_l597_59726


namespace log_positive_interval_l597_59749

noncomputable def f (a x : ℝ) : ℝ := Real.log (2 * x - a) / Real.log a

theorem log_positive_interval (a : ℝ) :
  (∀ x, x ∈ Set.Icc (1 / 2) (2 / 3) → f a x > 0) ↔ (1 / 3 < a ∧ a < 1) := by
  sorry

end log_positive_interval_l597_59749


namespace noah_ate_burgers_l597_59788

theorem noah_ate_burgers :
  ∀ (weight_hotdog weight_burger weight_pie : ℕ) 
    (mason_hotdog_weight : ℕ) 
    (jacob_pies noah_burgers mason_hotdogs : ℕ),
    weight_hotdog = 2 →
    weight_burger = 5 →
    weight_pie = 10 →
    (jacob_pies + 3 = noah_burgers) →
    (mason_hotdogs = 3 * jacob_pies) →
    (mason_hotdog_weight = 30) →
    (mason_hotdog_weight / weight_hotdog = mason_hotdogs) →
    noah_burgers = 8 :=
by
  intros weight_hotdog weight_burger weight_pie mason_hotdog_weight
         jacob_pies noah_burgers mason_hotdogs
         h1 h2 h3 h4 h5 h6 h7
  sorry

end noah_ate_burgers_l597_59788


namespace corey_gave_more_books_l597_59717

def books_given_by_mike : ℕ := 10
def total_books_received_by_lily : ℕ := 35
def books_given_by_corey : ℕ := total_books_received_by_lily - books_given_by_mike
def difference_in_books (a b : ℕ) : ℕ := a - b

theorem corey_gave_more_books :
  difference_in_books books_given_by_corey books_given_by_mike = 15 := by
sorry

end corey_gave_more_books_l597_59717


namespace orlando_weight_gain_l597_59724

def weight_gain_statement (x J F : ℝ) : Prop :=
  J = 2 * x + 2 ∧ F = 1/2 * J - 3 ∧ x + J + F = 20

theorem orlando_weight_gain :
  ∃ x J F : ℝ, weight_gain_statement x J F ∧ x = 5 :=
by {
  sorry
}

end orlando_weight_gain_l597_59724


namespace sum_lengths_AMC_l597_59711

theorem sum_lengths_AMC : 
  let length_A := 2 * (Real.sqrt 2) + 2
  let length_M := 3 + 3 + 2 * (Real.sqrt 2)
  let length_C := 3 + 3 + 2
  length_A + length_M + length_C = 13 + 4 * (Real.sqrt 2)
  := by
  sorry

end sum_lengths_AMC_l597_59711


namespace distinct_connected_stamps_l597_59776

theorem distinct_connected_stamps (n : ℕ) : 
  ∃ d : ℕ → ℝ, 
    d (n+1) = 1 / 4 * (1 + Real.sqrt 2)^(n + 3) + 1 / 4 * (1 - Real.sqrt 2)^(n + 3) - 2 * n - 7 / 2 :=
sorry

end distinct_connected_stamps_l597_59776


namespace find_eagle_feathers_times_l597_59780

theorem find_eagle_feathers_times (x : ℕ) (hawk_feathers : ℕ) (total_feathers_before_give : ℕ) (total_feathers : ℕ) (left_after_selling : ℕ) :
  hawk_feathers = 6 →
  total_feathers_before_give = 6 + 6 * x →
  total_feathers = total_feathers_before_give - 10 →
  left_after_selling = total_feathers / 2 →
  left_after_selling = 49 →
  x = 17 :=
by
  intros h_hawk h_total_before_give h_total h_left h_after_selling
  sorry

end find_eagle_feathers_times_l597_59780


namespace quadratic_has_two_distinct_real_roots_l597_59787

-- Given the discriminant condition Δ = b^2 - 4ac > 0
theorem quadratic_has_two_distinct_real_roots (a b c : ℝ) (h : b^2 - 4 * a * c > 0) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) := 
  sorry

end quadratic_has_two_distinct_real_roots_l597_59787


namespace factorize_difference_of_squares_l597_59798

theorem factorize_difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) :=
sorry

end factorize_difference_of_squares_l597_59798


namespace mountain_height_correct_l597_59793

noncomputable def height_of_mountain : ℝ :=
  15 / (1 / Real.tan (Real.pi * 10 / 180) + 1 / Real.tan (Real.pi * 12 / 180))

theorem mountain_height_correct :
  abs (height_of_mountain - 1.445) < 0.001 :=
sorry

end mountain_height_correct_l597_59793


namespace shaded_rectangle_ratio_l597_59743

variable (a : ℝ) (h : 0 < a)  -- side length of the square is 'a' and it is positive

theorem shaded_rectangle_ratio :
  (∃ l w : ℝ, (l = a / 2 ∧ w = a / 3 ∧ (l * w = a^2 / 6) ∧ (a^2 / 6 = a * a / 6))) → (l / w = 1.5) :=
by {
  -- Proof is to be provided
  sorry
}

end shaded_rectangle_ratio_l597_59743


namespace general_solution_linear_diophantine_l597_59779

theorem general_solution_linear_diophantine (a b c : ℤ) (h_coprime : Int.gcd a b = 1)
    (x1 y1 : ℤ) (h_particular_solution : a * x1 + b * y1 = c) :
    ∃ (t : ℤ), (∃ (x y : ℤ), x = x1 + b * t ∧ y = y1 - a * t ∧ a * x + b * y = c) ∧
               (∃ (x' y' : ℤ), x' = x1 - b * t ∧ y' = y1 + a * t ∧ a * x' + b * y' = c) :=
by
  sorry

end general_solution_linear_diophantine_l597_59779


namespace inequality_solution_set_inequality_range_of_a_l597_59704

theorem inequality_solution_set (a : ℝ) (x : ℝ) (h : a = -8) :
  (|x - 3| + |x + 2| ≤ |a + 1|) ↔ (-3 ≤ x ∧ x ≤ 4) :=
by sorry

theorem inequality_range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - 3| + |x + 2| ≤ |a + 1|) ↔ (a ≤ -6 ∨ a ≥ 4) :=
by sorry

end inequality_solution_set_inequality_range_of_a_l597_59704


namespace parameter_values_for_roots_l597_59754

theorem parameter_values_for_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 = 5 * x2 ∧ a * x1^2 - (2 * a + 5) * x1 + 10 = 0 ∧ a * x2^2 - (2 * a + 5) * x2 + 10 = 0)
  ↔ (a = 5 / 3 ∨ a = 5) := 
sorry

end parameter_values_for_roots_l597_59754


namespace negation_equiv_l597_59710

-- Define the initial proposition
def initial_proposition (x : ℝ) : Prop :=
  x^2 - x + 1 > 0

-- Define the negation of the initial proposition
def negated_proposition : Prop :=
  ∃ x₀ : ℝ, x₀^2 - x₀ + 1 ≤ 0

-- The statement asserting the negation equivalence
theorem negation_equiv :
  (¬ ∀ x : ℝ, initial_proposition x) ↔ negated_proposition :=
by sorry

end negation_equiv_l597_59710


namespace quadratic_roots_l597_59723

variable {a b c : ℝ}

theorem quadratic_roots (h₁ : a > 0) (h₂ : b > 0) (h₃ : c < 0) : 
  ∃ x₁ x₂ : ℝ, (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) ∧ 
  (x₁ ≠ x₂) ∧ (x₁ > 0) ∧ (x₂ < 0) ∧ (|x₂| > |x₁|) := 
sorry

end quadratic_roots_l597_59723


namespace units_digit_of_k_squared_plus_2_k_l597_59764

def k := 2008^2 + 2^2008

theorem units_digit_of_k_squared_plus_2_k : 
  (k^2 + 2^k) % 10 = 7 :=
by {
  -- The proof will be inserted here
  sorry
}

end units_digit_of_k_squared_plus_2_k_l597_59764


namespace find_integer_solutions_l597_59719

theorem find_integer_solutions (x y : ℤ) :
  8 * x^2 * y^2 + x^2 + y^2 = 10 * x * y ↔
  (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) := 
by 
  sorry

end find_integer_solutions_l597_59719


namespace cone_height_ratio_l597_59789

theorem cone_height_ratio (r h : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) 
  (rolls_19_times : 19 * 2 * Real.pi * r = 2 * Real.pi * Real.sqrt (r^2 + h^2)) :
  h / r = 6 * Real.sqrt 10 :=
by
  -- problem setup and mathematical manipulations
  sorry

end cone_height_ratio_l597_59789


namespace simplify_expression_l597_59770

theorem simplify_expression (a : ℝ) (h : a = Real.sqrt 3 - 3) : 
  (a^2 - 4 * a + 4) / (a^2 - 4) / ((a - 2) / (a^2 + 2 * a)) + 3 = Real.sqrt 3 :=
by
  sorry

end simplify_expression_l597_59770


namespace find_natural_numbers_l597_59740

theorem find_natural_numbers (x : ℕ) : (x % 7 = 3) ∧ (x % 9 = 4) ∧ (x < 100) ↔ (x = 31) ∨ (x = 94) := 
by sorry

end find_natural_numbers_l597_59740


namespace cube_plus_eleven_mul_divisible_by_six_l597_59763

theorem cube_plus_eleven_mul_divisible_by_six (a : ℤ) : 6 ∣ (a^3 + 11 * a) := 
by sorry

end cube_plus_eleven_mul_divisible_by_six_l597_59763


namespace equivalent_representations_l597_59748

theorem equivalent_representations :
  (16 / 20 = 24 / 30) ∧
  (80 / 100 = 4 / 5) ∧
  (4 / 5 = 0.8) :=
by 
  sorry

end equivalent_representations_l597_59748


namespace squirrel_spiral_distance_l597_59768

/-- The squirrel runs up a cylindrical post in a perfect spiral path, making one circuit for each rise of 4 feet.
Given the post is 16 feet tall and 3 feet in circumference, the total distance traveled by the squirrel is 20 feet. -/
theorem squirrel_spiral_distance :
  let height : ℝ := 16
  let circumference : ℝ := 3
  let rise_per_circuit : ℝ := 4
  let number_of_circuits := height / rise_per_circuit
  let distance_per_circuit := (circumference^2 + rise_per_circuit^2).sqrt
  number_of_circuits * distance_per_circuit = 20 := by
  sorry

end squirrel_spiral_distance_l597_59768


namespace toy_selling_price_l597_59709

theorem toy_selling_price (x : ℝ) (units_sold : ℝ) (profit_per_day : ℝ) : 
  (units_sold = 200 + 20 * (80 - x)) → 
  (profit_per_day = (x - 60) * units_sold) → 
  profit_per_day = 2500 → 
  x ≤ 60 * 1.4 → 
  x = 65 :=
by
  intros h1 h2 h3 h4
  sorry

end toy_selling_price_l597_59709


namespace geometric_arithmetic_sequence_l597_59753

theorem geometric_arithmetic_sequence (a q : ℝ) 
    (h₁ : a + a * q + a * q ^ 2 = 19) 
    (h₂ : a * (q - 1) = -1) : 
  (a = 4 ∧ q = 1.5) ∨ (a = 9 ∧ q = 2/3) :=
by
  sorry

end geometric_arithmetic_sequence_l597_59753


namespace breadth_of_rectangular_plot_l597_59786

theorem breadth_of_rectangular_plot (b l A : ℕ) (h1 : A = 20 * b) (h2 : l = b + 10) 
    (h3 : A = l * b) : b = 10 := by
  sorry

end breadth_of_rectangular_plot_l597_59786


namespace abs_neg_three_l597_59782

theorem abs_neg_three : abs (-3) = 3 := 
by 
  -- Skipping proof with sorry
  sorry

end abs_neg_three_l597_59782


namespace find_a_l597_59755

-- Define the polynomial f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 - 0.8

-- Define the intermediate values v_0, v_1, and v_2 using Horner's method
def v_0 (a : ℝ) : ℝ := a
def v_1 (a : ℝ) (x : ℝ) : ℝ := v_0 a * x + 2
def v_2 (a : ℝ) (x : ℝ) : ℝ := v_1 a x * x + 3.5 * x - 2.6 * x + 13.5

-- The condition for v_2 when x = 5
axiom v2_value (a : ℝ) : v_2 a 5 = 123.5

-- Prove that a = 4
theorem find_a : ∃ a : ℝ, v_2 a 5 = 123.5 ∧ a = 4 := by
  sorry

end find_a_l597_59755


namespace find_x_such_that_ceil_mul_x_eq_168_l597_59700

theorem find_x_such_that_ceil_mul_x_eq_168 (x : ℝ) (h_pos : x > 0)
  (h_eq : ⌈x⌉ * x = 168) (h_ceil: ⌈x⌉ - 1 < x ∧ x ≤ ⌈x⌉) :
  x = 168 / 13 :=
by
  sorry

end find_x_such_that_ceil_mul_x_eq_168_l597_59700


namespace point_between_circles_l597_59794

theorem point_between_circles 
  (a b c x1 x2 : ℝ)
  (ellipse_eq : ∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c > 0)
  (quad_eq : a * x1^2 + b * x1 - c = 0)
  (quad_eq2 : a * x2^2 + b * x2 - c = 0)
  (sum_roots : x1 + x2 = -b / a)
  (prod_roots : x1 * x2 = -c / a) :
  1 < x1^2 + x2^2 ∧ x1^2 + x2^2 < 2 :=
sorry

end point_between_circles_l597_59794


namespace engineering_student_max_marks_l597_59706

/-- 
If an engineering student has to secure 36% marks to pass, and he gets 130 marks but fails by 14 marks, 
then the maximum number of marks is 400.
-/
theorem engineering_student_max_marks (M : ℝ) (passing_percentage : ℝ) (marks_obtained : ℝ) (marks_failed_by : ℝ) (pass_marks : ℝ) :
  passing_percentage = 0.36 →
  marks_obtained = 130 →
  marks_failed_by = 14 →
  pass_marks = marks_obtained + marks_failed_by →
  pass_marks = passing_percentage * M →
  M = 400 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end engineering_student_max_marks_l597_59706


namespace price_of_third_variety_l597_59741

-- Define the given conditions
def price1 : ℝ := 126
def price2 : ℝ := 135
def average_price : ℝ := 153
def ratio1 : ℝ := 1
def ratio2 : ℝ := 1
def ratio3 : ℝ := 2

-- Define the total ratio
def total_ratio : ℝ := ratio1 + ratio2 + ratio3

-- Define the equation based on the given conditions
def weighted_avg_price (P : ℝ) : Prop :=
  (ratio1 * price1 + ratio2 * price2 + ratio3 * P) / total_ratio = average_price

-- Statement of the proof
theorem price_of_third_variety :
  ∃ P : ℝ, weighted_avg_price P ∧ P = 175.5 :=
by {
  -- Proof omitted
  sorry
}

end price_of_third_variety_l597_59741


namespace find_a_l597_59799

noncomputable def f (x a : ℝ) : ℝ := x + Real.exp (x - a)
noncomputable def g (x a : ℝ) : ℝ := Real.log (x + 2) - 4 * Real.exp (a - x)

theorem find_a (x0 a : ℝ) (h : f x0 a - g x0 a = 3) : a = -1 - Real.log 2 := sorry

end find_a_l597_59799


namespace find_value_of_expression_l597_59721

theorem find_value_of_expression
  (x y z : ℝ)
  (h1 : 3 * x - 4 * y - 2 * z = 0)
  (h2 : x + 2 * y - 7 * z = 0)
  (hz : z ≠ 0) :
  (x^2 - 2 * x * y) / (y^2 + 4 * z^2) = -0.252 := 
sorry

end find_value_of_expression_l597_59721


namespace megs_cat_weight_l597_59733

/-- The ratio of the weight of Meg's cat to Anne's cat is 5:7 and Anne's cat weighs 8 kg more than Meg's cat. Prove that the weight of Meg's cat is 20 kg. -/
theorem megs_cat_weight
  (M A : ℝ)
  (h1 : M / A = 5 / 7)
  (h2 : A = M + 8) :
  M = 20 :=
sorry

end megs_cat_weight_l597_59733


namespace values_of_n_eq_100_l597_59736

theorem values_of_n_eq_100 :
  ∃ (n_count : ℕ), n_count = 100 ∧
    ∀ (a b c : ℕ),
      a + 11 * b + 111 * c = 900 →
      (∀ (a : ℕ), a ≥ 0) →
      (∃ (n : ℕ), n = a + 2 * b + 3 * c ∧ n_count = 100) :=
sorry

end values_of_n_eq_100_l597_59736


namespace three_different_suits_probability_l597_59772

def probability_three_different_suits := (39 / 51) * (35 / 50) = 91 / 170

theorem three_different_suits_probability (deck : Finset (Fin 52)) (h : deck.card = 52) :
  probability_three_different_suits :=
sorry

end three_different_suits_probability_l597_59772


namespace correct_multiplication_result_l597_59729

theorem correct_multiplication_result :
  ∃ x : ℕ, (x * 9 = 153) ∧ (x * 6 = 102) :=
by
  sorry

end correct_multiplication_result_l597_59729


namespace pants_cost_correct_l597_59751

-- Define the conditions as variables
def initial_money : ℕ := 71
def shirt_cost : ℕ := 5
def num_shirts : ℕ := 5
def remaining_money : ℕ := 20

-- Define intermediates necessary to show the connection between conditions and the question
def money_spent_on_shirts : ℕ := num_shirts * shirt_cost
def money_left_after_shirts : ℕ := initial_money - money_spent_on_shirts
def pants_cost : ℕ := money_left_after_shirts - remaining_money

-- The main theorem to prove the question is equal to the correct answer
theorem pants_cost_correct : pants_cost = 26 :=
by
  sorry

end pants_cost_correct_l597_59751


namespace area_triangle_PCB_correct_l597_59783

noncomputable def area_of_triangle_PCB (ABCD : Type) (A B C D P : ABCD)
  (AB_parallel_CD : ∀ (l m : ABCD → ABCD → Prop), l B A = m D C)
  (diagonals_intersect_P : ∀ (a b c d : ABCD → ABCD → ABCD → Prop), a A C P = b B D P)
  (area_APB : ℝ) (area_CPD : ℝ) : ℝ :=
  6

theorem area_triangle_PCB_correct (ABCD : Type) (A B C D P : ABCD)
  (AB_parallel_CD : ∀ (l m : ABCD → ABCD → Prop), l B A = m D C)
  (diagonals_intersect_P : ∀ (a b c d : ABCD → ABCD → ABCD → Prop), a A C P = b B D P)
  (area_APB : ℝ) (area_CPD : ℝ) :
  area_APB = 4 ∧ area_CPD = 9 → area_of_triangle_PCB ABCD A B C D P AB_parallel_CD diagonals_intersect_P area_APB area_CPD = 6 :=
by
  sorry

end area_triangle_PCB_correct_l597_59783


namespace simplify_and_evaluate_l597_59769

theorem simplify_and_evaluate (a b : ℝ) (h : |a + 2| + (b - 1)^2 = 0) : 
  (a + 3 * b) * (2 * a - b) - 2 * (a - b)^2 = -23 := by
  sorry

end simplify_and_evaluate_l597_59769


namespace money_left_after_purchase_l597_59701

def initial_toonies : Nat := 4
def value_per_toonie : Nat := 2
def total_coins : Nat := 10
def value_per_loonie : Nat := 1
def frappuccino_cost : Nat := 3

def toonies_value : Nat := initial_toonies * value_per_toonie
def loonies : Nat := total_coins - initial_toonies
def loonies_value : Nat := loonies * value_per_loonie
def initial_total : Nat := toonies_value + loonies_value
def remaining_money : Nat := initial_total - frappuccino_cost

theorem money_left_after_purchase : remaining_money = 11 := by
  sorry

end money_left_after_purchase_l597_59701


namespace problem_statement_l597_59707

theorem problem_statement (x : ℝ) (h : x + 1/x = 3) : (x - 3)^4 + 81 / (x - 3)^4 = 63 :=
by
  sorry

end problem_statement_l597_59707


namespace student_B_incorrect_l597_59797

-- Define the quadratic function and the non-zero condition on 'a'
def quadratic (a b x : ℝ) : ℝ := a * x^2 + b * x - 6

-- Conditions stated by the students
def student_A_condition (a b : ℝ) : Prop := -b / (2 * a) = 1
def student_B_condition (a b : ℝ) : Prop := quadratic a b 3 = -6
def student_C_condition (a b : ℝ) : Prop := (4 * a * (-6) - b^2) / (4 * a) = -8
def student_D_condition (a b : ℝ) : Prop := quadratic a b 3 = 0

-- The proof problem: Student B's conclusion is incorrect
theorem student_B_incorrect : 
  ∀ (a b : ℝ), 
  a ≠ 0 → 
  student_A_condition a b ∧ 
  student_C_condition a b ∧ 
  student_D_condition a b → 
  ¬ student_B_condition a b :=
by 
  -- problem converted to Lean problem format 
  -- based on the conditions provided
  sorry

end student_B_incorrect_l597_59797


namespace janet_spending_difference_l597_59737

-- Definitions for the conditions
def clarinet_hourly_rate : ℝ := 40
def clarinet_hours_per_week : ℝ := 3
def piano_hourly_rate : ℝ := 28
def piano_hours_per_week : ℝ := 5
def weeks_per_year : ℕ := 52

-- The theorem to be proven
theorem janet_spending_difference :
  (piano_hourly_rate * piano_hours_per_week * weeks_per_year - clarinet_hourly_rate * clarinet_hours_per_week * weeks_per_year) = 1040 :=
by
  sorry

end janet_spending_difference_l597_59737


namespace train_speed_l597_59731

theorem train_speed
  (length_of_train : ℕ)
  (time_to_cross_bridge : ℕ)
  (length_of_bridge : ℕ)
  (speed_conversion_factor : ℕ)
  (H1 : length_of_train = 120)
  (H2 : time_to_cross_bridge = 30)
  (H3 : length_of_bridge = 255)
  (H4 : speed_conversion_factor = 36) : 
  (length_of_train + length_of_bridge) / (time_to_cross_bridge / speed_conversion_factor) = 45 :=
by
  sorry

end train_speed_l597_59731


namespace find_a_plus_b_l597_59720

theorem find_a_plus_b (a b : ℤ) (ha : a > 0) (hb : b > 0) (h : a^2 - b^4 = 2009) : a + b = 47 :=
by
  sorry

end find_a_plus_b_l597_59720


namespace angle_measure_l597_59758

theorem angle_measure (x y : ℝ) 
  (h1 : y = 3 * x + 10) 
  (h2 : x + y = 180) : x = 42.5 :=
by
  -- Proof goes here
  sorry

end angle_measure_l597_59758


namespace melanie_marbles_l597_59762

noncomputable def melanie_blue_marbles : ℕ :=
  let sandy_dozen_marbles := 56
  let dozen := 12
  let sandy_marbles := sandy_dozen_marbles * dozen
  let ratio := 8
  sandy_marbles / ratio

theorem melanie_marbles (h1 : ∀ sandy_dozen_marbles dozen ratio, 56 = sandy_dozen_marbles ∧ sandy_dozen_marbles * dozen = 672 ∧ ratio = 8) : melanie_blue_marbles = 84 := by
  sorry

end melanie_marbles_l597_59762


namespace A_2013_eq_neg_1007_l597_59718

def A (n : ℕ) : ℤ :=
  (-1)^n * ((n + 1) / 2)

theorem A_2013_eq_neg_1007 : A 2013 = -1007 :=
by
  sorry

end A_2013_eq_neg_1007_l597_59718


namespace general_term_formula_l597_59771

variable (a : ℕ → ℤ) -- A sequence of integers 
variable (d : ℤ) -- The common difference 

-- Conditions provided
axiom h1 : a 1 = 6
axiom h2 : a 3 + a 5 = 0
axiom h_arithmetic : ∀ n, a (n + 1) = a n + d -- Arithmetic progression condition

-- The general term formula we need to prove
theorem general_term_formula : ∀ n, a n = 8 - 2 * n := 
by 
  sorry -- Proof goes here


end general_term_formula_l597_59771


namespace a7_is_1_S2022_is_4718_l597_59752

def harmonious_progressive (a : ℕ → ℕ) : Prop :=
  ∀ p q : ℕ, p > 0 → q > 0 → a p = a q → a (p + 1) = a (q + 1)

variables (a : ℕ → ℕ) (S : ℕ → ℕ)

axiom harmonious_seq : harmonious_progressive a
axiom a1 : a 1 = 1
axiom a2 : a 2 = 2
axiom a4 : a 4 = 1
axiom a6_plus_a8 : a 6 + a 8 = 6

theorem a7_is_1 : a 7 = 1 := sorry

theorem S2022_is_4718 : S 2022 = 4718 := sorry

end a7_is_1_S2022_is_4718_l597_59752


namespace lucca_bread_fraction_l597_59732

theorem lucca_bread_fraction 
  (total_bread : ℕ)
  (initial_fraction_eaten : ℚ)
  (final_pieces : ℕ)
  (bread_first_day : ℚ)
  (bread_second_day : ℚ)
  (bread_third_day : ℚ)
  (remaining_pieces_after_first_day : ℕ)
  (remaining_pieces_after_second_day : ℕ)
  (remaining_pieces_after_third_day : ℕ) :
  total_bread = 200 →
  initial_fraction_eaten = 1/4 →
  bread_first_day = initial_fraction_eaten * total_bread →
  remaining_pieces_after_first_day = total_bread - bread_first_day →
  bread_second_day = (remaining_pieces_after_first_day * bread_second_day) →
  remaining_pieces_after_second_day = remaining_pieces_after_first_day - bread_second_day →
  bread_third_day = 1/2 * remaining_pieces_after_second_day →
  remaining_pieces_after_third_day = remaining_pieces_after_second_day - bread_third_day →
  remaining_pieces_after_third_day = 45 →
  bread_second_day = 2/5 :=
by
  sorry

end lucca_bread_fraction_l597_59732


namespace a_eq_zero_l597_59722

noncomputable def f (x a : ℝ) := x^2 - abs (x + a)

theorem a_eq_zero (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 0 :=
by
  sorry

end a_eq_zero_l597_59722


namespace lighting_effect_improves_l597_59761

theorem lighting_effect_improves (a b m : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : 0 < m) : 
    (a + m) / (b + m) > a / b := 
sorry

end lighting_effect_improves_l597_59761


namespace length_of_AB_l597_59742

variables {A B P Q : ℝ}
variables (x y : ℝ)

-- Conditions
axiom h1 : A < P ∧ P < Q ∧ Q < B
axiom h2 : P - A = 3 * x
axiom h3 : B - P = 5 * x
axiom h4 : Q - A = 2 * y
axiom h5 : B - Q = 3 * y
axiom h6 : Q - P = 3

-- Theorem statement
theorem length_of_AB : B - A = 120 :=
by
  sorry

end length_of_AB_l597_59742


namespace min_value_75_l597_59784

def min_value (x y z : ℝ) := x^2 + y^2 + z^2

theorem min_value_75 
  (x y z : ℝ) 
  (h1 : (x + 5) * (y - 5) = 0) 
  (h2 : (y + 5) * (z - 5) = 0) 
  (h3 : (z + 5) * (x - 5) = 0) :
  min_value x y z = 75 := 
sorry

end min_value_75_l597_59784


namespace shorter_side_ratio_l597_59715

variable {x y : ℝ}
variables (h1 : x < y)
variables (h2 : x + y - Real.sqrt (x^2 + y^2) = 1/2 * y)

theorem shorter_side_ratio (h1 : x < y) (h2 : x + y - Real.sqrt (x^2 + y^2) = 1 / 2 * y) : x / y = 3 / 4 := 
sorry

end shorter_side_ratio_l597_59715


namespace jury_deliberation_days_l597_59791

theorem jury_deliberation_days
  (jury_selection_days trial_times jury_duty_days deliberation_hours_per_day hours_in_day : ℕ)
  (h1 : jury_selection_days = 2)
  (h2 : trial_times = 4)
  (h3 : jury_duty_days = 19)
  (h4 : deliberation_hours_per_day = 16)
  (h5 : hours_in_day = 24) :
  (jury_duty_days - jury_selection_days - (trial_times * jury_selection_days)) * deliberation_hours_per_day / hours_in_day = 6 := 
by
  sorry

end jury_deliberation_days_l597_59791


namespace julia_total_kids_l597_59747

def kidsMonday : ℕ := 7
def kidsTuesday : ℕ := 13
def kidsThursday : ℕ := 18
def kidsWednesdayCards : ℕ := 20
def kidsWednesdayHideAndSeek : ℕ := 11
def kidsWednesdayPuzzle : ℕ := 9
def kidsFridayBoardGame : ℕ := 15
def kidsFridayDrawingCompetition : ℕ := 12

theorem julia_total_kids : 
  kidsMonday + kidsTuesday + kidsThursday + kidsWednesdayCards + kidsWednesdayHideAndSeek + kidsWednesdayPuzzle + kidsFridayBoardGame + kidsFridayDrawingCompetition = 105 :=
by
  sorry

end julia_total_kids_l597_59747


namespace find_third_month_sale_l597_59734

theorem find_third_month_sale
  (sale_1 sale_2 sale_3 sale_4 sale_5 sale_6 : ℕ)
  (h1 : sale_1 = 800)
  (h2 : sale_2 = 900)
  (h4 : sale_4 = 700)
  (h5 : sale_5 = 800)
  (h6 : sale_6 = 900)
  (h_avg : (sale_1 + sale_2 + sale_3 + sale_4 + sale_5 + sale_6) / 6 = 850) : 
  sale_3 = 1000 :=
by
  sorry

end find_third_month_sale_l597_59734


namespace fraction_value_l597_59738

theorem fraction_value
  (a b c d : ℚ)
  (h1 : a / b = 1 / 4)
  (h2 : c / d = 1 / 4)
  (h3 : b ≠ 0)
  (h4 : d ≠ 0)
  (h5 : b + d ≠ 0) :
  (a + 2 * c) / (2 * b + 4 * d) = 1 / 8 :=
sorry

end fraction_value_l597_59738


namespace no_integer_a_exists_l597_59739

theorem no_integer_a_exists (a x : ℤ)
  (h : x^3 - a * x^2 - 6 * a * x + a^2 - 3 = 0)
  (unique_sol : ∀ y : ℤ, (y^3 - a * y^2 - 6 * a * y + a^2 - 3 = 0 → y = x)) :
  false :=
by 
  sorry

end no_integer_a_exists_l597_59739


namespace probability_of_event_A_l597_59730

noncomputable def probability_both_pieces_no_less_than_three_meters (L : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  if h : L = a + b 
  then (if a ≥ 3 ∧ b ≥ 3 then (L - 2 * 3) / L else 0)
  else 0

theorem probability_of_event_A : 
  probability_both_pieces_no_less_than_three_meters 11 6 5 = 5 / 11 :=
by
  -- Additional context to ensure proper definition of the problem
  sorry

end probability_of_event_A_l597_59730


namespace arccos_gt_arctan_on_interval_l597_59750

noncomputable def c : ℝ := sorry -- placeholder for the numerical solution of arccos x = arctan x

theorem arccos_gt_arctan_on_interval (x : ℝ) (hx : -1 ≤ x ∧ x < c) :
  Real.arccos x > Real.arctan x := 
sorry

end arccos_gt_arctan_on_interval_l597_59750


namespace breadth_of_rectangular_plot_l597_59757

variable (b l : ℕ)

def length_eq_thrice_breadth (b : ℕ) : ℕ := 3 * b

def area_of_rectangle_eq_2700 (b l : ℕ) : Prop := l * b = 2700

theorem breadth_of_rectangular_plot (h1 : l = 3 * b) (h2 : l * b = 2700) : b = 30 :=
by
  sorry

end breadth_of_rectangular_plot_l597_59757


namespace mark_total_payment_l597_59785

def total_cost (work_hours : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  work_hours * hourly_rate + part_cost

theorem mark_total_payment :
  total_cost 2 75 150 = 300 :=
by
  -- Proof omitted, sorry used to skip the proof
  sorry

end mark_total_payment_l597_59785


namespace find_g_of_2_l597_59760

-- Define the assumptions
variables (g : ℝ → ℝ)
axiom condition : ∀ x : ℝ, x ≠ 0 → 5 * g (1 / x) + (3 * g x) / x = Real.sqrt x

-- State the theorem to prove
theorem find_g_of_2 : g 2 = -(Real.sqrt 2) / 16 :=
by
  sorry

end find_g_of_2_l597_59760


namespace horizontal_length_of_rectangle_l597_59778

theorem horizontal_length_of_rectangle
  (P : ℕ)
  (h v : ℕ)
  (hP : P = 54)
  (hv : v = h - 3) :
  2*h + 2*v = 54 → h = 15 :=
by sorry

end horizontal_length_of_rectangle_l597_59778
