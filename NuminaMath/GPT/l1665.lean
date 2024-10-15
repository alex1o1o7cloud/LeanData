import Mathlib

namespace NUMINAMATH_GPT_inequality_am_gm_l1665_166555

theorem inequality_am_gm (a b c d : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end NUMINAMATH_GPT_inequality_am_gm_l1665_166555


namespace NUMINAMATH_GPT_molecular_weight_of_one_mole_l1665_166507

-- Conditions
def molecular_weight_6_moles : ℤ := 1404
def num_moles : ℤ := 6

-- Theorem
theorem molecular_weight_of_one_mole : (molecular_weight_6_moles / num_moles) = 234 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_one_mole_l1665_166507


namespace NUMINAMATH_GPT_sequence_term_n_l1665_166595

theorem sequence_term_n (a : ℕ → ℕ) (a1 d : ℕ) (n : ℕ) (h1 : a 1 = a1) (h2 : d = 2)
  (h3 : a n = 19) (h_seq : ∀ n, a n = a1 + (n - 1) * d) : n = 10 :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_n_l1665_166595


namespace NUMINAMATH_GPT_cannot_determine_students_answered_both_correctly_l1665_166580

-- Definitions based on the given conditions
def students_enrolled : ℕ := 25
def students_answered_q1_correctly : ℕ := 22
def students_not_taken_test : ℕ := 3
def some_students_answered_q2_correctly : Prop := -- definition stating that there's an undefined number of students that answered question 2 correctly
  ∃ n : ℕ, (n ≤ students_enrolled) ∧ n > 0

-- Statement for the proof problem
theorem cannot_determine_students_answered_both_correctly :
  ∃ n, (n ≤ students_answered_q1_correctly) ∧ n > 0 → false :=
by sorry

end NUMINAMATH_GPT_cannot_determine_students_answered_both_correctly_l1665_166580


namespace NUMINAMATH_GPT_find_angle_y_l1665_166587

-- Definitions of the angles in the triangle
def angle_ACD : ℝ := 90
def angle_DEB : ℝ := 58

-- Theorem proving the value of angle DCE (denoted as y)
theorem find_angle_y (angle_sum_property : angle_ACD + y + angle_DEB = 180) : y = 32 :=
by sorry

end NUMINAMATH_GPT_find_angle_y_l1665_166587


namespace NUMINAMATH_GPT_function_behaviour_l1665_166592

theorem function_behaviour (a : ℝ) (h : a ≠ 0) :
  ¬ ((a * (-2)^2 + 2 * a * (-2) + 1 > a * (-1)^2 + 2 * a * (-1) + 1) ∧
     (a * (-1)^2 + 2 * a * (-1) + 1 > a * 0^2 + 2 * a * 0 + 1)) :=
by
  sorry

end NUMINAMATH_GPT_function_behaviour_l1665_166592


namespace NUMINAMATH_GPT_minimum_value_y_range_of_a_l1665_166502

noncomputable def f (x a : ℝ) : ℝ := x^2 - 2*a*x - 1 + a

theorem minimum_value_y (x : ℝ) 
  (hx_pos : x > 0) : (f x 2 / x) = -2 :=
by sorry

theorem range_of_a : 
  ∀ a : ℝ, ∀ x ∈ (Set.Icc 0 2), (f x a) ≤ a ↔ a ≥ 3 / 4 :=
by sorry

end NUMINAMATH_GPT_minimum_value_y_range_of_a_l1665_166502


namespace NUMINAMATH_GPT_f_odd_and_increasing_l1665_166584

noncomputable def f (x : ℝ) : ℝ := 3^x - (1/3)^x

theorem f_odd_and_increasing : (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end NUMINAMATH_GPT_f_odd_and_increasing_l1665_166584


namespace NUMINAMATH_GPT_value_of_expression_l1665_166568

theorem value_of_expression (a b : ℝ) (h1 : ∃ x : ℝ, x^2 + 3 * x - 5 = 0)
  (h2 : ∃ y : ℝ, y^2 + 3 * y - 5 = 0)
  (h3 : a ≠ b)
  (h4 : ∀ r : ℝ, r^2 + 3 * r - 5 = 0 → r = a ∨ r = b) : a^2 + 3 * a * b + a - 2 * b = -4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1665_166568


namespace NUMINAMATH_GPT_mass_percentage_O_in_Al2_CO3_3_correct_l1665_166597

noncomputable def mass_percentage_O_in_Al2_CO3_3 : ℚ := 
  let mass_O := 9 * 16.00
  let molar_mass_Al2_CO3_3 := (2 * 26.98) + (3 * 12.01) + (9 * 16.00)
  (mass_O / molar_mass_Al2_CO3_3) * 100

theorem mass_percentage_O_in_Al2_CO3_3_correct :
  mass_percentage_O_in_Al2_CO3_3 = 61.54 :=
by
  unfold mass_percentage_O_in_Al2_CO3_3
  sorry

end NUMINAMATH_GPT_mass_percentage_O_in_Al2_CO3_3_correct_l1665_166597


namespace NUMINAMATH_GPT_display_glasses_count_l1665_166566

noncomputable def tall_cupboards := 2
noncomputable def wide_cupboards := 2
noncomputable def narrow_cupboards := 2
noncomputable def shelves_per_narrow_cupboard := 3
noncomputable def glasses_tall_cupboard := 30
noncomputable def glasses_wide_cupboard := 2 * glasses_tall_cupboard
noncomputable def glasses_narrow_cupboard := 45
noncomputable def broken_shelf_glasses := glasses_narrow_cupboard / shelves_per_narrow_cupboard

theorem display_glasses_count :
  (tall_cupboards * glasses_tall_cupboard) +
  (wide_cupboards * glasses_wide_cupboard) +
  (1 * (broken_shelf_glasses * (shelves_per_narrow_cupboard - 1)) + glasses_narrow_cupboard) =
  255 :=
by sorry

end NUMINAMATH_GPT_display_glasses_count_l1665_166566


namespace NUMINAMATH_GPT_angle_at_intersection_l1665_166524

theorem angle_at_intersection (n : ℕ) (h₁ : n = 8)
  (h₂ : ∀ i j : ℕ, (i + 1) % n ≠ j ∧ i < j)
  (h₃ : ∀ i : ℕ, i < n)
  (h₄ : ∀ i j : ℕ, (i + 1) % n = j ∨ (i + n - 1) % n = j)
  : (2 * (180 / n - (180 * (n - 2) / n) / 2)) = 90 :=
by
  sorry

end NUMINAMATH_GPT_angle_at_intersection_l1665_166524


namespace NUMINAMATH_GPT_correct_conclusions_l1665_166535

noncomputable def quadratic_solution_set (a b c : ℝ) : Prop :=
  ∀ x : ℝ, (-1 / 2 < x ∧ x < 3) ↔ (a * x^2 + b * x + c > 0)

theorem correct_conclusions (a b c : ℝ) (h : quadratic_solution_set a b c) : c > 0 ∧ 4 * a + 2 * b + c > 0 :=
  sorry

end NUMINAMATH_GPT_correct_conclusions_l1665_166535


namespace NUMINAMATH_GPT_total_amount_collected_l1665_166517

-- Define ticket prices and quantities
def adult_ticket_price : ℕ := 12
def child_ticket_price : ℕ := 4
def total_tickets_sold : ℕ := 130
def adult_tickets_sold : ℕ := 40

-- Calculate the number of child tickets sold
def child_tickets_sold : ℕ := total_tickets_sold - adult_tickets_sold

-- Calculate the total amount collected from adult tickets
def total_adult_amount_collected : ℕ := adult_tickets_sold * adult_ticket_price

-- Calculate the total amount collected from child tickets
def total_child_amount_collected : ℕ := child_tickets_sold * child_ticket_price

-- Prove the total amount collected from ticket sales
theorem total_amount_collected : total_adult_amount_collected + total_child_amount_collected = 840 := by
  sorry

end NUMINAMATH_GPT_total_amount_collected_l1665_166517


namespace NUMINAMATH_GPT_sum_of_distinct_products_of_6_23H_508_3G4_l1665_166585

theorem sum_of_distinct_products_of_6_23H_508_3G4 (G H : ℕ) : 
  (G < 10) → (H < 10) →
  (623 * 1000 + H * 100 + 508 * 10 + 3 * 10 + G * 1 + 4) % 72 = 0 →
  (if G = 0 then 0 + if G = 4 then 4 else 0 else 0) = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_sum_of_distinct_products_of_6_23H_508_3G4_l1665_166585


namespace NUMINAMATH_GPT_pipe_length_l1665_166562

theorem pipe_length (L x : ℝ) 
  (h1 : 20 = L - x)
  (h2 : 140 = L + 7 * x) : 
  L = 35 := by
  sorry

end NUMINAMATH_GPT_pipe_length_l1665_166562


namespace NUMINAMATH_GPT_sum_of_cubes_eq_five_l1665_166533

noncomputable def root_polynomial (a b c : ℂ) : Prop :=
  (a + b + c = 2) ∧ (a*b + b*c + c*a = 3) ∧ (a*b*c = 5)

theorem sum_of_cubes_eq_five (a b c : ℂ) (h : root_polynomial a b c) :
  a^3 + b^3 + c^3 = 5 :=
sorry

end NUMINAMATH_GPT_sum_of_cubes_eq_five_l1665_166533


namespace NUMINAMATH_GPT_ratio_of_awards_l1665_166500

theorem ratio_of_awards 
  (Scott_awards : ℕ) (Scott_awards_eq : Scott_awards = 4)
  (Jessie_awards : ℕ) (Jessie_awards_eq : Jessie_awards = 3 * Scott_awards)
  (rival_awards : ℕ) (rival_awards_eq : rival_awards = 24) :
  rival_awards / Jessie_awards = 2 :=
by sorry

end NUMINAMATH_GPT_ratio_of_awards_l1665_166500


namespace NUMINAMATH_GPT_factorize_expression_l1665_166582

theorem factorize_expression (a : ℚ) : 2 * a^2 - 4 * a = 2 * a * (a - 2) := by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1665_166582


namespace NUMINAMATH_GPT_sin_585_eq_neg_sqrt2_div_2_l1665_166589

theorem sin_585_eq_neg_sqrt2_div_2 : Real.sin (585 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_585_eq_neg_sqrt2_div_2_l1665_166589


namespace NUMINAMATH_GPT_tanya_number_75_less_l1665_166541

def rotate180 (d : ℕ) : ℕ :=
  match d with
  | 0 => 0
  | 1 => 1
  | 6 => 9
  | 8 => 8
  | 9 => 6
  | _ => 0 -- invalid assumption for digits outside the defined scope

def two_digit_upside_down (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  10 * rotate180 units + rotate180 tens

theorem tanya_number_75_less (n : ℕ) : 
  ∀ n, (∃ a b, n = 10 * a + b ∧ (a = 0 ∨ a = 1 ∨ a = 6 ∨ a = 8 ∨ a = 9) ∧ 
      (b = 0 ∨ b = 1 ∨ b = 6 ∨ b = 8 ∨ b = 9) ∧  
      n - two_digit_upside_down n = 75) :=
by {
  sorry
}

end NUMINAMATH_GPT_tanya_number_75_less_l1665_166541


namespace NUMINAMATH_GPT_negate_exists_l1665_166594

theorem negate_exists : 
  (¬ ∃ x : ℝ, x^2 + x - 1 > 0) ↔ (∀ x : ℝ, x^2 + x - 1 ≤ 0) :=
by sorry

end NUMINAMATH_GPT_negate_exists_l1665_166594


namespace NUMINAMATH_GPT_real_solutions_count_l1665_166593

-- Define the system of equations
def sys_eqs (x y z w : ℝ) :=
  (x = z + w + z * w * x) ∧
  (z = x + y + x * y * z) ∧
  (y = w + x + w * x * y) ∧
  (w = y + z + y * z * w)

-- The statement of the proof problem
theorem real_solutions_count : ∃ S : Finset (ℝ × ℝ × ℝ × ℝ), (∀ t : ℝ × ℝ × ℝ × ℝ, t ∈ S ↔ sys_eqs t.1 t.2.1 t.2.2.1 t.2.2.2) ∧ S.card = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_real_solutions_count_l1665_166593


namespace NUMINAMATH_GPT_outlet_pipe_rate_l1665_166577

theorem outlet_pipe_rate (V_ft : ℝ) (cf : ℝ) (V_in : ℝ) (r_in : ℝ) (r_out1 : ℝ) (t : ℝ) (r_out2 : ℝ) :
    V_ft = 30 ∧ cf = 1728 ∧
    V_in = V_ft * cf ∧
    r_in = 5 ∧ r_out1 = 9 ∧ t = 4320 ∧
    V_in = (r_out1 + r_out2 - r_in) * t →
    r_out2 = 8 := by
  intros h
  sorry

end NUMINAMATH_GPT_outlet_pipe_rate_l1665_166577


namespace NUMINAMATH_GPT_masha_dolls_l1665_166504

theorem masha_dolls (n : ℕ) (h : (n / 2) * 1 + (n / 4) * 2 + (n / 4) * 4 = 24) : n = 12 :=
sorry

end NUMINAMATH_GPT_masha_dolls_l1665_166504


namespace NUMINAMATH_GPT_maximum_value_l1665_166548

theorem maximum_value (R P K : ℝ) (h₁ : 3 * Real.sqrt 3 * R ≥ P) (h₂ : K = P * R / 4) : 
  (K * P) / (R^3) ≤ 27 / 4 :=
by
  sorry

end NUMINAMATH_GPT_maximum_value_l1665_166548


namespace NUMINAMATH_GPT_mary_initial_baseball_cards_l1665_166579

theorem mary_initial_baseball_cards (X : ℕ) :
  (X - 8 + 26 + 40 = 84) → (X = 26) :=
by
  sorry

end NUMINAMATH_GPT_mary_initial_baseball_cards_l1665_166579


namespace NUMINAMATH_GPT_rachel_weight_l1665_166532

theorem rachel_weight :
  ∃ R : ℝ, (R + (R + 6) + (R - 15)) / 3 = 72 ∧ R = 75 :=
by
  sorry

end NUMINAMATH_GPT_rachel_weight_l1665_166532


namespace NUMINAMATH_GPT_percentage_discount_on_pencils_l1665_166552

-- Establish the given conditions
variable (cucumbers pencils price_per_cucumber price_per_pencil total_spent : ℕ)
variable (h1 : cucumbers = 100)
variable (h2 : price_per_cucumber = 20)
variable (h3 : price_per_pencil = 20)
variable (h4 : total_spent = 2800)
variable (h5 : cucumbers = 2 * pencils)

-- Propose the statement to be proved
theorem percentage_discount_on_pencils : 20 * pencils * price_per_pencil = 20 * (total_spent - cucumbers * price_per_cucumber) ∧ pencils = 50 ∧ ((total_spent - cucumbers * price_per_cucumber) * 100 = 80 * pencils * price_per_pencil) :=
by
  sorry

end NUMINAMATH_GPT_percentage_discount_on_pencils_l1665_166552


namespace NUMINAMATH_GPT_age_of_B_present_l1665_166596

theorem age_of_B_present (A B C : ℕ) (h1 : A + B + C = 90)
  (h2 : (A - 10) * 2 = (B - 10))
  (h3 : (B - 10) * 3 = (C - 10) * 2) :
  B = 30 := 
sorry

end NUMINAMATH_GPT_age_of_B_present_l1665_166596


namespace NUMINAMATH_GPT_line_through_intersection_of_circles_l1665_166518

theorem line_through_intersection_of_circles :
  ∀ (x y : ℝ),
    (x^2 + y^2 + 4 * x - 4 * y - 12 = 0) ∧
    (x^2 + y^2 + 2 * x + 4 * y - 4 = 0) →
    (x - 4 * y - 4 = 0) :=
by sorry

end NUMINAMATH_GPT_line_through_intersection_of_circles_l1665_166518


namespace NUMINAMATH_GPT_valid_votes_per_candidate_l1665_166560

theorem valid_votes_per_candidate (total_votes : ℕ) (invalid_percentage valid_percentage_A valid_percentage_B : ℚ) 
                                  (A_votes B_votes C_votes valid_votes : ℕ) :
  total_votes = 1250000 →
  invalid_percentage = 20 →
  valid_percentage_A = 45 →
  valid_percentage_B = 35 →
  valid_votes = total_votes * (1 - invalid_percentage / 100) →
  A_votes = valid_votes * (valid_percentage_A / 100) →
  B_votes = valid_votes * (valid_percentage_B / 100) →
  C_votes = valid_votes - A_votes - B_votes →
  valid_votes = 1000000 ∧ A_votes = 450000 ∧ B_votes = 350000 ∧ C_votes = 200000 :=
by {
  sorry
}

end NUMINAMATH_GPT_valid_votes_per_candidate_l1665_166560


namespace NUMINAMATH_GPT_sum_of_proper_divisors_less_than_100_of_780_l1665_166569

def is_divisor (n d : ℕ) : Bool :=
  d ∣ n

def proper_divisors (n : ℕ) : List ℕ :=
  (List.range n).filter (λ d => d ∣ n ∧ d < n)

def proper_divisors_less_than (n bound : ℕ) : List ℕ :=
  (proper_divisors n).filter (λ d => d < bound)

def sum_list (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => acc + x) 0

theorem sum_of_proper_divisors_less_than_100_of_780 :
  sum_list (proper_divisors_less_than 780 100) = 428 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_proper_divisors_less_than_100_of_780_l1665_166569


namespace NUMINAMATH_GPT_opposite_of_neg_one_third_l1665_166515

theorem opposite_of_neg_one_third : -(- (1 / 3)) = (1 / 3) :=
by sorry

end NUMINAMATH_GPT_opposite_of_neg_one_third_l1665_166515


namespace NUMINAMATH_GPT_jerry_trips_l1665_166591

-- Define the conditions
def trays_per_trip : Nat := 8
def trays_table1 : Nat := 9
def trays_table2 : Nat := 7

-- Define the proof problem
theorem jerry_trips :
  trays_table1 + trays_table2 = 16 →
  (16 / trays_per_trip) = 2 :=
by
  sorry

end NUMINAMATH_GPT_jerry_trips_l1665_166591


namespace NUMINAMATH_GPT_tylenol_intake_proof_l1665_166529

noncomputable def calculate_tylenol_intake_grams
  (tablet_mg : ℕ) (tablets_per_dose : ℕ) (hours_per_dose : ℕ) (total_hours : ℕ) : ℕ :=
  let doses := total_hours / hours_per_dose
  let total_mg := doses * tablets_per_dose * tablet_mg
  total_mg / 1000

theorem tylenol_intake_proof : calculate_tylenol_intake_grams 500 2 4 12 = 3 :=
  by sorry

end NUMINAMATH_GPT_tylenol_intake_proof_l1665_166529


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l1665_166509

theorem angle_in_third_quadrant (α : ℝ) (h1 : Real.cos α < 0) (h2 : Real.tan α > 0) : 
  (π < α ∧ α < 3 * π / 2) :=
sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l1665_166509


namespace NUMINAMATH_GPT_geometric_sequence_q_cubed_l1665_166505

theorem geometric_sequence_q_cubed (q a_1 : ℝ) (h1 : q ≠ 0) (h2 : q ≠ 1) 
(h3 : 2 * (a_1 * (1 - q^9) / (1 - q)) = (a_1 * (1 - q^3) / (1 - q)) + (a_1 * (1 - q^6) / (1 - q))) : 
  q^3 = -1/2 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_q_cubed_l1665_166505


namespace NUMINAMATH_GPT_jackson_difference_l1665_166559

theorem jackson_difference :
  let Jackson_initial := 500
  let Brandon_initial := 500
  let Meagan_initial := 700
  let Jackson_final := Jackson_initial * 4
  let Brandon_final := Brandon_initial * 0.20
  let Meagan_final := Meagan_initial + (Meagan_initial * 0.50)
  Jackson_final - (Brandon_final + Meagan_final) = 850 :=
by
  sorry

end NUMINAMATH_GPT_jackson_difference_l1665_166559


namespace NUMINAMATH_GPT_solve_for_x_l1665_166549

open Real

theorem solve_for_x (x : ℝ) (h1 : x > 0) (h2 : 6 * sqrt (4 + x) + 6 * sqrt (4 - x) = 9 * sqrt 2) : 
  x = sqrt 255 / 4 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1665_166549


namespace NUMINAMATH_GPT_pencils_distribution_count_l1665_166544

def count_pencils_distribution : ℕ :=
  let total_pencils := 10
  let friends := 4
  let adjusted_pencils := total_pencils - friends
  Nat.choose (adjusted_pencils + friends - 1) (friends - 1)

theorem pencils_distribution_count :
  count_pencils_distribution = 84 := 
  by sorry

end NUMINAMATH_GPT_pencils_distribution_count_l1665_166544


namespace NUMINAMATH_GPT_system_of_equations_solution_l1665_166563

theorem system_of_equations_solution (x y z : ℝ) 
  (h1 : x + 2 * y = 4)
  (h2 : 2 * x + 5 * y - 2 * z = 11)
  (h3 : 3 * x - 5 * y + 2 * z = -1) : 
  x = 2 ∧ y = 1 ∧ z = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_system_of_equations_solution_l1665_166563


namespace NUMINAMATH_GPT_polynomials_symmetric_l1665_166573

noncomputable def P : ℕ → (ℝ → ℝ → ℝ → ℝ)
  | 0       => λ x y z => 1
  | (m + 1) => λ x y z => (x + z) * (y + z) * (P m x y (z + 1)) - z^2 * (P m x y z)

theorem polynomials_symmetric (m : ℕ) (x y z : ℝ) : 
  P m x y z = P m y x z ∧ P m x y z = P m x z y := 
sorry

end NUMINAMATH_GPT_polynomials_symmetric_l1665_166573


namespace NUMINAMATH_GPT_turnip_heavier_than_zhuchka_l1665_166527

theorem turnip_heavier_than_zhuchka {C B M T : ℝ} 
  (h1 : B = 3 * C)
  (h2 : M = C / 10)
  (h3 : T = 60 * M) : 
  T / B = 2 :=
by
  sorry

end NUMINAMATH_GPT_turnip_heavier_than_zhuchka_l1665_166527


namespace NUMINAMATH_GPT_gcd_7488_12467_eq_39_l1665_166565

noncomputable def gcd_7488_12467 : ℕ := Nat.gcd 7488 12467

theorem gcd_7488_12467_eq_39 : gcd_7488_12467 = 39 :=
sorry

end NUMINAMATH_GPT_gcd_7488_12467_eq_39_l1665_166565


namespace NUMINAMATH_GPT_cost_of_purchasing_sandwiches_and_sodas_l1665_166528

def sandwich_price : ℕ := 4
def soda_price : ℕ := 1
def num_sandwiches : ℕ := 6
def num_sodas : ℕ := 5
def total_cost : ℕ := 29

theorem cost_of_purchasing_sandwiches_and_sodas :
  (num_sandwiches * sandwich_price + num_sodas * soda_price) = total_cost :=
by
  sorry

end NUMINAMATH_GPT_cost_of_purchasing_sandwiches_and_sodas_l1665_166528


namespace NUMINAMATH_GPT_factorization_of_polynomial_solve_quadratic_equation_l1665_166543

-- Problem 1: Factorization
theorem factorization_of_polynomial : ∀ y : ℝ, 2 * y^2 - 8 = 2 * (y + 2) * (y - 2) :=
by
  intro y
  sorry

-- Problem 2: Solving the quadratic equation
theorem solve_quadratic_equation : ∀ x : ℝ, x^2 + 4 * x + 3 = 0 ↔ x = -1 ∨ x = -3 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factorization_of_polynomial_solve_quadratic_equation_l1665_166543


namespace NUMINAMATH_GPT_black_white_area_ratio_l1665_166512

theorem black_white_area_ratio :
  let r1 := 2
  let r2 := 6
  let r3 := 10
  let r4 := 14
  let r5 := 18
  let area (r : ℝ) := π * r^2
  let black_area := area r1 + (area r3 - area r2) + (area r5 - area r4)
  let white_area := (area r2 - area r1) + (area r4 - area r3)
  black_area / white_area = (49 : ℝ) / 32 :=
by
  sorry

end NUMINAMATH_GPT_black_white_area_ratio_l1665_166512


namespace NUMINAMATH_GPT_product_repeating_decimal_l1665_166534

theorem product_repeating_decimal (p : ℚ) (h₁ : p = 152 / 333) : 
  p * 7 = 1064 / 333 :=
  by
    sorry

end NUMINAMATH_GPT_product_repeating_decimal_l1665_166534


namespace NUMINAMATH_GPT_cell_cycle_correct_statement_l1665_166574

theorem cell_cycle_correct_statement :
  ∃ (correct_statement : String), correct_statement = "In the cell cycle, chromatin DNA is easier to replicate than chromosome DNA" :=
by
  let A := "The separation of alleles occurs during the interphase of the cell cycle"
  let B := "In the cell cycle of plant cells, spindle fibers appear during the interphase"
  let C := "In the cell cycle, chromatin DNA is easier to replicate than chromosome DNA"
  let D := "In the cell cycle of liver cells, chromosomes exist for a longer time than chromatin"
  existsi C
  sorry

end NUMINAMATH_GPT_cell_cycle_correct_statement_l1665_166574


namespace NUMINAMATH_GPT_friends_count_l1665_166501

-- Define that Laura has 28 blocks
def blocks := 28

-- Define that each friend gets 7 blocks
def blocks_per_friend := 7

-- The proof statement we want to prove
theorem friends_count : blocks / blocks_per_friend = 4 := by
  sorry

end NUMINAMATH_GPT_friends_count_l1665_166501


namespace NUMINAMATH_GPT_parabola_shifting_produces_k_l1665_166520

theorem parabola_shifting_produces_k
  (k : ℝ)
  (h1 : -k/2 > 0)
  (h2 : (0 : ℝ) = (((0 : ℝ) - 3) + k/2)^2 - (5*k^2)/4 + 1)
  :
  k = -5 :=
sorry

end NUMINAMATH_GPT_parabola_shifting_produces_k_l1665_166520


namespace NUMINAMATH_GPT_remainder_of_poly_division_l1665_166583

theorem remainder_of_poly_division :
  ∀ (x : ℝ), (x^2023 + x + 1) % (x^6 - x^4 + x^2 - 1) = x^7 + x + 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_poly_division_l1665_166583


namespace NUMINAMATH_GPT_angle_relation_in_triangle_l1665_166523

theorem angle_relation_in_triangle
    (a b c : ℝ)
    (A B C : ℝ)
    (h1 : b * (a + b) * (b + c) = a^3 + b * (a^2 + c^2) + c^3)
    (h2 : A + B + C = π) 
    (h3 : A > 0) 
    (h4 : B > 0) 
    (h5 : C > 0) :
    (1 / (Real.sqrt A + Real.sqrt B)) + (1 / (Real.sqrt B + Real.sqrt C)) = (2 / (Real.sqrt C + Real.sqrt A)) :=
sorry

end NUMINAMATH_GPT_angle_relation_in_triangle_l1665_166523


namespace NUMINAMATH_GPT_percentage_less_than_a_plus_d_l1665_166588

def symmetric_distribution (a d : ℝ) (p : ℝ) : Prop :=
  p = (68 / 100 : ℝ) ∧ 
  (p / 2) = (34 / 100 : ℝ)

theorem percentage_less_than_a_plus_d (a d : ℝ) 
  (symmetry : symmetric_distribution a d (68 / 100)) : 
  (0.5 + (34 / 100) : ℝ) = (84 / 100 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_percentage_less_than_a_plus_d_l1665_166588


namespace NUMINAMATH_GPT_base_2_base_3_product_is_144_l1665_166537

def convert_base_2_to_10 (n : ℕ) : ℕ :=
  match n with
  | 1001 => 9
  | _ => 0 -- For simplicity, only handle 1001_2

def convert_base_3_to_10 (n : ℕ) : ℕ :=
  match n with
  | 121 => 16
  | _ => 0 -- For simplicity, only handle 121_3

theorem base_2_base_3_product_is_144 :
  convert_base_2_to_10 1001 * convert_base_3_to_10 121 = 144 :=
by
  sorry

end NUMINAMATH_GPT_base_2_base_3_product_is_144_l1665_166537


namespace NUMINAMATH_GPT_total_distance_l1665_166581

def morning_distance : ℕ := 2
def evening_multiplier : ℕ := 5

theorem total_distance : morning_distance + (evening_multiplier * morning_distance) = 12 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_l1665_166581


namespace NUMINAMATH_GPT_students_only_english_l1665_166542

variable (total_students both_english_german enrolled_german: ℕ)

theorem students_only_english :
  total_students = 45 ∧ both_english_german = 12 ∧ enrolled_german = 22 ∧
  (∀ S E G B : ℕ, S = total_students ∧ B = both_english_german ∧ G = enrolled_german - B ∧
   (S = E + G + B) → E = 23) :=
by
  sorry

end NUMINAMATH_GPT_students_only_english_l1665_166542


namespace NUMINAMATH_GPT_complete_square_solution_l1665_166550

theorem complete_square_solution (x : ℝ) :
  x^2 - 2*x - 3 = 0 → (x - 1)^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_complete_square_solution_l1665_166550


namespace NUMINAMATH_GPT_total_bricks_required_l1665_166571

def courtyard_length : ℕ := 24 * 100  -- convert meters to cm
def courtyard_width : ℕ := 14 * 100  -- convert meters to cm
def brick_length : ℕ := 25
def brick_width : ℕ := 15

-- Calculate the area of the courtyard in square centimeters
def courtyard_area : ℕ := courtyard_length * courtyard_width

-- Calculate the area of one brick in square centimeters
def brick_area : ℕ := brick_length * brick_width

theorem total_bricks_required :  courtyard_area / brick_area = 8960 := by
  -- This part will have the proof, for now, we use sorry to skip it
  sorry

end NUMINAMATH_GPT_total_bricks_required_l1665_166571


namespace NUMINAMATH_GPT_find_word_l1665_166513

theorem find_word (antonym : Nat) (cond : antonym = 26) : String :=
  "seldom"

end NUMINAMATH_GPT_find_word_l1665_166513


namespace NUMINAMATH_GPT_E1_E2_complementary_l1665_166572

-- Define the universal set for a fair die with six faces
def universalSet : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define each event as a set based on the problem conditions
def E1 : Set ℕ := {1, 3, 5}
def E2 : Set ℕ := {2, 4, 6}
def E3 : Set ℕ := {4, 5, 6}
def E4 : Set ℕ := {1, 2}

-- Define complementary events
def areComplementary (A B : Set ℕ) : Prop :=
  (A ∪ B = universalSet) ∧ (A ∩ B = ∅)

-- State the theorem that events E1 and E2 are complementary
theorem E1_E2_complementary : areComplementary E1 E2 :=
sorry

end NUMINAMATH_GPT_E1_E2_complementary_l1665_166572


namespace NUMINAMATH_GPT_largest_number_of_stores_visited_l1665_166521

-- Definitions of the conditions
def num_stores := 7
def total_visits := 21
def num_shoppers := 11
def two_stores_visitors := 7
def at_least_one_store (n : ℕ) : Prop := n ≥ 1

-- The goal statement
theorem largest_number_of_stores_visited :
  ∃ n, n ≤ num_shoppers ∧ 
       at_least_one_store n ∧ 
       (n * 2 + (num_shoppers - n)) <= total_visits ∧ 
       (num_shoppers - n) ≥ 3 → 
       n = 4 :=
sorry

end NUMINAMATH_GPT_largest_number_of_stores_visited_l1665_166521


namespace NUMINAMATH_GPT_find_a_l1665_166564

noncomputable def f (a x : ℝ) : ℝ :=
if x < 1 then 2 * x + a else -x - 2 * a

theorem find_a (a : ℝ) (h : a ≠ 0) (h_eq : f a (1 - a) = f a (1 + a)) : a = -3 / 4 :=
sorry

end NUMINAMATH_GPT_find_a_l1665_166564


namespace NUMINAMATH_GPT_g_two_gt_one_third_g_n_gt_one_third_l1665_166575

def seq_a (n : ℕ) : ℕ := 3 * n - 2
noncomputable def f (n : ℕ) : ℝ := (Finset.range n).sum (λ i => 1 / (seq_a (i + 1) : ℝ))
noncomputable def g (n : ℕ) : ℝ := f (n^2) - f (n - 1)

theorem g_two_gt_one_third : g 2 > 1 / 3 :=
sorry

theorem g_n_gt_one_third (n : ℕ) (h : n ≥ 3) : g n > 1 / 3 :=
sorry

end NUMINAMATH_GPT_g_two_gt_one_third_g_n_gt_one_third_l1665_166575


namespace NUMINAMATH_GPT_find_m_of_inverse_proportion_l1665_166553

theorem find_m_of_inverse_proportion (k : ℝ) (m : ℝ) 
(A_cond : (-1) * 3 = k) 
(B_cond : 2 * m = k) : 
m = -3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_m_of_inverse_proportion_l1665_166553


namespace NUMINAMATH_GPT_max_value_f_1_max_value_f_2_max_value_f_3_l1665_166586
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.log x - m * x

theorem max_value_f_1 (m : ℝ) (h : m ≤ 1 / Real.exp 1) :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → f x m ≤ 1 - m * Real.exp 1 :=
sorry

theorem max_value_f_2 (m : ℝ) (h1 : 1 / Real.exp 1 < m) (h2 : m < 1) :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → f x m ≤ -Real.log m - 1 :=
sorry

theorem max_value_f_3 (m : ℝ) (h : m ≥ 1) :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → f x m ≤ -m :=
sorry

end NUMINAMATH_GPT_max_value_f_1_max_value_f_2_max_value_f_3_l1665_166586


namespace NUMINAMATH_GPT_min_value_quadratic_l1665_166519

theorem min_value_quadratic (x : ℝ) : -2 * x^2 + 8 * x + 5 ≥ -2 * (2 - x)^2 + 13 :=
by
  sorry

end NUMINAMATH_GPT_min_value_quadratic_l1665_166519


namespace NUMINAMATH_GPT_count_1320_factors_l1665_166530

-- Prime factorization function
def primeFactors (n : ℕ) : List ℕ :=
  sorry

-- Count factors function based on prime factorization
def countFactors (factors : List (ℕ × ℕ)) : ℕ :=
  factors.foldr (fun (p : ℕ × ℕ) acc => acc * (p.snd + 1)) 1

theorem count_1320_factors : countFactors [(2, 3), (3, 1), (5, 1), (11, 1)] = 32 :=
by
  sorry

end NUMINAMATH_GPT_count_1320_factors_l1665_166530


namespace NUMINAMATH_GPT_triangle_properties_l1665_166511

-- Definitions of sides of the triangle
def a : ℕ := 15
def b : ℕ := 11
def c : ℕ := 18

-- Definition of the triangle inequality theorem in the context
def triangle_inequality (x y z : ℕ) : Prop :=
  x + y > z ∧ x + z > y ∧ y + z > x

-- Perimeter calculation
def perimeter (x y z : ℕ) : ℕ :=
  x + y + z

-- Stating the proof problem
theorem triangle_properties : triangle_inequality a b c ∧ perimeter a b c = 44 :=
by
  -- Start the process for the actual proof that will be filled out
  sorry

end NUMINAMATH_GPT_triangle_properties_l1665_166511


namespace NUMINAMATH_GPT_rhombus_diagonal_l1665_166547

theorem rhombus_diagonal
  (d1 : ℝ) (d2 : ℝ) (area : ℝ) 
  (h1 : d1 = 17) (h2 : area = 170) 
  (h3 : area = (d1 * d2) / 2) : d2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_l1665_166547


namespace NUMINAMATH_GPT_fraction_decomposition_l1665_166503

theorem fraction_decomposition :
  (1 : ℚ) / 4 = (1 : ℚ) / 8 + (1 : ℚ) / 8 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_fraction_decomposition_l1665_166503


namespace NUMINAMATH_GPT_tickets_used_l1665_166561

def total_rides (ferris_wheel_rides bumper_car_rides : ℕ) : ℕ :=
  ferris_wheel_rides + bumper_car_rides

def tickets_per_ride : ℕ := 3

def total_tickets (total_rides tickets_per_ride : ℕ) : ℕ :=
  total_rides * tickets_per_ride

theorem tickets_used :
  total_tickets (total_rides 7 3) tickets_per_ride = 30 := by
  sorry

end NUMINAMATH_GPT_tickets_used_l1665_166561


namespace NUMINAMATH_GPT_find_k_l1665_166525

theorem find_k 
  (t k r : ℝ)
  (h1 : t = 5 / 9 * (k - 32))
  (h2 : r = 3 * t)
  (h3 : r = 150) : 
  k = 122 := 
sorry

end NUMINAMATH_GPT_find_k_l1665_166525


namespace NUMINAMATH_GPT_robert_cash_spent_as_percentage_l1665_166599

theorem robert_cash_spent_as_percentage 
  (raw_material_cost : ℤ) (machinery_cost : ℤ) (total_amount : ℤ) 
  (h_raw : raw_material_cost = 100) 
  (h_machinery : machinery_cost = 125) 
  (h_total : total_amount = 250) :
  ((total_amount - (raw_material_cost + machinery_cost)) * 100 / total_amount) = 10 := 
by 
  -- Proof will be filled here
  sorry

end NUMINAMATH_GPT_robert_cash_spent_as_percentage_l1665_166599


namespace NUMINAMATH_GPT_remainder_of_division_l1665_166570

theorem remainder_of_division : 
  ∀ (L x : ℕ), (L = 1430) → 
               (L - x = 1311) → 
               (L = 11 * x + (L % x)) → 
               (L % x = 121) :=
by
  intros L x L_value diff quotient
  sorry

end NUMINAMATH_GPT_remainder_of_division_l1665_166570


namespace NUMINAMATH_GPT_smallest_x_y_sum_l1665_166578

theorem smallest_x_y_sum (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ y)
                        (h4 : (1 / (x : ℝ)) + (1 / (y : ℝ)) = (1 / 20)) :
    x + y = 81 :=
sorry

end NUMINAMATH_GPT_smallest_x_y_sum_l1665_166578


namespace NUMINAMATH_GPT_nikita_productivity_l1665_166558

theorem nikita_productivity 
  (x y : ℕ) 
  (h1 : 3 * x + 2 * y = 7) 
  (h2 : 5 * x + 3 * y = 11) : 
  y = 2 := 
sorry

end NUMINAMATH_GPT_nikita_productivity_l1665_166558


namespace NUMINAMATH_GPT_quadratic_roots_l1665_166567

theorem quadratic_roots {a : ℝ} :
  (4 < a ∧ a < 6) ∨ (a > 12) → 
  (∃ x1 x2 : ℝ, x1 = a + Real.sqrt (18 * (a - 4)) ∧ x2 = a - Real.sqrt (18 * (a - 4)) ∧ x1 > 0 ∧ x2 > 0) :=
by sorry

end NUMINAMATH_GPT_quadratic_roots_l1665_166567


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_l1665_166538

variable (a : ℕ → ℝ)

theorem arithmetic_sequence_a5 (h : a 2 + a 8 = 15 - a 5) : a 5 = 5 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_l1665_166538


namespace NUMINAMATH_GPT_range_of_a_satisfies_l1665_166526

noncomputable def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x : ℝ, f (-(x + 1)) = -f (x + 1)) ∧
  (∀ x1 x2 : ℝ, 0 ≤ x1 ∧ 0 ≤ x2 → (f x1 - f x2) / (x1 - x2) > -1) ∧
  (∀ a : ℝ, f (a^2 - 1) + f (a - 1) + a^2 + a > 2)

theorem range_of_a_satisfies (f : ℝ → ℝ) (hf_conditions : satisfies_conditions f) :
  {a : ℝ | f (a^2 - 1) + f (a - 1) + a^2 + a > 2} = {a | a < -2 ∨ a > 1} :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_satisfies_l1665_166526


namespace NUMINAMATH_GPT_largest_d_l1665_166554

variable (a b c d : ℤ)

def condition : Prop := a + 2 = b - 1 ∧ a + 2 = c + 3 ∧ a + 2 = d - 4

theorem largest_d (h : condition a b c d) : d > a ∧ d > b ∧ d > c :=
by
  -- Assuming the condition holds, we need to prove d > a, d > b, and d > c
  sorry

end NUMINAMATH_GPT_largest_d_l1665_166554


namespace NUMINAMATH_GPT_train_stoppage_time_l1665_166508

theorem train_stoppage_time
    (speed_without_stoppages : ℕ)
    (speed_with_stoppages : ℕ)
    (time_unit : ℕ)
    (h1 : speed_without_stoppages = 50)
    (h2 : speed_with_stoppages = 30)
    (h3 : time_unit = 60) :
    (time_unit * (speed_without_stoppages - speed_with_stoppages) / speed_without_stoppages) = 24 :=
by
  sorry

end NUMINAMATH_GPT_train_stoppage_time_l1665_166508


namespace NUMINAMATH_GPT_solve_for_y_l1665_166557

theorem solve_for_y (y : ℝ) : 7 - y = 4 → y = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l1665_166557


namespace NUMINAMATH_GPT_ratio_of_third_week_growth_l1665_166576

-- Define the given conditions
def week1_growth : ℕ := 2  -- growth in week 1
def week2_growth : ℕ := 2 * week1_growth  -- growth in week 2
def total_height : ℕ := 22  -- total height after three weeks

/- 
  Statement: Prove that the growth in the third week divided by 
  the growth in the second week is 4, i.e., the ratio 4:1.
-/
theorem ratio_of_third_week_growth :
  ∃ x : ℕ, 4 * x = (total_height - week1_growth - week2_growth) ∧ x = 4 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_ratio_of_third_week_growth_l1665_166576


namespace NUMINAMATH_GPT_evaluate_g_at_neg3_l1665_166536

def g (x : ℤ) : ℤ := x^2 - x + 2 * x^3

theorem evaluate_g_at_neg3 : g (-3) = -42 := by
  sorry

end NUMINAMATH_GPT_evaluate_g_at_neg3_l1665_166536


namespace NUMINAMATH_GPT_probability_of_selecting_A_and_B_l1665_166506

-- Define the combination function
def combination (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Define the conditions
def students : ℕ := 5
def to_select : ℕ := 3

-- Define the favorable outcomes where A and B are selected
def favorable_outcomes : ℕ := combination 3 1

-- Define the total outcomes of selecting 3 out of 5 students
def total_outcomes : ℕ := combination 5 3

-- Define the probability calculation
def probability_of_A_and_B : ℚ := favorable_outcomes / total_outcomes

-- Proposition to prove
theorem probability_of_selecting_A_and_B : probability_of_A_and_B = 3 / 10 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_A_and_B_l1665_166506


namespace NUMINAMATH_GPT_domain_g_l1665_166551

def domain_f (a : ℝ) : Prop := 0 ≤ a ∧ a ≤ 2
def g (x : ℝ) (f : ℝ → ℝ) : Prop := 
  ((1 < x) ∧ (x ≤ Real.sqrt 3)) ∧ domain_f (x^2 - 1) ∧ (0 < x - 1 ∧ x - 1 < 1)

theorem domain_g (x : ℝ) (f : ℝ → ℝ) (hf : ∀ a, domain_f a → True) : 
  g x f ↔ 1 < x ∧ x ≤ Real.sqrt 3 :=
by 
  sorry

end NUMINAMATH_GPT_domain_g_l1665_166551


namespace NUMINAMATH_GPT_robin_total_spending_l1665_166540

def jelly_bracelets_total_cost : ℕ :=
  let names := ["Jessica", "Tori", "Lily", "Patrice"]
  let total_letters := names.foldl (λ acc name => acc + name.length) 0
  total_letters * 2

theorem robin_total_spending : jelly_bracelets_total_cost = 44 := by
  sorry

end NUMINAMATH_GPT_robin_total_spending_l1665_166540


namespace NUMINAMATH_GPT_g_of_3_l1665_166522

def g (x : ℝ) : ℝ := 5 * x ^ 4 + 4 * x ^ 3 - 7 * x ^ 2 + 3 * x - 2

theorem g_of_3 : g 3 = 401 :=
by
    -- proof will go here
    sorry

end NUMINAMATH_GPT_g_of_3_l1665_166522


namespace NUMINAMATH_GPT_cheaper_store_difference_in_cents_l1665_166514

/-- Given the following conditions:
1. Best Deals offers \$12 off the list price of \$52.99.
2. Market Value offers 20% off the list price of \$52.99.
 -/
theorem cheaper_store_difference_in_cents :
  let list_price : ℝ := 52.99
  let best_deals_price := list_price - 12
  let market_value_price := list_price * 0.80
  best_deals_price < market_value_price →
  let difference_in_dollars := market_value_price - best_deals_price
  let difference_in_cents := difference_in_dollars * 100
  difference_in_cents = 140 := by
  intro h
  let list_price : ℝ := 52.99
  let best_deals_price := list_price - 12
  let market_value_price := list_price * 0.80
  let difference_in_dollars := market_value_price - best_deals_price
  let difference_in_cents := difference_in_dollars * 100
  sorry

end NUMINAMATH_GPT_cheaper_store_difference_in_cents_l1665_166514


namespace NUMINAMATH_GPT_samatha_routes_l1665_166516

-- Definitions based on the given conditions
def blocks_from_house_to_southwest_corner := 4
def blocks_through_park := 1
def blocks_from_northeast_corner_to_school := 4
def blocks_from_school_to_library := 3

-- Number of ways to arrange movements
def number_of_routes_house_to_southwest : ℕ :=
  Nat.choose blocks_from_house_to_southwest_corner 1

def number_of_routes_through_park : ℕ := blocks_through_park

def number_of_routes_northeast_to_school : ℕ :=
  Nat.choose blocks_from_northeast_corner_to_school 1

def number_of_routes_school_to_library : ℕ :=
  Nat.choose blocks_from_school_to_library 1

-- Total number of different routes
def total_number_of_routes : ℕ :=
  number_of_routes_house_to_southwest *
  number_of_routes_through_park *
  number_of_routes_northeast_to_school *
  number_of_routes_school_to_library

theorem samatha_routes (n : ℕ) (h : n = 48) :
  total_number_of_routes = n :=
  by
    -- Proof is skipped
    sorry

end NUMINAMATH_GPT_samatha_routes_l1665_166516


namespace NUMINAMATH_GPT_simplify_expression_l1665_166556

theorem simplify_expression : 8 * (15 / 4) * (-56 / 45) = -112 / 3 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l1665_166556


namespace NUMINAMATH_GPT_problem_statement_l1665_166546

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 2 * x - 3

theorem problem_statement : (f (g (f 1))) / (g (f (g 1))) = (-23 : ℝ) / 5 :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l1665_166546


namespace NUMINAMATH_GPT_num_integer_terms_sequence_l1665_166531

noncomputable def sequence_starting_at_8820 : Nat := 8820

def divide_by_5 (n : Nat) : Nat := n / 5

theorem num_integer_terms_sequence :
  let seq := [sequence_starting_at_8820, divide_by_5 sequence_starting_at_8820]
  seq = [8820, 1764] →
  seq.length = 2 := by
  sorry

end NUMINAMATH_GPT_num_integer_terms_sequence_l1665_166531


namespace NUMINAMATH_GPT_sin_150_eq_half_l1665_166539

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_sin_150_eq_half_l1665_166539


namespace NUMINAMATH_GPT_maximum_value_of_a_squared_b_l1665_166598

theorem maximum_value_of_a_squared_b {a b : ℝ} (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : a * (a + b) = 27) : 
  a^2 * b ≤ 54 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_a_squared_b_l1665_166598


namespace NUMINAMATH_GPT_square_circle_area_ratio_l1665_166510

theorem square_circle_area_ratio {r : ℝ} (h : ∀ s : ℝ, 2 * r = s * Real.sqrt 2) :
  (2 * r ^ 2) / (Real.pi * r ^ 2) = 2 / Real.pi :=
by
  sorry

end NUMINAMATH_GPT_square_circle_area_ratio_l1665_166510


namespace NUMINAMATH_GPT_digit_B_in_4B52B_divisible_by_9_l1665_166545

theorem digit_B_in_4B52B_divisible_by_9 (B : ℕ) (h : (2 * B + 11) % 9 = 0) : B = 8 :=
by {
  sorry
}

end NUMINAMATH_GPT_digit_B_in_4B52B_divisible_by_9_l1665_166545


namespace NUMINAMATH_GPT_exists_N_for_sqrt_expressions_l1665_166590

theorem exists_N_for_sqrt_expressions 
  (p q n : ℕ) (hp : 0 < p) (hq : 0 < q) (hn : 0 < n) (h_q_le_p2 : q ≤ p^2) :
  ∃ N : ℕ, 
    (N > 0) ∧ 
    ((p - Real.sqrt (p^2 - q))^n = N - Real.sqrt (N^2 - q^n)) ∧ 
    ((p + Real.sqrt (p^2 - q))^n = N + Real.sqrt (N^2 - q^n)) :=
sorry

end NUMINAMATH_GPT_exists_N_for_sqrt_expressions_l1665_166590
