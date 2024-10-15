import Mathlib

namespace NUMINAMATH_GPT_spaghetti_tortellini_ratio_l525_52585

theorem spaghetti_tortellini_ratio (students_surveyed : ℕ)
                                    (spaghetti_lovers : ℕ)
                                    (tortellini_lovers : ℕ)
                                    (h1 : students_surveyed = 850)
                                    (h2 : spaghetti_lovers = 300)
                                    (h3 : tortellini_lovers = 200) :
  spaghetti_lovers / tortellini_lovers = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_spaghetti_tortellini_ratio_l525_52585


namespace NUMINAMATH_GPT_find_x_if_perpendicular_l525_52549

-- Definitions based on the conditions provided
structure Vector2 := (x : ℚ) (y : ℚ)

def a : Vector2 := ⟨2, 3⟩
def b (x : ℚ) : Vector2 := ⟨x, 4⟩

def dot_product (v1 v2 : Vector2) : ℚ := v1.x * v2.x + v1.y * v2.y

theorem find_x_if_perpendicular :
  ∀ x : ℚ, dot_product a (Vector2.mk (a.x - (b x).x) (a.y - (b x).y)) = 0 → x = 1/2 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_find_x_if_perpendicular_l525_52549


namespace NUMINAMATH_GPT_no_HCl_formed_l525_52540

-- Definitions
def NaCl_moles : Nat := 3
def HNO3_moles : Nat := 3
def HCl_moles : Nat := 0

-- Hypothetical reaction context
-- if the reaction would produce HCl
axiom hypothetical_reaction : (NaCl_moles = 3) → (HNO3_moles = 3) → (∃ h : Nat, h = 3)

-- Proof under normal conditions that no HCl is formed
theorem no_HCl_formed : (NaCl_moles = 3) → (HNO3_moles = 3) → HCl_moles = 0 := by
  intros hNaCl hHNO3
  sorry

end NUMINAMATH_GPT_no_HCl_formed_l525_52540


namespace NUMINAMATH_GPT_athletes_leave_rate_l525_52529

theorem athletes_leave_rate (R : ℝ) (h : 300 - 4 * R + 105 = 307) : R = 24.5 :=
  sorry

end NUMINAMATH_GPT_athletes_leave_rate_l525_52529


namespace NUMINAMATH_GPT_octagon_diagonals_l525_52550

def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

theorem octagon_diagonals : number_of_diagonals 8 = 20 :=
by
  sorry

end NUMINAMATH_GPT_octagon_diagonals_l525_52550


namespace NUMINAMATH_GPT_max_value_of_expression_l525_52599

theorem max_value_of_expression (m : ℝ) : 4 - |2 - m| ≤ 4 :=
by 
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l525_52599


namespace NUMINAMATH_GPT_problem_statement_b_problem_statement_c_l525_52535

def clubsuit (x y : ℝ) : ℝ := |x - y + 3|

theorem problem_statement_b :
  ∃ x y : ℝ, 3 * (clubsuit x y) ≠ clubsuit (3 * x + 3) (3 * y + 3) := by
  sorry

theorem problem_statement_c :
  ∃ x : ℝ, clubsuit x (-3) ≠ x := by
  sorry

end NUMINAMATH_GPT_problem_statement_b_problem_statement_c_l525_52535


namespace NUMINAMATH_GPT_b_2030_is_5_l525_52589

def seq (b : ℕ → ℚ) : Prop :=
  b 1 = 4 ∧ b 2 = 5 ∧ ∀ n ≥ 3, b (n + 1) = b n / b (n - 1)

theorem b_2030_is_5 (b : ℕ → ℚ) (h : seq b) : 
  b 2030 = 5 :=
sorry

end NUMINAMATH_GPT_b_2030_is_5_l525_52589


namespace NUMINAMATH_GPT_marble_count_l525_52512

noncomputable def total_marbles (blue red white: ℕ) : ℕ := blue + red + white

theorem marble_count (W : ℕ) (h_prob : (9 + W) / (6 + 9 + W : ℝ) = 0.7) : 
  total_marbles 6 9 W = 20 :=
by
  sorry

end NUMINAMATH_GPT_marble_count_l525_52512


namespace NUMINAMATH_GPT_probability_of_first_spade_or_ace_and_second_ace_l525_52510

theorem probability_of_first_spade_or_ace_and_second_ace :
  let deck_size := 52
  let aces := 4
  let spades := 13
  let non_ace_spades := spades - 1
  let other_aces := aces - 1
  let prob_first_non_ace_spade := non_ace_spades / deck_size
  let prob_first_ace_not_spade := (aces - 1) / deck_size
  let prob_first_ace_spade := 1 / deck_size
  let prob_second_ace_after_non_ace_spade := aces / (deck_size - 1)
  let prob_second_ace_after_ace_not_spade := (aces - 1) / (deck_size - 1)
  let prob_second_ace_after_ace_spade := (aces - 1) / (deck_size - 1)
  ((prob_first_non_ace_spade * prob_second_ace_after_non_ace_spade) +
   (prob_first_ace_not_spade * prob_second_ace_after_ace_not_spade) +
   (prob_first_ace_spade * prob_second_ace_after_ace_spade)) = 5 / 221 :=
by
  let deck_size := 52
  let aces := 4
  let spades := 13
  let non_ace_spades := spades - 1
  let other_aces := aces - 1
  let prob_first_non_ace_spade := non_ace_spades / deck_size
  let prob_first_ace_not_spade := (aces - 1) / deck_size
  let prob_first_ace_spade := 1 / deck_size
  let prob_second_ace_after_non_ace_spade := aces / (deck_size - 1)
  let prob_second_ace_after_ace_not_spade := (aces - 1) / (deck_size - 1)
  let prob_second_ace_after_ace_spade := (aces - 1) / (deck_size - 1)
  sorry

end NUMINAMATH_GPT_probability_of_first_spade_or_ace_and_second_ace_l525_52510


namespace NUMINAMATH_GPT_solution_for_x_l525_52557

theorem solution_for_x (t : ℤ) :
  ∃ x : ℤ, (∃ (k1 k2 k3 : ℤ), 
    (2 * x + 1 = 3 * k1) ∧ (3 * x + 1 = 4 * k2) ∧ (4 * x + 1 = 5 * k3)) :=
  sorry

end NUMINAMATH_GPT_solution_for_x_l525_52557


namespace NUMINAMATH_GPT_evaluate_polynomial_at_2_l525_52504

def polynomial (x : ℝ) := x^2 + 5*x - 14

theorem evaluate_polynomial_at_2 : polynomial 2 = 0 := by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_2_l525_52504


namespace NUMINAMATH_GPT_sum_of_hundreds_and_tens_digits_of_product_l525_52509

def hundreds_digit (n : ℕ) : ℕ := (n / 100) % 10
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def seq_num (a : ℕ) (x : ℕ) := List.foldr (λ _ acc => acc * 1000 + a) 0 (List.range x)

noncomputable def num_a := seq_num 707 101
noncomputable def num_b := seq_num 909 101

noncomputable def product := num_a * num_b

theorem sum_of_hundreds_and_tens_digits_of_product :
  hundreds_digit product + tens_digit product = 8 := by
  sorry

end NUMINAMATH_GPT_sum_of_hundreds_and_tens_digits_of_product_l525_52509


namespace NUMINAMATH_GPT_price_per_working_game_l525_52513

theorem price_per_working_game 
  (total_games : ℕ) (non_working_games : ℕ) (total_earnings : ℕ)
  (h1 : total_games = 16) (h2 : non_working_games = 8) (h3 : total_earnings = 56) :
  total_earnings / (total_games - non_working_games) = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_price_per_working_game_l525_52513


namespace NUMINAMATH_GPT_mary_and_joan_marbles_l525_52556

theorem mary_and_joan_marbles : 9 + 3 = 12 :=
by
  rfl

end NUMINAMATH_GPT_mary_and_joan_marbles_l525_52556


namespace NUMINAMATH_GPT_sum_of_squares_eq_ten_l525_52577

noncomputable def x1 : ℝ := Real.sqrt 3 - Real.sqrt 2
noncomputable def x2 : ℝ := Real.sqrt 3 + Real.sqrt 2

theorem sum_of_squares_eq_ten : x1^2 + x2^2 = 10 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_eq_ten_l525_52577


namespace NUMINAMATH_GPT_determine_m_l525_52552

variable (A B : Set ℝ)
variable (m : ℝ)

theorem determine_m (hA : A = {-1, 3, m}) (hB : B = {3, 4}) (h_inter : B ∩ A = B) : m = 4 :=
sorry

end NUMINAMATH_GPT_determine_m_l525_52552


namespace NUMINAMATH_GPT_total_votes_l525_52574

theorem total_votes (V : ℝ) (h1 : 0.35 * V + (0.35 * V + 1650) = V) : V = 5500 := 
by 
  sorry

end NUMINAMATH_GPT_total_votes_l525_52574


namespace NUMINAMATH_GPT_dandelion_dog_puffs_l525_52523

theorem dandelion_dog_puffs :
  let original_puffs := 40
  let mom_puffs := 3
  let sister_puffs := 3
  let grandmother_puffs := 5
  let friends := 3
  let puffs_per_friend := 9
  original_puffs - (mom_puffs + sister_puffs + grandmother_puffs + friends * puffs_per_friend) = 2 :=
by
  sorry

end NUMINAMATH_GPT_dandelion_dog_puffs_l525_52523


namespace NUMINAMATH_GPT_find_h_l525_52576

-- Define the quadratic expression
def quadratic_expr (x : ℝ) : ℝ := 3 * x^2 + 9 * x + 20

-- Main statement
theorem find_h : 
  ∃ (a k h : ℝ), (∀ x : ℝ, quadratic_expr x = a * (x - h)^2 + k) ∧ h = -3/2 := 
by
  sorry

end NUMINAMATH_GPT_find_h_l525_52576


namespace NUMINAMATH_GPT_solution_l525_52569

noncomputable def problem (a b c x y z : ℝ) :=
  11 * x + b * y + c * z = 0 ∧
  a * x + 19 * y + c * z = 0 ∧
  a * x + b * y + 37 * z = 0 ∧
  a ≠ 11 ∧
  x ≠ 0

theorem solution (a b c x y z : ℝ) (h : problem a b c x y z) :
  (a / (a - 11)) + (b / (b - 19)) + (c / (c - 37)) = 1 :=
sorry

end NUMINAMATH_GPT_solution_l525_52569


namespace NUMINAMATH_GPT_num_adults_l525_52545

-- Definitions of the conditions
def num_children : Nat := 11
def child_ticket_cost : Nat := 4
def adult_ticket_cost : Nat := 8
def total_cost : Nat := 124

-- The proof problem statement
theorem num_adults (A : Nat) 
  (h1 : total_cost = num_children * child_ticket_cost + A * adult_ticket_cost) : 
  A = 10 := 
by
  sorry

end NUMINAMATH_GPT_num_adults_l525_52545


namespace NUMINAMATH_GPT_dale_slices_of_toast_l525_52543

theorem dale_slices_of_toast
  (slice_cost : ℤ) (egg_cost : ℤ)
  (dale_eggs : ℤ) (andrew_slices : ℤ) (andrew_eggs : ℤ)
  (total_cost : ℤ)
  (cost_eq : slice_cost = 1)
  (egg_cost_eq : egg_cost = 3)
  (dale_eggs_eq : dale_eggs = 2)
  (andrew_slices_eq : andrew_slices = 1)
  (andrew_eggs_eq : andrew_eggs = 2)
  (total_cost_eq : total_cost = 15)
  :
  ∃ T : ℤ, (slice_cost * T + egg_cost * dale_eggs) + (slice_cost * andrew_slices + egg_cost * andrew_eggs) = total_cost ∧ T = 2 :=
by
  sorry

end NUMINAMATH_GPT_dale_slices_of_toast_l525_52543


namespace NUMINAMATH_GPT_brick_length_is_20_cm_l525_52551

-- Define the conditions given in the problem
def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def num_bricks : ℕ := 20000
def brick_width_cm : ℝ := 10
def total_area_cm2 : ℝ := 4000000

-- Define the goal to prove that the length of each brick is 20 cm
theorem brick_length_is_20_cm :
  (total_area_cm2 = num_bricks * (brick_width_cm * length)) → (length = 20) :=
by
  -- Assume the given conditions
  sorry

end NUMINAMATH_GPT_brick_length_is_20_cm_l525_52551


namespace NUMINAMATH_GPT_ellipse_eccentricity_l525_52586

theorem ellipse_eccentricity (k : ℝ) : 
  (∀ x y : ℝ, x^2 / (k + 8) + y^2 / 9 = 1) ∧ (∃ e : ℝ, e = 1 / 2) → 
  (k = 4 ∨ k = -5 / 4) := sorry

end NUMINAMATH_GPT_ellipse_eccentricity_l525_52586


namespace NUMINAMATH_GPT_problem_equivalent_proof_l525_52568

theorem problem_equivalent_proof (a : ℝ) (h : a / 2 - 2 / a = 5) :
  (a^8 - 256) / (16 * a^4) * (2 * a / (a^2 + 4)) = 81 :=
sorry

end NUMINAMATH_GPT_problem_equivalent_proof_l525_52568


namespace NUMINAMATH_GPT_fifth_term_is_2_11_over_60_l525_52594

noncomputable def fifth_term_geo_prog (a₁ a₂ a₃ : ℝ) (r : ℝ) : ℝ :=
  a₃ * r^2

theorem fifth_term_is_2_11_over_60
  (a₁ a₂ a₃ : ℝ)
  (h₁ : a₁ = 2^(1/4))
  (h₂ : a₂ = 2^(1/5))
  (h₃ : a₃ = 2^(1/6))
  (r : ℝ)
  (common_ratio : r = a₂ / a₁) :
  fifth_term_geo_prog a₁ a₂ a₃ r = 2^(11/60) :=
by
  sorry

end NUMINAMATH_GPT_fifth_term_is_2_11_over_60_l525_52594


namespace NUMINAMATH_GPT_edward_initial_amount_l525_52500

theorem edward_initial_amount (spent received final_amount : ℤ) 
  (h_spent : spent = 17) 
  (h_received : received = 10) 
  (h_final : final_amount = 7) : 
  ∃ initial_amount : ℤ, (initial_amount - spent + received = final_amount) ∧ (initial_amount = 14) :=
by
  sorry

end NUMINAMATH_GPT_edward_initial_amount_l525_52500


namespace NUMINAMATH_GPT_inverse_proposition_vertical_angles_false_l525_52541

-- Define the statement "Vertical angles are equal"
def vertical_angles_equal (α β : ℝ) : Prop :=
  α = β

-- Define the inverse proposition
def inverse_proposition_vertical_angles : Prop :=
  ∀ α β : ℝ, α = β → vertical_angles_equal α β

-- The proof goal
theorem inverse_proposition_vertical_angles_false : ¬inverse_proposition_vertical_angles :=
by
  sorry

end NUMINAMATH_GPT_inverse_proposition_vertical_angles_false_l525_52541


namespace NUMINAMATH_GPT_pears_count_l525_52570

theorem pears_count (A F P : ℕ)
  (hA : A = 12)
  (hF : F = 4 * 12 + 3)
  (hP : P = F - A) :
  P = 39 := by
  sorry

end NUMINAMATH_GPT_pears_count_l525_52570


namespace NUMINAMATH_GPT_stock_investment_decrease_l525_52507

theorem stock_investment_decrease (x : ℝ) (d1 d2 : ℝ) (hx : x > 0)
  (increase : x * 1.30 = 1.30 * x) :
  d1 = 20 ∧ d2 = 3.85 → 1.30 * (1 - d1 / 100) * (1 - d2 / 100) = 1 := by
  sorry

end NUMINAMATH_GPT_stock_investment_decrease_l525_52507


namespace NUMINAMATH_GPT_company_p_employees_in_january_l525_52584

-- Conditions
def employees_in_december (january_employees : ℝ) : ℝ := january_employees + 0.15 * january_employees

theorem company_p_employees_in_january (january_employees : ℝ) :
  employees_in_december january_employees = 490 → january_employees = 426 :=
by
  intro h
  -- The proof steps will be filled here.
  sorry

end NUMINAMATH_GPT_company_p_employees_in_january_l525_52584


namespace NUMINAMATH_GPT_irrational_root_exists_l525_52561

theorem irrational_root_exists 
  (a b c d : ℤ)
  (h_poly : ∀ x : ℚ, a * x^3 + b * x^2 + c * x + d ≠ 0) 
  (h_odd : a * d % 2 = 1) 
  (h_even : b * c % 2 = 0) : 
  ∃ x : ℚ, ¬ ∃ y : ℚ, y ≠ x ∧ y ≠ x ∧ a * x^3 + b * x^2 + c * x + d = 0 :=
sorry

end NUMINAMATH_GPT_irrational_root_exists_l525_52561


namespace NUMINAMATH_GPT_exists_infinite_n_for_multiple_of_prime_l525_52592

theorem exists_infinite_n_for_multiple_of_prime (p : ℕ) (hp : Nat.Prime p) :
  ∃ᶠ n in at_top, 2 ^ n - n ≡ 0 [MOD p] :=
by
  sorry

end NUMINAMATH_GPT_exists_infinite_n_for_multiple_of_prime_l525_52592


namespace NUMINAMATH_GPT_perimeter_result_l525_52524

-- Define the side length of the square
def side_length : ℕ := 100

-- Define the dimensions of the rectangle
def rectangle_dim1 : ℕ := side_length
def rectangle_dim2 : ℕ := side_length / 2

-- Perimeter calculation based on the arrangement
def perimeter : ℕ :=
  3 * rectangle_dim1 + 4 * rectangle_dim2

-- The statement of the problem
theorem perimeter_result :
  perimeter = 500 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_result_l525_52524


namespace NUMINAMATH_GPT_shara_shells_final_count_l525_52519

def initial_shells : ℕ := 20
def first_vacation_found : ℕ := 5 * 3 + 6
def first_vacation_lost : ℕ := 4
def second_vacation_found : ℕ := 4 * 2 + 7
def second_vacation_gifted : ℕ := 3
def third_vacation_found : ℕ := 8 + 4 + 3 * 2
def third_vacation_misplaced : ℕ := 5

def total_shells_after_first_vacation : ℕ :=
  initial_shells + first_vacation_found - first_vacation_lost

def total_shells_after_second_vacation : ℕ :=
  total_shells_after_first_vacation + second_vacation_found - second_vacation_gifted

def total_shells_after_third_vacation : ℕ :=
  total_shells_after_second_vacation + third_vacation_found - third_vacation_misplaced

theorem shara_shells_final_count : total_shells_after_third_vacation = 62 := by
  sorry

end NUMINAMATH_GPT_shara_shells_final_count_l525_52519


namespace NUMINAMATH_GPT_jennifer_fish_tank_problem_l525_52567

theorem jennifer_fish_tank_problem :
  let built_tanks := 3
  let fish_per_built_tank := 15
  let planned_tanks := 3
  let fish_per_planned_tank := 10
  let total_built_fish := built_tanks * fish_per_built_tank
  let total_planned_fish := planned_tanks * fish_per_planned_tank
  let total_fish := total_built_fish + total_planned_fish
  total_fish = 75 := by
    let built_tanks := 3
    let fish_per_built_tank := 15
    let planned_tanks := 3
    let fish_per_planned_tank := 10
    let total_built_fish := built_tanks * fish_per_built_tank
    let total_planned_fish := planned_tanks * fish_per_planned_tank
    let total_fish := total_built_fish + total_planned_fish
    have h₁ : total_built_fish = 45 := by sorry
    have h₂ : total_planned_fish = 30 := by sorry
    have h₃ : total_fish = 75 := by sorry
    exact h₃

end NUMINAMATH_GPT_jennifer_fish_tank_problem_l525_52567


namespace NUMINAMATH_GPT_highest_possible_value_l525_52591

theorem highest_possible_value 
  (t q r1 r2 : ℝ)
  (h_eq : r1 + r2 = t)
  (h_cond : ∀ n : ℕ, n > 0 → r1^n + r2^n = t) :
  t = 2 → q = 1 → 
  r1 = 1 → r2 = 1 →
  (1 / r1^1004 + 1 / r2^1004 = 2) :=
by
  intros h_t h_q h_r1 h_r2
  rw [h_r1, h_r2]
  norm_num

end NUMINAMATH_GPT_highest_possible_value_l525_52591


namespace NUMINAMATH_GPT_theater_ticket_area_l525_52562

theorem theater_ticket_area
  (P width : ℕ)
  (hP : P = 28)
  (hwidth : width = 6)
  (length : ℕ)
  (hlength : 2 * (length + width) = P) :
  length * width = 48 :=
by
  sorry

end NUMINAMATH_GPT_theater_ticket_area_l525_52562


namespace NUMINAMATH_GPT_smallest_two_digit_integer_l525_52572

-- Define the problem parameters and condition
theorem smallest_two_digit_integer (n : ℕ) (a b : ℕ) 
  (h1 : n = 10 * a + b) 
  (h2 : 1 ≤ a) (h3 : a ≤ 9) (h4 : 0 ≤ b) (h5 : b ≤ 9) 
  (h6 : 19 * a = 8 * b + 3) : 
  n = 12 :=
sorry

end NUMINAMATH_GPT_smallest_two_digit_integer_l525_52572


namespace NUMINAMATH_GPT_delta_value_l525_52506

theorem delta_value : ∃ Δ : ℤ, 4 * (-3) = Δ + 5 ∧ Δ = -17 := 
by
  use -17
  sorry

end NUMINAMATH_GPT_delta_value_l525_52506


namespace NUMINAMATH_GPT_smallest_n_l525_52516

theorem smallest_n (n : ℕ) (h : ↑n > 0 ∧ (Real.sqrt (↑n) - Real.sqrt (↑n - 1)) < 0.02) : n = 626 := 
by
  sorry

end NUMINAMATH_GPT_smallest_n_l525_52516


namespace NUMINAMATH_GPT_neg_sqrt_two_sq_l525_52593

theorem neg_sqrt_two_sq : (- Real.sqrt 2) ^ 2 = 2 := 
by
  sorry

end NUMINAMATH_GPT_neg_sqrt_two_sq_l525_52593


namespace NUMINAMATH_GPT_book_total_pages_l525_52555

theorem book_total_pages (n : ℕ) (h1 : 5 * n / 8 - 3 * n / 7 = 33) : n = n :=
by 
  -- We skip the proof as instructed
  sorry

end NUMINAMATH_GPT_book_total_pages_l525_52555


namespace NUMINAMATH_GPT_usual_time_is_120_l525_52578

variable (S T : ℕ) (h1 : 0 < S) (h2 : 0 < T)
variable (h3 : (4 : ℚ) / 3 = 1 + (40 : ℚ) / T)

theorem usual_time_is_120 : T = 120 := by
  sorry

end NUMINAMATH_GPT_usual_time_is_120_l525_52578


namespace NUMINAMATH_GPT_number_of_meetings_l525_52534

noncomputable def selena_radius : ℝ := 70
noncomputable def bashar_radius : ℝ := 80
noncomputable def selena_speed : ℝ := 200
noncomputable def bashar_speed : ℝ := 240
noncomputable def active_time_together : ℝ := 30

noncomputable def selena_circumference : ℝ := 2 * Real.pi * selena_radius
noncomputable def bashar_circumference : ℝ := 2 * Real.pi * bashar_radius

noncomputable def selena_angular_speed : ℝ := (selena_speed / selena_circumference) * (2 * Real.pi)
noncomputable def bashar_angular_speed : ℝ := (bashar_speed / bashar_circumference) * (2 * Real.pi)

noncomputable def relative_angular_speed : ℝ := selena_angular_speed + bashar_angular_speed
noncomputable def time_to_meet_once : ℝ := (2 * Real.pi) / relative_angular_speed

theorem number_of_meetings : Int := 
    ⌊active_time_together / time_to_meet_once⌋

example : number_of_meetings = 21 := by
  sorry

end NUMINAMATH_GPT_number_of_meetings_l525_52534


namespace NUMINAMATH_GPT_relation_of_variables_l525_52508

theorem relation_of_variables (x y z w : ℝ) 
  (h : (x + 2 * y) / (2 * y + 3 * z) = (3 * z + 4 * w) / (4 * w + x)) : 
  (x = 3 * z) ∨ (x + 2 * y + 4 * w + 3 * z = 0) := 
by
  sorry

end NUMINAMATH_GPT_relation_of_variables_l525_52508


namespace NUMINAMATH_GPT_avg_age_difference_l525_52595

noncomputable def team_size : ℕ := 11
noncomputable def avg_age_team : ℝ := 26
noncomputable def wicket_keeper_extra_age : ℝ := 3
noncomputable def num_remaining_players : ℕ := 9
noncomputable def avg_age_remaining_players : ℝ := 23

theorem avg_age_difference :
  avg_age_team - avg_age_remaining_players = 0.33 := 
by
  sorry

end NUMINAMATH_GPT_avg_age_difference_l525_52595


namespace NUMINAMATH_GPT_exists_k_not_divisible_l525_52558

theorem exists_k_not_divisible (a b c n : ℤ) (hn : n ≥ 3) :
  ∃ k : ℤ, ¬(n ∣ (k + a)) ∧ ¬(n ∣ (k + b)) ∧ ¬(n ∣ (k + c)) :=
sorry

end NUMINAMATH_GPT_exists_k_not_divisible_l525_52558


namespace NUMINAMATH_GPT_machine_a_produces_50_parts_in_10_minutes_l525_52563

/-- 
Given that machine A produces parts twice as fast as machine B,
and machine B produces 100 parts in 40 minutes at a constant rate,
prove that machine A produces 50 parts in 10 minutes.
-/
theorem machine_a_produces_50_parts_in_10_minutes :
  (machine_b_rate : ℕ → ℕ) → 
  (machine_a_rate : ℕ → ℕ) →
  (htwice_as_fast: ∀ t, machine_a_rate t = (2 * machine_b_rate t)) →
  (hconstant_rate_b: ∀ t1 t2, t1 * machine_b_rate t2 = 100 * t2 / 40)→
  machine_a_rate 10 = 50 :=
by
  sorry

end NUMINAMATH_GPT_machine_a_produces_50_parts_in_10_minutes_l525_52563


namespace NUMINAMATH_GPT_area_of_square_l525_52582

def side_length (x : ℕ) : ℕ := 3 * x - 12

def side_length_alt (x : ℕ) : ℕ := 18 - 2 * x

theorem area_of_square (x : ℕ) (h : 3 * x - 12 = 18 - 2 * x) : (side_length x) ^ 2 = 36 :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_l525_52582


namespace NUMINAMATH_GPT_value_of_a_minus_3_l525_52518

variable {α : Type*} [Field α] (f : α → α) (a : α)

-- Conditions
variable (h_invertible : Function.Injective f)
variable (h_fa : f a = 3)
variable (h_f3 : f 3 = 6)

-- Statement to prove
theorem value_of_a_minus_3 : a - 3 = -2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_minus_3_l525_52518


namespace NUMINAMATH_GPT_mary_can_keep_warm_l525_52527

def sticks_from_chairs (n_c : ℕ) (c_1 : ℕ) : ℕ := n_c * c_1
def sticks_from_tables (n_t : ℕ) (t_1 : ℕ) : ℕ := n_t * t_1
def sticks_from_cabinets (n_cb : ℕ) (cb_1 : ℕ) : ℕ := n_cb * cb_1
def sticks_from_stools (n_s : ℕ) (s_1 : ℕ) : ℕ := n_s * s_1

def total_sticks (n_c n_t n_cb n_s c_1 t_1 cb_1 s_1 : ℕ) : ℕ :=
  sticks_from_chairs n_c c_1
  + sticks_from_tables n_t t_1 
  + sticks_from_cabinets n_cb cb_1 
  + sticks_from_stools n_s s_1

noncomputable def hours (total_sticks r : ℕ) : ℕ :=
  total_sticks / r

theorem mary_can_keep_warm (n_c n_t n_cb n_s : ℕ) (c_1 t_1 cb_1 s_1 r : ℕ) :
  n_c = 25 → n_t = 12 → n_cb = 5 → n_s = 8 → c_1 = 8 → t_1 = 12 → cb_1 = 16 → s_1 = 3 → r = 7 →
  hours (total_sticks n_c n_t n_cb n_s c_1 t_1 cb_1 s_1) r = 64 :=
by
  intros h_nc h_nt h_ncb h_ns h_c1 h_t1 h_cb1 h_s1 h_r
  sorry

end NUMINAMATH_GPT_mary_can_keep_warm_l525_52527


namespace NUMINAMATH_GPT_option_a_correct_option_b_incorrect_option_c_incorrect_option_d_incorrect_l525_52528

theorem option_a_correct (a : ℝ) : 2 * a^2 - 3 * a^2 = - a^2 :=
by
  sorry

theorem option_b_incorrect : (-3)^2 ≠ 6 :=
by
  sorry

theorem option_c_incorrect (a : ℝ) : 6 * a^3 + 4 * a^4 ≠ 10 * a^7 :=
by
  sorry

theorem option_d_incorrect (a b : ℝ) : 3 * a^2 * b - 3 * b^2 * a ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_option_a_correct_option_b_incorrect_option_c_incorrect_option_d_incorrect_l525_52528


namespace NUMINAMATH_GPT_measure_of_angle_4_l525_52522

theorem measure_of_angle_4 
  (angle1 angle2 angle3 : ℝ)
  (h1 : angle1 = 100)
  (h2 : angle2 = 60)
  (h3 : angle3 = 90)
  (h_sum : angle1 + angle2 + angle3 + angle4 = 360) : 
  angle4 = 110 :=
by
  sorry

end NUMINAMATH_GPT_measure_of_angle_4_l525_52522


namespace NUMINAMATH_GPT_evaluate_expression_l525_52538

theorem evaluate_expression : (3 + 2) * (3^2 + 2^2) * (3^4 + 2^4) = 6255 := sorry

end NUMINAMATH_GPT_evaluate_expression_l525_52538


namespace NUMINAMATH_GPT_not_multiple_of_121_l525_52548

theorem not_multiple_of_121 (n : ℤ) : ¬ ∃ k : ℤ, n^2 + 2*n + 12 = 121*k := 
sorry

end NUMINAMATH_GPT_not_multiple_of_121_l525_52548


namespace NUMINAMATH_GPT_find_numbers_l525_52531

def seven_digit_number (n : ℕ) : Prop := 10^6 ≤ n ∧ n < 10^7

theorem find_numbers (x y : ℕ) (hx: seven_digit_number x) (hy: seven_digit_number y) :
  10^7 * x + y = 3 * x * y → x = 1666667 ∧ y = 3333334 :=
by
  sorry

end NUMINAMATH_GPT_find_numbers_l525_52531


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l525_52565

theorem line_passes_through_fixed_point (m n : ℝ) (h : m + 2 * n - 1 = 0) :
  mx + 3 * y + n = 0 → (x, y) = (1/2, -1/6) :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l525_52565


namespace NUMINAMATH_GPT_pairs_of_managers_refusing_l525_52517

theorem pairs_of_managers_refusing (h_comb : (Nat.choose 8 4) = 70) (h_restriction : 55 = 70 - n * (Nat.choose 6 2)) : n = 1 :=
by
  have h1 : Nat.choose 8 4 = 70 := h_comb
  have h2 : Nat.choose 6 2 = 15 := by sorry -- skipped calculation for (6 choose 2), which is 15
  have h3 : 55 = 70 - n * 15 := h_restriction
  sorry -- proof steps to show n = 1

end NUMINAMATH_GPT_pairs_of_managers_refusing_l525_52517


namespace NUMINAMATH_GPT_find_a_l525_52544

theorem find_a (a b : ℤ) (h : ∀ x, x^2 - x - 1 = 0 → ax^18 + bx^17 + 1 = 0) : a = 1597 :=
sorry

end NUMINAMATH_GPT_find_a_l525_52544


namespace NUMINAMATH_GPT_only_B_is_linear_system_l525_52526

def linear_equation (eq : String) : Prop := 
-- Placeholder for the actual definition
sorry 

def system_B_is_linear : Prop :=
  linear_equation "x + y = 2" ∧ linear_equation "x - y = 4"

theorem only_B_is_linear_system 
: (∀ (A B C D : Prop), 
       (A ↔ (linear_equation "3x + 4y = 6" ∧ linear_equation "5z - 6y = 4")) → 
       (B ↔ (linear_equation "x + y = 2" ∧ linear_equation "x - y = 4")) → 
       (C ↔ (linear_equation "x + y = 2" ∧ linear_equation "x^2 - y^2 = 8")) → 
       (D ↔ (linear_equation "x + y = 2" ∧ linear_equation "1/x - 1/y = 1/2")) → 
       (B ∧ ¬A ∧ ¬C ∧ ¬D))
:= 
sorry

end NUMINAMATH_GPT_only_B_is_linear_system_l525_52526


namespace NUMINAMATH_GPT_jordan_run_7_miles_in_112_div_3_minutes_l525_52530

noncomputable def time_for_steve (distance : ℝ) : ℝ := 36 / 4.5 * distance
noncomputable def jordan_initial_time (steve_time : ℝ) : ℝ := steve_time / 3
noncomputable def jordan_speed (distance time : ℝ) : ℝ := distance / time
noncomputable def adjusted_speed (speed : ℝ) : ℝ := speed * 0.9
noncomputable def running_time (distance speed : ℝ) : ℝ := distance / speed

theorem jordan_run_7_miles_in_112_div_3_minutes : running_time 7 ((jordan_speed 2.5 (jordan_initial_time (time_for_steve 4.5))) * 0.9) = 112 / 3 :=
by
  sorry

end NUMINAMATH_GPT_jordan_run_7_miles_in_112_div_3_minutes_l525_52530


namespace NUMINAMATH_GPT_overall_rate_of_profit_is_25_percent_l525_52532

def cost_price_A : ℕ := 50
def selling_price_A : ℕ := 70
def cost_price_B : ℕ := 80
def selling_price_B : ℕ := 100
def cost_price_C : ℕ := 150
def selling_price_C : ℕ := 180

def profit (sp cp : ℕ) : ℕ := sp - cp

def total_cost_price : ℕ := cost_price_A + cost_price_B + cost_price_C
def total_selling_price : ℕ := selling_price_A + selling_price_B + selling_price_C
def total_profit : ℕ := profit selling_price_A cost_price_A +
                        profit selling_price_B cost_price_B +
                        profit selling_price_C cost_price_C

def overall_rate_of_profit : ℚ := (total_profit : ℚ) / (total_cost_price : ℚ) * 100

theorem overall_rate_of_profit_is_25_percent :
  overall_rate_of_profit = 25 :=
by sorry

end NUMINAMATH_GPT_overall_rate_of_profit_is_25_percent_l525_52532


namespace NUMINAMATH_GPT_value_of_a_squared_b_plus_ab_squared_eq_4_l525_52525

variable (a b : ℝ)
variable (h_a : a = 2 + Real.sqrt 3)
variable (h_b : b = 2 - Real.sqrt 3)

theorem value_of_a_squared_b_plus_ab_squared_eq_4 :
  a^2 * b + a * b^2 = 4 := by
  sorry

end NUMINAMATH_GPT_value_of_a_squared_b_plus_ab_squared_eq_4_l525_52525


namespace NUMINAMATH_GPT_number_of_even_multiples_of_3_l525_52573

theorem number_of_even_multiples_of_3 :
  ∃ n, n = (198 - 6) / 6 + 1 := by
  sorry

end NUMINAMATH_GPT_number_of_even_multiples_of_3_l525_52573


namespace NUMINAMATH_GPT_pipe_A_fills_tank_in_28_hours_l525_52553

variable (A B C : ℝ)
-- Conditions
axiom h1 : C = 2 * B
axiom h2 : B = 2 * A
axiom h3 : A + B + C = 1 / 4

theorem pipe_A_fills_tank_in_28_hours : 1 / A = 28 := by
  -- proof omitted for the exercise
  sorry

end NUMINAMATH_GPT_pipe_A_fills_tank_in_28_hours_l525_52553


namespace NUMINAMATH_GPT_minimum_value_of_z_l525_52537

def z (x y : ℝ) : ℝ := 3 * x ^ 2 + 4 * y ^ 2 + 12 * x - 8 * y + 3 * x * y + 30

theorem minimum_value_of_z : ∃ (x y : ℝ), z x y = 8 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_z_l525_52537


namespace NUMINAMATH_GPT_maximum_ratio_l525_52536

-- Defining the conditions
def two_digit_positive_integer (n : ℕ) : Prop := n >= 10 ∧ n < 100

-- Proving the main theorem
theorem maximum_ratio (x y : ℕ) (hx : two_digit_positive_integer x) (hy : two_digit_positive_integer y) (h_sum : x + y = 100) : 
  ∃ m, m = 9 ∧ ∀ r, r = x / y → r ≤ 9 := sorry

end NUMINAMATH_GPT_maximum_ratio_l525_52536


namespace NUMINAMATH_GPT_range_of_f_log_gt_zero_l525_52542

open Real

noncomputable def f (x : ℝ) : ℝ := -- Placeholder function definition
  sorry

theorem range_of_f_log_gt_zero :
  (∀ x, f x = f (-x)) ∧
  (∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y) ∧
  (f (1 / 3) = 0) →
  {x : ℝ | f ((log x) / (log (1 / 8))) > 0} = 
    (Set.Ioo 0 (1 / 2) ∪ Set.Ioi 2) :=
  sorry

end NUMINAMATH_GPT_range_of_f_log_gt_zero_l525_52542


namespace NUMINAMATH_GPT_dan_baseball_cards_total_l525_52521

-- Define the initial conditions
def initial_baseball_cards : Nat := 97
def torn_baseball_cards : Nat := 8
def sam_bought_cards : Nat := 15
def alex_bought_fraction : Nat := 4
def gift_cards : Nat := 6

-- Define the number of cards    
def non_torn_baseball_cards : Nat := initial_baseball_cards - torn_baseball_cards
def remaining_after_sam : Nat := non_torn_baseball_cards - sam_bought_cards
def remaining_after_alex : Nat := remaining_after_sam - remaining_after_sam / alex_bought_fraction
def final_baseball_cards : Nat := remaining_after_alex + gift_cards

-- The theorem to prove 
theorem dan_baseball_cards_total : final_baseball_cards = 62 := by
  sorry

end NUMINAMATH_GPT_dan_baseball_cards_total_l525_52521


namespace NUMINAMATH_GPT_sum_of_three_largest_consecutive_numbers_l525_52596

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end NUMINAMATH_GPT_sum_of_three_largest_consecutive_numbers_l525_52596


namespace NUMINAMATH_GPT_part1_part2_l525_52597

def f (x : ℝ) := |x + 2|

theorem part1 (x : ℝ) : 2 * f x < 4 - |x - 1| ↔ -7/3 < x ∧ x < -1 :=
by sorry

theorem part2 (a : ℝ) (m n : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) :
  (∀ x, |x - a| - f x ≤ 1/m + 1/n) ↔ -6 ≤ a ∧ a ≤ 2 :=
by sorry

end NUMINAMATH_GPT_part1_part2_l525_52597


namespace NUMINAMATH_GPT_full_price_ticket_revenue_l525_52560

-- Given conditions
variable {f d p : ℕ}
variable (h1 : f + d = 160)
variable (h2 : f * p + d * (2 * p / 3) = 2800)

-- Goal: Prove the full-price ticket revenue is 1680.
theorem full_price_ticket_revenue : f * p = 1680 :=
sorry

end NUMINAMATH_GPT_full_price_ticket_revenue_l525_52560


namespace NUMINAMATH_GPT_ellipse_equation_1_ellipse_equation_2_l525_52575

-- Proof Problem 1
theorem ellipse_equation_1 (x y : ℝ) 
  (foci_condition : (x+2) * (x+2) + y*y + (x-2) * (x-2) + y*y = 36) :
  x^2 / 9 + y^2 / 5 = 1 :=
sorry

-- Proof Problem 2
theorem ellipse_equation_2 (x y : ℝ)
  (foci_condition : (x^2 + (y+5)^2 = 0) ∧ (x^2 + (y-5)^2 = 0))
  (point_on_ellipse : 3^2 / 15 + 4^2 / (15 + 25) = 1) :
  y^2 / 40 + x^2 / 15 = 1 :=
sorry

end NUMINAMATH_GPT_ellipse_equation_1_ellipse_equation_2_l525_52575


namespace NUMINAMATH_GPT_parabola_decreasing_m_geq_neg2_l525_52514

theorem parabola_decreasing_m_geq_neg2 (m : ℝ) :
  (∀ x ≥ 2, ∃ y, y = -5 * (x + m)^2 - 3 ∧ (∀ x1 y1, x1 ≥ 2 → y1 = -5 * (x1 + m)^2 - 3 → y1 ≤ y)) →
  m ≥ -2 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_parabola_decreasing_m_geq_neg2_l525_52514


namespace NUMINAMATH_GPT_calculate_expression_l525_52587

variable (x : ℝ)

def quadratic_condition : Prop := x^2 + x - 1 = 0

theorem calculate_expression (h : quadratic_condition x) : 2*x^3 + 3*x^2 - x = 1 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l525_52587


namespace NUMINAMATH_GPT_dye_jobs_scheduled_l525_52580

noncomputable def revenue_from_haircuts (n : ℕ) : ℕ := n * 30
noncomputable def revenue_from_perms (n : ℕ) : ℕ := n * 40
noncomputable def revenue_from_dye_jobs (n : ℕ) : ℕ := n * (60 - 10)
noncomputable def total_revenue (haircuts perms dye_jobs : ℕ) (tips : ℕ) : ℕ :=
  revenue_from_haircuts haircuts + revenue_from_perms perms + revenue_from_dye_jobs dye_jobs + tips

theorem dye_jobs_scheduled : 
  (total_revenue 4 1 dye_jobs 50 = 310) → (dye_jobs = 2) := 
by
  sorry

end NUMINAMATH_GPT_dye_jobs_scheduled_l525_52580


namespace NUMINAMATH_GPT_correct_units_l525_52588

def units_time := ["hour", "minute", "second"]
def units_mass := ["gram", "kilogram", "ton"]
def units_length := ["millimeter", "centimeter", "decimeter", "meter", "kilometer"]

theorem correct_units :
  (units_time = ["hour", "minute", "second"]) ∧
  (units_mass = ["gram", "kilogram", "ton"]) ∧
  (units_length = ["millimeter", "centimeter", "decimeter", "meter", "kilometer"]) :=
by
  -- Please provide the proof here
  sorry

end NUMINAMATH_GPT_correct_units_l525_52588


namespace NUMINAMATH_GPT_frac_mul_sub_eq_l525_52571

/-
  Theorem:
  The result of multiplying 2/9 by 4/5 and then subtracting 1/45 is equal to 7/45.
-/
theorem frac_mul_sub_eq :
  (2/9 * 4/5 - 1/45) = 7/45 :=
by
  sorry

end NUMINAMATH_GPT_frac_mul_sub_eq_l525_52571


namespace NUMINAMATH_GPT_shiela_bottles_l525_52501

theorem shiela_bottles (num_stars : ℕ) (stars_per_bottle : ℕ) (num_bottles : ℕ) 
  (h1 : num_stars = 45) (h2 : stars_per_bottle = 5) : num_bottles = 9 :=
sorry

end NUMINAMATH_GPT_shiela_bottles_l525_52501


namespace NUMINAMATH_GPT_linear_function_intersects_x_axis_at_2_0_l525_52598

theorem linear_function_intersects_x_axis_at_2_0
  (f : ℝ → ℝ)
  (h : ∀ x, f x = -x + 2) :
  ∃ x, f x = 0 ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_linear_function_intersects_x_axis_at_2_0_l525_52598


namespace NUMINAMATH_GPT_intersection_A_B_l525_52515

/-- Define the set A -/
def A : Set ℝ := { x | ∃ y, y = Real.log (2 - x) }

/-- Define the set B -/
def B : Set ℝ := { y | ∃ x, y = Real.sqrt x }

/-- Define the intersection of A and B and prove that it equals [0, 2) -/
theorem intersection_A_B : (A ∩ B) = { x | 0 ≤ x ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l525_52515


namespace NUMINAMATH_GPT_red_ball_probability_l525_52511

-- Definitions based on conditions
def numBallsA := 10
def redBallsA := 5
def greenBallsA := numBallsA - redBallsA

def numBallsBC := 10
def redBallsBC := 7
def greenBallsBC := numBallsBC - redBallsBC

def probSelectContainer := 1 / 3
def probRedBallA := redBallsA / numBallsA
def probRedBallBC := redBallsBC / numBallsBC

-- Theorem statement to be proved
theorem red_ball_probability : (probSelectContainer * probRedBallA) + (probSelectContainer * probRedBallBC) + (probSelectContainer * probRedBallBC) = 4 / 5 := 
sorry

end NUMINAMATH_GPT_red_ball_probability_l525_52511


namespace NUMINAMATH_GPT_product_xyz_l525_52559

noncomputable def xyz_value (x y z : ℝ) :=
  x * y * z

theorem product_xyz (x y z : ℝ) (h1 : x + 1 / y = 3) (h2 : y + 1 / z = 3) :
  xyz_value x y z = -1 :=
by
  sorry

end NUMINAMATH_GPT_product_xyz_l525_52559


namespace NUMINAMATH_GPT_points_on_line_eqdist_quadrants_l525_52520

theorem points_on_line_eqdist_quadrants :
  ∀ (x y : ℝ), 4 * x - 3 * y = 12 ∧ |x| = |y| → 
  (x > 0 ∧ y > 0 ∨ x > 0 ∧ y < 0) :=
by
  sorry

end NUMINAMATH_GPT_points_on_line_eqdist_quadrants_l525_52520


namespace NUMINAMATH_GPT_num_koi_fish_after_3_weeks_l525_52505

noncomputable def total_days : ℕ := 3 * 7

noncomputable def koi_fish_added_per_day : ℕ := 2
noncomputable def goldfish_added_per_day : ℕ := 5

noncomputable def total_fish_initial : ℕ := 280
noncomputable def goldfish_final : ℕ := 200

noncomputable def koi_fish_total_after_3_weeks : ℕ :=
  let koi_fish_added := koi_fish_added_per_day * total_days
  let goldfish_added := goldfish_added_per_day * total_days
  let goldfish_initial := goldfish_final - goldfish_added
  let koi_fish_initial := total_fish_initial - goldfish_initial
  koi_fish_initial + koi_fish_added

theorem num_koi_fish_after_3_weeks :
  koi_fish_total_after_3_weeks = 227 := by
  sorry

end NUMINAMATH_GPT_num_koi_fish_after_3_weeks_l525_52505


namespace NUMINAMATH_GPT_simplify_and_evaluate_l525_52564

theorem simplify_and_evaluate (m : ℝ) (h : m = 5) :
  (m + 2 - (5 / (m - 2))) / ((3 * m - m^2) / (m - 2)) = - (8 / 5) :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l525_52564


namespace NUMINAMATH_GPT_pq_plus_p_plus_q_eq_1_l525_52533

-- Define the polynomial
def poly (x : ℝ) : ℝ := x^4 - 6 * x - 1

-- Prove the target statement
theorem pq_plus_p_plus_q_eq_1 (p q : ℝ) (hpq : poly p = 0) (hq : poly q = 0) :
  p * q + p + q = 1 := by
  sorry

end NUMINAMATH_GPT_pq_plus_p_plus_q_eq_1_l525_52533


namespace NUMINAMATH_GPT_no_nat_pairs_satisfy_eq_l525_52583

theorem no_nat_pairs_satisfy_eq (a b : ℕ) : ¬ (2019 * a ^ 2018 = 2017 + b ^ 2016) :=
sorry

end NUMINAMATH_GPT_no_nat_pairs_satisfy_eq_l525_52583


namespace NUMINAMATH_GPT_investment_ratio_l525_52503

theorem investment_ratio (A B : ℕ) (hA : A = 12000) (hB : B = 12000) 
  (interest_A : ℕ := 11 * A / 100) (interest_B : ℕ := 9 * B / 100) 
  (total_interest : interest_A + interest_B = 2400) :
  A / B = 1 :=
by
  sorry

end NUMINAMATH_GPT_investment_ratio_l525_52503


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_factors_of_12_l525_52590

theorem sum_of_reciprocals_of_factors_of_12 :
  (1:ℚ) / 1 + 1 / 2 + 1 / 3 + 1 / 4 + 1 / 6 + 1 / 12 = 7 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_factors_of_12_l525_52590


namespace NUMINAMATH_GPT_other_root_of_equation_l525_52554

theorem other_root_of_equation (c : ℝ) (h : 3^2 - 5 * 3 + c = 0) : 
  ∃ x : ℝ, x ≠ 3 ∧ x^2 - 5 * x + c = 0 ∧ x = 2 := 
by 
  sorry

end NUMINAMATH_GPT_other_root_of_equation_l525_52554


namespace NUMINAMATH_GPT_problem_statement_l525_52546

theorem problem_statement (x y : ℝ) (h1 : 3 = 0.15 * x) (h2 : 3 = 0.25 * y) : x - y = 8 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l525_52546


namespace NUMINAMATH_GPT_minimum_production_quantity_l525_52539

-- Define the total cost function
def total_cost (x : ℝ) : ℝ := 3000 + 20 * x - 0.1 * x^2

-- Define the revenue function given the selling price per unit
def revenue (x : ℝ) : ℝ := 25 * x

-- Define the interval for x
def x_range (x : ℝ) : Prop := 0 < x ∧ x < 240

-- State the minimum production quantity required to avoid a loss
theorem minimum_production_quantity (x : ℝ) (h : x_range x) : 150 <= x :=
by
  -- Sorry replaces the detailed proof steps
  sorry

end NUMINAMATH_GPT_minimum_production_quantity_l525_52539


namespace NUMINAMATH_GPT_functional_eq_solution_l525_52502

theorem functional_eq_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) :
  ∀ x : ℝ, f x = x :=
sorry

end NUMINAMATH_GPT_functional_eq_solution_l525_52502


namespace NUMINAMATH_GPT_coexistence_of_properties_l525_52566

structure Trapezoid (α : Type _) [Field α] :=
(base1 base2 leg1 leg2 : α)
(height : α)

def isIsosceles {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
T.leg1 = T.leg2

def diagonalsPerpendicular {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
sorry  -- Define this property based on coordinate geometry or vector inner products

def heightsEqual {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
T.base1 = T.base2

def midsegmentEqualHeight {α : Type _} [Field α] (T : Trapezoid α) : Prop :=
(T.base1 + T.base2) / 2 = T.height

theorem coexistence_of_properties (α : Type _) [Field α] (T : Trapezoid α) :
  isIsosceles T → heightsEqual T → midsegmentEqualHeight T → True :=
by sorry

end NUMINAMATH_GPT_coexistence_of_properties_l525_52566


namespace NUMINAMATH_GPT_nine_digit_divisible_by_11_l525_52547

theorem nine_digit_divisible_by_11 (m : ℕ) (k : ℤ) (h1 : 8 + 4 + m + 6 + 8 = 26 + m)
(h2 : 5 + 2 + 7 + 1 = 15)
(h3 : 26 + m - 15 = 11 + m)
(h4 : 11 + m = 11 * k) :
m = 0 := by
  sorry

end NUMINAMATH_GPT_nine_digit_divisible_by_11_l525_52547


namespace NUMINAMATH_GPT_fourth_power_sqrt_eq_256_l525_52579

theorem fourth_power_sqrt_eq_256 (x : ℝ) (h : (x^(1/2))^4 = 256) : x = 16 := by sorry

end NUMINAMATH_GPT_fourth_power_sqrt_eq_256_l525_52579


namespace NUMINAMATH_GPT_square_area_from_circle_l525_52581

-- Define the conditions for the circle's equation
def circle_equation (x y : ℝ) : Prop :=
  2 * x^2 = -2 * y^2 + 8 * x - 8 * y + 28 

-- State the main theorem to prove the area of the square
theorem square_area_from_circle (x y : ℝ) (h : circle_equation x y) :
  ∃ s : ℝ, s^2 = 88 :=
sorry

end NUMINAMATH_GPT_square_area_from_circle_l525_52581
