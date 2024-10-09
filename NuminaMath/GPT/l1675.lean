import Mathlib

namespace train_speed_l1675_167570

theorem train_speed (L1 L2: ℕ) (V2: ℕ) (T: ℕ) (V1: ℕ) : 
  L1 = 120 -> 
  L2 = 280 -> 
  V2 = 30 -> 
  T = 20 -> 
  (L1 + L2) * 18 = (V1 + V2) * T * 100 -> 
  V1 = 42 := 
by
  intros h1 h2 h3 h4 h5
  sorry

end train_speed_l1675_167570


namespace remainder_5_pow_2023_mod_11_l1675_167515

theorem remainder_5_pow_2023_mod_11 : (5^2023) % 11 = 4 :=
by
  have h1 : 5^2 % 11 = 25 % 11 := sorry
  have h2 : 25 % 11 = 3 := sorry
  have h3 : (3^5) % 11 = 1 := sorry
  have h4 : 3^1011 % 11 = ((3^5)^202 * 3) % 11 := sorry
  have h5 : ((3^5)^202 * 3) % 11 = (1^202 * 3) % 11 := sorry
  have h6 : (1^202 * 3) % 11 = 3 % 11 := sorry
  have h7 : (5^2023) % 11 = (3 * 5) % 11 := sorry
  have h8 : (3 * 5) % 11 = 15 % 11 := sorry
  have h9 : 15 % 11 = 4 := sorry
  exact h9

end remainder_5_pow_2023_mod_11_l1675_167515


namespace work_fraction_left_l1675_167580

theorem work_fraction_left (A_days B_days : ℕ) (work_days : ℕ)
  (hA : A_days = 15) (hB : B_days = 20) (h_work : work_days = 3) :
  1 - (work_days * ((1 / A_days) + (1 / B_days))) = 13 / 20 :=
by
  rw [hA, hB, h_work]
  simp
  sorry

end work_fraction_left_l1675_167580


namespace trey_more_turtles_than_kristen_l1675_167556

theorem trey_more_turtles_than_kristen (kristen_turtles : ℕ) 
  (H1 : kristen_turtles = 12) 
  (H2 : ∀ kris_turtles, kris_turtles = (1 / 4) * kristen_turtles)
  (H3 : ∀ kris_turtles trey_turtles, trey_turtles = 7 * kris_turtles) :
  ∃ trey_turtles, trey_turtles - kristen_turtles = 9 :=
by {
  sorry
}

end trey_more_turtles_than_kristen_l1675_167556


namespace necessary_but_not_sufficient_l1675_167561

-- Define the quadratic equation
def quadratic_eq (a : ℝ) (x : ℝ) : ℝ := x^2 + 2 * x + a

-- State the necessary but not sufficient condition proof statement
theorem necessary_but_not_sufficient (a : ℝ) :
  (∃ x y : ℝ, quadratic_eq a x = 0 ∧ quadratic_eq a y = 0 ∧ x > 0 ∧ y < 0) → a < 1 :=
sorry

end necessary_but_not_sufficient_l1675_167561


namespace range_of_a_l1675_167526

noncomputable def log_a (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 1 → log_a a (2 - a * x) < log_a a (2 - a * (x / 2))) →
  1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l1675_167526


namespace find_f_of_f_l1675_167572

noncomputable def f (x : ℝ) : ℝ :=
if x = 0 then 0 else (4 * x + 1 - 2 / x) / 3

theorem find_f_of_f (h : ∀ x : ℝ, x ≠ 0 → f x + 2 * f (1 / x) = 2 * x + 1) : 
  f 2 = -1/3 :=
sorry

end find_f_of_f_l1675_167572


namespace Rebecca_tips_calculation_l1675_167591

def price_haircut : ℤ := 30
def price_perm : ℤ := 40
def price_dye_job : ℤ := 60
def cost_hair_dye_box : ℤ := 10
def num_haircuts : ℕ := 4
def num_perms : ℕ := 1
def num_dye_jobs : ℕ := 2
def total_end_day : ℤ := 310

noncomputable def total_service_earnings : ℤ := 
  num_haircuts * price_haircut + num_perms * price_perm + num_dye_jobs * price_dye_job

noncomputable def total_hair_dye_cost : ℤ := 
  num_dye_jobs * cost_hair_dye_box

noncomputable def earnings_after_cost : ℤ := 
  total_service_earnings - total_hair_dye_cost

noncomputable def tips : ℤ := 
  total_end_day - earnings_after_cost

theorem Rebecca_tips_calculation : tips = 50 := by
  sorry

end Rebecca_tips_calculation_l1675_167591


namespace quadratic_no_real_roots_l1675_167582

theorem quadratic_no_real_roots (c : ℝ) : 
  (∀ x : ℝ, ¬(x^2 + x - c = 0)) ↔ c < -1/4 := 
sorry

end quadratic_no_real_roots_l1675_167582


namespace surface_area_original_cube_l1675_167581

theorem surface_area_original_cube
  (n : ℕ)
  (edge_length_smaller : ℕ)
  (smaller_cubes : ℕ)
  (original_surface_area : ℕ)
  (h1 : n = 3)
  (h2 : edge_length_smaller = 4)
  (h3 : smaller_cubes = 27)
  (h4 : 6 * (n * edge_length_smaller) ^ 2 = original_surface_area) :
  original_surface_area = 864 := by
  sorry

end surface_area_original_cube_l1675_167581


namespace find_multiplier_l1675_167595

theorem find_multiplier (x y : ℝ) (hx : x = 0.42857142857142855) (hx_nonzero : x ≠ 0) (h_eq : (x * y) / 7 = x^2) : y = 3 :=
sorry

end find_multiplier_l1675_167595


namespace symmetric_line_equation_l1675_167539

theorem symmetric_line_equation 
  (L : ℝ → ℝ → Prop)
  (H : ∀ x y, L x y ↔ x - 2 * y + 1 = 0) : 
  ∃ L' : ℝ → ℝ → Prop, 
    (∀ x y, L' x y ↔ x + 2 * y - 3 = 0) ∧ 
    ( ∀ x y, L (2 - x) y ↔ L' x y ) := 
sorry

end symmetric_line_equation_l1675_167539


namespace cows_in_group_l1675_167562

theorem cows_in_group (c h : ℕ) (L H: ℕ) 
  (legs_eq : L = 4 * c + 2 * h)
  (heads_eq : H = c + h)
  (legs_heads_relation : L = 2 * H + 14) 
  : c = 7 :=
by
  sorry

end cows_in_group_l1675_167562


namespace ziggy_rap_requests_l1675_167545

variables (total_songs electropop dance rock oldies djs_choice rap : ℕ)

-- Given conditions
axiom total_songs_eq : total_songs = 30
axiom electropop_eq : electropop = total_songs / 2
axiom dance_eq : dance = electropop / 3
axiom rock_eq : rock = 5
axiom oldies_eq : oldies = rock - 3
axiom djs_choice_eq : djs_choice = oldies / 2

-- Proof statement
theorem ziggy_rap_requests : rap = total_songs - electropop - dance - rock - oldies - djs_choice :=
by
  -- Apply the axioms and conditions to prove the resulting rap count
  sorry

end ziggy_rap_requests_l1675_167545


namespace proportion_equiv_l1675_167557

theorem proportion_equiv (X : ℕ) (h : 8 / 4 = X / 240) : X = 480 :=
by
  sorry

end proportion_equiv_l1675_167557


namespace lastNumberIsOneOverSeven_l1675_167516

-- Definitions and conditions
def seq (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, 2 ≤ k ∧ k ≤ 99 → a k = a (k - 1) * a (k + 1)

def nonZeroSeq (a : ℕ → ℝ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 100 → a k ≠ 0

def firstSeq7 (a : ℕ → ℝ) : Prop :=
  a 1 = 7

-- Theorem statement
theorem lastNumberIsOneOverSeven (a : ℕ → ℝ) :
  seq a → nonZeroSeq a → firstSeq7 a → a 100 = 1 / 7 :=
by
  sorry

end lastNumberIsOneOverSeven_l1675_167516


namespace sum_of_possible_values_l1675_167538

theorem sum_of_possible_values
  (x : ℝ)
  (h : (x + 3) * (x - 4) = 22) :
  ∃ s : ℝ, s = 1 :=
sorry

end sum_of_possible_values_l1675_167538


namespace area_of_garden_l1675_167529

variable (P : ℝ) (A : ℝ)

theorem area_of_garden (hP : P = 38) (hA : A = 2 * P + 14.25) : A = 90.25 :=
by
  sorry

end area_of_garden_l1675_167529


namespace teammates_of_oliver_l1675_167592

-- Define the player characteristics
structure Player :=
  (name   : String)
  (eyes   : String)
  (hair   : String)

-- Define the list of players with their given characteristics
def players : List Player := [
  {name := "Daniel", eyes := "Green", hair := "Red"},
  {name := "Oliver", eyes := "Gray", hair := "Brown"},
  {name := "Mia", eyes := "Gray", hair := "Red"},
  {name := "Ella", eyes := "Green", hair := "Brown"},
  {name := "Leo", eyes := "Green", hair := "Red"},
  {name := "Zoe", eyes := "Green", hair := "Brown"}
]

-- Define the condition for being on the same team
def same_team (p1 p2 : Player) : Bool :=
  (p1.eyes = p2.eyes && p1.hair ≠ p2.hair) || (p1.eyes ≠ p2.eyes && p1.hair = p2.hair)

-- Define the criterion to check if two players are on the same team as Oliver
def is_teammate_of_oliver (p : Player) : Bool :=
  let oliver := players[1] -- Oliver is the second player in the list
  same_team oliver p

-- Formal proof statement
theorem teammates_of_oliver : 
  is_teammate_of_oliver players[2] = true ∧ is_teammate_of_oliver players[3] = true :=
by
  -- Provide the intended proof here
  sorry

end teammates_of_oliver_l1675_167592


namespace eggs_needed_per_month_l1675_167589

def saly_needs : ℕ := 10
def ben_needs : ℕ := 14
def ked_needs : ℕ := ben_needs / 2
def weeks_in_month : ℕ := 4

def total_weekly_need : ℕ := saly_needs + ben_needs + ked_needs
def total_monthly_need : ℕ := total_weekly_need * weeks_in_month

theorem eggs_needed_per_month : total_monthly_need = 124 := by
  sorry

end eggs_needed_per_month_l1675_167589


namespace range_of_a_l1675_167500

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3 * x + a

theorem range_of_a (a : ℝ) :
  (∃ (m n p : ℝ), m ≠ n ∧ n ≠ p ∧ m ≠ p ∧ f m a = 2024 ∧ f n a = 2024 ∧ f p a = 2024) ↔
  2022 < a ∧ a < 2026 :=
sorry

end range_of_a_l1675_167500


namespace sum_of_squares_l1675_167579

theorem sum_of_squares (a b c : ℝ) :
  a + b + c = 4 → ab + ac + bc = 4 → a^2 + b^2 + c^2 = 8 :=
by
  sorry

end sum_of_squares_l1675_167579


namespace area_enclosed_by_equation_is_96_l1675_167511

-- Definitions based on the conditions
def equation (x y : ℝ) : Prop := |x| + |3 * y| = 12

-- The theorem to prove the area enclosed by the graph is 96 square units
theorem area_enclosed_by_equation_is_96 :
  (∃ x y : ℝ, equation x y) → ∃ A : ℝ, A = 96 :=
sorry

end area_enclosed_by_equation_is_96_l1675_167511


namespace triangle_base_and_height_l1675_167536

theorem triangle_base_and_height (h b : ℕ) (A : ℕ) (hb : b = h - 4) (hA : A = 96) 
  (hArea : A = (1 / 2) * b * h) : (b = 12 ∧ h = 16) :=
by
  sorry

end triangle_base_and_height_l1675_167536


namespace opposite_of_2023_is_neg_2023_l1675_167549

def opposite_of (x : Int) : Int := -x

theorem opposite_of_2023_is_neg_2023 : opposite_of 2023 = -2023 :=
by
  sorry

end opposite_of_2023_is_neg_2023_l1675_167549


namespace find_certain_number_l1675_167544

noncomputable def certain_number (x : ℝ) : Prop :=
  3005 - 3000 + x = 2705

theorem find_certain_number : ∃ x : ℝ, certain_number x ∧ x = 2700 :=
by
  use 2700
  unfold certain_number
  sorry

end find_certain_number_l1675_167544


namespace necessary_and_sufficient_condition_l1675_167531

variable (f : ℝ → ℝ)

-- Define even function
def even_function : Prop := ∀ x, f x = f (-x)

-- Define periodic function with period 2
def periodic_function : Prop := ∀ x, f (x + 2) = f x

-- Define increasing function on [0, 1]
def increasing_on_0_1 : Prop := ∀ x y, 0 ≤ x → x ≤ y → y ≤ 1 → f x ≤ f y

-- Define decreasing function on [3, 4]
def decreasing_on_3_4 : Prop := ∀ x y, 3 ≤ x → x ≤ y → y ≤ 4 → f x ≥ f y

theorem necessary_and_sufficient_condition :
  even_function f →
  periodic_function f →
  (increasing_on_0_1 f ↔ decreasing_on_3_4 f) :=
by
  intros h_even h_periodic
  sorry

end necessary_and_sufficient_condition_l1675_167531


namespace derivative_at_pi_over_3_l1675_167505

noncomputable def f (x : ℝ) : ℝ := Real.cos x + Real.sqrt 3 * Real.sin x

theorem derivative_at_pi_over_3 : 
  (deriv f) (Real.pi / 3) = 0 := 
by 
  sorry

end derivative_at_pi_over_3_l1675_167505


namespace cos_product_identity_l1675_167509

noncomputable def L : ℝ := 3.418 * (Real.cos (2 * Real.pi / 31)) *
                               (Real.cos (4 * Real.pi / 31)) *
                               (Real.cos (8 * Real.pi / 31)) *
                               (Real.cos (16 * Real.pi / 31)) *
                               (Real.cos (32 * Real.pi / 31))

theorem cos_product_identity : L = 1 / 32 := by
  sorry

end cos_product_identity_l1675_167509


namespace evaluate_g_h_2_l1675_167558

def g (x : ℝ) : ℝ := 3 * x^2 - 4 
def h (x : ℝ) : ℝ := -2 * x^3 + 2 

theorem evaluate_g_h_2 : g (h 2) = 584 := by
  sorry

end evaluate_g_h_2_l1675_167558


namespace g_at_2_l1675_167508

-- Assuming g is a function from ℝ to ℝ such that it satisfies the given condition.
def g : ℝ → ℝ := sorry

-- Condition of the problem
axiom g_condition : ∀ x : ℝ, g (2 ^ x) + x * g (2 ^ (-x)) = 2

-- The statement we want to prove
theorem g_at_2 : g (2) = 0 :=
by
  sorry

end g_at_2_l1675_167508


namespace sin_double_angle_l1675_167512

theorem sin_double_angle (x : ℝ) (h : Real.sin (π / 4 - x) = 3 / 5) : Real.sin (2 * x) = 7 / 25 := by
  sorry

end sin_double_angle_l1675_167512


namespace solve_x_perpendicular_l1675_167567

def vec_a : ℝ × ℝ := (1, 3)
def vec_b (x : ℝ) : ℝ × ℝ := (3, x)

def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem solve_x_perpendicular (x : ℝ) (h : perpendicular vec_a (vec_b x)) : x = -1 :=
by {
  sorry
}

end solve_x_perpendicular_l1675_167567


namespace h_at_7_over_5_eq_0_l1675_167578

def h (x : ℝ) : ℝ := 5 * x - 7

theorem h_at_7_over_5_eq_0 : h (7 / 5) = 0 := 
by 
  sorry

end h_at_7_over_5_eq_0_l1675_167578


namespace find_fx_neg_l1675_167501

variable (f : ℝ → ℝ)

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def f_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → f x = x^2 - 2*x

theorem find_fx_neg (h1 : odd_function f) (h2 : f_nonneg f) : 
  ∀ x : ℝ, x < 0 → f x = -x^2 - 2*x := 
by
  sorry

end find_fx_neg_l1675_167501


namespace mike_falls_short_l1675_167518

theorem mike_falls_short : 
  ∀ (max_marks mike_score : ℕ) (pass_percentage : ℚ),
  pass_percentage = 0.30 → 
  max_marks = 800 → 
  mike_score = 212 → 
  (pass_percentage * max_marks - mike_score) = 28 :=
by
  intros max_marks mike_score pass_percentage h1 h2 h3
  sorry

end mike_falls_short_l1675_167518


namespace bicycle_helmet_lock_costs_l1675_167514

-- Given total cost, relationships between costs, and the specific costs
theorem bicycle_helmet_lock_costs (H : ℝ) (bicycle helmet lock : ℝ) 
  (h1 : bicycle = 5 * H) 
  (h2 : helmet = H) 
  (h3 : lock = H / 2)
  (total_cost : bicycle + helmet + lock = 360) :
  H = 55.38 ∧ bicycle = 276.90 ∧ lock = 27.72 :=
by 
  -- The proof would go here
  sorry

end bicycle_helmet_lock_costs_l1675_167514


namespace vector_equation_l1675_167519

noncomputable def vec_a : (ℝ × ℝ) := (1, -1)
noncomputable def vec_b : (ℝ × ℝ) := (2, 1)
noncomputable def vec_c : (ℝ × ℝ) := (-2, 1)

theorem vector_equation (x y : ℝ) 
  (h : vec_c = (x * vec_a.1 + y * vec_b.1, x * vec_a.2 + y * vec_b.2)) : 
  x - y = -1 := 
by { sorry }

end vector_equation_l1675_167519


namespace diagonal_of_rectangle_l1675_167513

theorem diagonal_of_rectangle (a b d : ℝ)
  (h_side : a = 15)
  (h_area : a * b = 120)
  (h_diag : a^2 + b^2 = d^2) :
  d = 17 :=
by
  sorry

end diagonal_of_rectangle_l1675_167513


namespace stones_required_correct_l1675_167594

/- 
Given:
- The hall measures 36 meters long and 15 meters broad.
- Each stone measures 6 decimeters by 5 decimeters.

We need to prove that the number of stones required to pave the hall is 1800.
-/
noncomputable def stones_required 
  (hall_length_m : ℕ) 
  (hall_breadth_m : ℕ) 
  (stone_length_dm : ℕ) 
  (stone_breadth_dm : ℕ) : ℕ :=
  (hall_length_m * 10) * (hall_breadth_m * 10) / (stone_length_dm * stone_breadth_dm)

theorem stones_required_correct : 
  stones_required 36 15 6 5 = 1800 :=
by 
  -- Placeholder for proof
  sorry

end stones_required_correct_l1675_167594


namespace min_value_l1675_167551

theorem min_value (a b : ℝ) (h : a * b > 0) : (∃ x, x = a^2 + 4 * b^2 + 1 / (a * b) ∧ ∀ y, y = a^2 + 4 * b^2 + 1 / (a * b) → y ≥ 4) :=
sorry

end min_value_l1675_167551


namespace zoo_ticket_sales_l1675_167534

theorem zoo_ticket_sales (A K : ℕ) (h1 : A + K = 254) (h2 : 28 * A + 12 * K = 3864) : K = 202 :=
by {
  sorry
}

end zoo_ticket_sales_l1675_167534


namespace geometric_sequence_sum_l1675_167510

variables (a : ℕ → ℤ) (q : ℤ)

-- assumption that the sequence is geometric
def geometric_sequence (a : ℕ → ℤ) (q : ℤ) : Prop := 
  ∀ n, a (n + 1) = a n * q

noncomputable def a2 := a 2
noncomputable def a3 := a 3
noncomputable def a4 := a 4
noncomputable def a5 := a 5
noncomputable def a6 := a 6
noncomputable def a7 := a 7

theorem geometric_sequence_sum
  (h_geom : geometric_sequence a q)
  (h1 : a2 + a3 = 1)
  (h2 : a3 + a4 = -2) :
  a5 + a6 + a7 = 24 :=
sorry

end geometric_sequence_sum_l1675_167510


namespace find_m_l1675_167521

variable (m : ℝ)

theorem find_m (h1 : 3 * (-7.5) - y = m) (h2 : -0.4 * (-7.5) + y = 3) : m = -22.5 :=
by
  sorry

end find_m_l1675_167521


namespace binomial_product_l1675_167588

def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem binomial_product : binomial 10 3 * binomial 8 3 = 6720 := 
by 
  sorry

end binomial_product_l1675_167588


namespace unique_zero_point_mn_l1675_167503

noncomputable def f (a : ℝ) (x : ℝ) := a * (x^2 + 2 / x) - Real.log x

theorem unique_zero_point_mn (a : ℝ) (m n x₀ : ℝ) (hmn : m + 1 = n) (a_pos : 0 < a) (f_zero : f a x₀ = 0) (x0_in_range : m < x₀ ∧ x₀ < n) : m + n = 5 := by
  sorry

end unique_zero_point_mn_l1675_167503


namespace James_wait_weeks_l1675_167584

def JamesExercising (daysPainSubside : ℕ) (healingMultiplier : ℕ) (delayAfterHealing : ℕ) (totalDaysUntilHeavyLift : ℕ) : ℕ :=
  let healingTime := daysPainSubside * healingMultiplier
  let startWorkingOut := healingTime + delayAfterHealing
  let waitingPeriodDays := totalDaysUntilHeavyLift - startWorkingOut
  waitingPeriodDays / 7

theorem James_wait_weeks : 
  JamesExercising 3 5 3 39 = 3 :=
by
  sorry

end James_wait_weeks_l1675_167584


namespace intersection_A_B_l1675_167548

def A : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 2 * x + 5}
def B : Set (ℝ × ℝ) := {p | ∃ x y, p = (x, y) ∧ y = 1 - 2 * x}
def inter : Set (ℝ × ℝ) := {(x, y) | x = -1 ∧ y = 3}

theorem intersection_A_B :
  A ∩ B = inter :=
sorry

end intersection_A_B_l1675_167548


namespace happy_boys_count_l1675_167587

def total_children := 60
def happy_children := 30
def sad_children := 10
def neither_happy_nor_sad_children := total_children - happy_children - sad_children

def total_boys := 19
def total_girls := 41
def sad_girls := 4
def neither_happy_nor_sad_boys := 7

def sad_boys := sad_children - sad_girls

theorem happy_boys_count :
  total_boys - sad_boys - neither_happy_nor_sad_boys = 6 :=
by
  sorry

end happy_boys_count_l1675_167587


namespace cheaper_fuji_shimla_l1675_167565

variable (S R F : ℝ)
variable (h : 1.05 * (S + R) = R + 0.90 * F + 250)

theorem cheaper_fuji_shimla : S - F = (-0.15 * S - 0.05 * R) / 0.90 + 250 / 0.90 :=
by
  sorry

end cheaper_fuji_shimla_l1675_167565


namespace initial_rows_of_chairs_l1675_167553

theorem initial_rows_of_chairs (x : ℕ) (h1 : 12 * x + 11 = 95) : x = 7 := 
by
  sorry

end initial_rows_of_chairs_l1675_167553


namespace average_speed_correct_l1675_167530

noncomputable def initial_odometer := 12321
noncomputable def final_odometer := 12421
noncomputable def time_hours := 4
noncomputable def distance := final_odometer - initial_odometer
noncomputable def avg_speed := distance / time_hours

theorem average_speed_correct : avg_speed = 25 := by
  sorry

end average_speed_correct_l1675_167530


namespace count_three_element_arithmetic_mean_subsets_l1675_167574
open Nat

theorem count_three_element_arithmetic_mean_subsets (n : ℕ) (h : n ≥ 3) :
    ∃ a_n : ℕ, a_n = (n / 2) * ((n - 1) / 2) :=
by
  sorry

end count_three_element_arithmetic_mean_subsets_l1675_167574


namespace find_page_words_l1675_167543
open Nat

-- Define the conditions
def condition1 : Nat := 150
def condition2 : Nat := 221
def total_words_modulo : Nat := 220
def upper_bound_words : Nat := 120

-- Define properties
def is_solution (p : Nat) : Prop :=
  Nat.Prime p ∧ p ≤ upper_bound_words ∧ (condition1 * p) % condition2 = total_words_modulo

-- The theorem to prove
theorem find_page_words (p : Nat) (hp : is_solution p) : p = 67 :=
by
  sorry

end find_page_words_l1675_167543


namespace cricket_innings_count_l1675_167540

theorem cricket_innings_count (n : ℕ) (h_avg_current : ∀ (total_runs : ℕ), total_runs = 32 * n)
  (h_runs_needed : ∀ (total_runs : ℕ), total_runs + 116 = 36 * (n + 1)) : n = 20 :=
by
  sorry

end cricket_innings_count_l1675_167540


namespace zero_function_is_uniq_l1675_167598

theorem zero_function_is_uniq (f : ℝ → ℝ) :
  (∀ (x : ℝ) (hx : x ≠ 0) (y : ℝ), f (x^2 + y) ≥ (1/x + 1) * f y) → 
  (∀ x, f x = 0) :=
by
  sorry

end zero_function_is_uniq_l1675_167598


namespace scheduling_arrangements_l1675_167575

-- We want to express this as a problem to prove the number of scheduling arrangements.

theorem scheduling_arrangements (n : ℕ) (h : n = 6) :
  (Nat.choose 6 1) * (Nat.choose 5 1) * (Nat.choose 4 2) = 180 := by
  sorry

end scheduling_arrangements_l1675_167575


namespace hiker_walks_18_miles_on_first_day_l1675_167541

noncomputable def miles_walked_first_day (h : ℕ) : ℕ := 3 * h

def total_miles_walked (h : ℕ) : ℕ := (3 * h) + (4 * (h - 1)) + (4 * h)

theorem hiker_walks_18_miles_on_first_day :
  (∃ h : ℕ, total_miles_walked h = 62) → miles_walked_first_day 6 = 18 :=
by
  sorry

end hiker_walks_18_miles_on_first_day_l1675_167541


namespace banana_price_l1675_167590

theorem banana_price (b : ℝ) : 
    (∃ x : ℕ, 0.70 * x + b * (9 - x) = 5.60 ∧ x + (9 - x) = 9) → b = 0.60 :=
by
  intro h
  obtain ⟨x, hx1, hx2⟩ := h
  -- equations to work with:
  -- 0.70 * x + b * (9 - x) = 5.60
  -- x + (9 - x) = 9
  sorry

end banana_price_l1675_167590


namespace find_y_l1675_167535

theorem find_y
  (x y : ℝ)
  (h1 : x^(3*y) = 8)
  (h2 : x = 2) :
  y = 1 :=
sorry

end find_y_l1675_167535


namespace smaller_root_of_equation_l1675_167550

theorem smaller_root_of_equation : 
  ∀ x : ℝ, (x - 3 / 4) * (x - 3 / 4) + (x - 3 / 4) * (x - 1 / 4) = 0 → x = 1 / 2 :=
by
  intros x h
  sorry

end smaller_root_of_equation_l1675_167550


namespace directrix_of_parabola_l1675_167597

-- Define the given conditions
def parabola_focus_on_line (p : ℝ) := ∃ (x y : ℝ), y^2 = 2 * p * x ∧ 2 * x + 3 * y - 8 = 0

-- Define the statement to be proven
theorem directrix_of_parabola (p : ℝ) (h: parabola_focus_on_line p) : 
   ∃ (d : ℝ), d = -4 := 
sorry

end directrix_of_parabola_l1675_167597


namespace an_plus_an_minus_1_eq_two_pow_n_l1675_167566

def a_n (n : ℕ) : ℕ := sorry -- Placeholder for the actual function a_n

theorem an_plus_an_minus_1_eq_two_pow_n (n : ℕ) (h : n ≥ 4) : a_n (n - 1) + a_n n = 2^n := 
by
  sorry

end an_plus_an_minus_1_eq_two_pow_n_l1675_167566


namespace sum_of_roots_l1675_167569

theorem sum_of_roots (x₁ x₂ : ℝ) 
  (h₁ : x₁^2 - 2 * x₁ - 8 = 0) 
  (h₂ : x₂^2 - 2 * x₂ - 8 = 0)
  (h_distinct : x₁ ≠ x₂) : 
  x₁ + x₂ = 2 := 
sorry

end sum_of_roots_l1675_167569


namespace proof_problem_l1675_167533

-- Definitions for the arithmetic and geometric sequences
def a_n (n : ℕ) : ℚ := 2 * n - 4
def b_n (n : ℕ) : ℚ := 2^(n - 2)

-- Conditions based on initial problem statements
axiom a_2 : a_n 2 = 0
axiom b_2 : b_n 2 = 1
axiom a_3_eq_b_3 : a_n 3 = b_n 3
axiom a_4_eq_b_4 : a_n 4 = b_n 4

-- Sum of first n terms of the sequence {n * b_n}
def S_n (n : ℕ) : ℚ := (n-1) * 2^(n-1) + 1/2

-- The main theorem to prove
theorem proof_problem (n : ℕ) : ∃ a_n b_n S_n, 
    (a_n = 2 * n - 4) ∧
    (b_n = 2^(n - 2)) ∧
    (S_n = (n-1) * 2^(n-1) + 1/2) :=
by {
    sorry
}

end proof_problem_l1675_167533


namespace circle_intersection_range_l1675_167560

theorem circle_intersection_range (m : ℝ) :
  (x^2 + y^2 - 4*x + 2*m*y + m + 6 = 0) ∧ 
  (∀ A B : ℝ, 
    (A - y = 0) ∧ (B - y = 0) → A * B > 0
  ) → 
  (m > 2 ∨ (-6 < m ∧ m < -2)) :=
by 
  sorry

end circle_intersection_range_l1675_167560


namespace inequality_one_inequality_system_l1675_167504

theorem inequality_one (x : ℝ) : 2 * x + 3 ≤ 5 * x ↔ x ≥ 1 := sorry

theorem inequality_system (x : ℝ) : 
  (5 * x - 1 ≤ 3 * (x + 1)) ∧ 
  ((2 * x - 1) / 2 - (5 * x - 1) / 4 < 1) ↔ 
  (-5 < x ∧ x ≤ 2) := sorry

end inequality_one_inequality_system_l1675_167504


namespace factory_earns_8100_per_day_l1675_167571

-- Define the conditions
def working_hours_machines := 23
def working_hours_fourth_machine := 12
def production_per_hour := 2
def price_per_kg := 50
def number_of_machines := 3

-- Calculate earnings
def total_earnings : ℕ :=
  let total_runtime_machines := number_of_machines * working_hours_machines
  let production_machines := total_runtime_machines * production_per_hour
  let production_fourth_machine := working_hours_fourth_machine * production_per_hour
  let total_production := production_machines + production_fourth_machine
  total_production * price_per_kg

theorem factory_earns_8100_per_day : total_earnings = 8100 :=
by
  sorry

end factory_earns_8100_per_day_l1675_167571


namespace total_distance_travelled_l1675_167596

theorem total_distance_travelled (total_time hours_foot hours_bicycle speed_foot speed_bicycle distance_foot : ℕ)
  (h1 : total_time = 7)
  (h2 : speed_foot = 8)
  (h3 : speed_bicycle = 16)
  (h4 : distance_foot = 32)
  (h5 : hours_foot = distance_foot / speed_foot)
  (h6 : hours_bicycle = total_time - hours_foot)
  (distance_bicycle := speed_bicycle * hours_bicycle) :
  distance_foot + distance_bicycle = 80 := 
by
  sorry

end total_distance_travelled_l1675_167596


namespace ab_plus_a_plus_b_l1675_167568

-- Define the polynomial
def poly (x : ℝ) : ℝ := x^4 - 6 * x^2 - x + 2
-- Define the conditions on a and b
def is_root (x : ℝ) : Prop := poly x = 0

-- State the theorem
theorem ab_plus_a_plus_b (a b : ℝ) (ha : is_root a) (hb : is_root b) : a * b + a + b = 1 :=
sorry

end ab_plus_a_plus_b_l1675_167568


namespace jim_out_of_pocket_cost_l1675_167563

theorem jim_out_of_pocket_cost {price1 price2 sale : ℕ} 
    (h1 : price1 = 10000)
    (h2 : price2 = 2 * price1)
    (h3 : sale = price1 / 2) :
    (price1 + price2 - sale = 25000) :=
by
  sorry

end jim_out_of_pocket_cost_l1675_167563


namespace smiths_bakery_multiple_l1675_167564

theorem smiths_bakery_multiple (x : ℤ) (mcgee_pies : ℤ) (smith_pies : ℤ) 
  (h1 : smith_pies = x * mcgee_pies + 6)
  (h2 : mcgee_pies = 16)
  (h3 : smith_pies = 70) : x = 4 :=
by
  sorry

end smiths_bakery_multiple_l1675_167564


namespace sum_of_reciprocals_is_five_l1675_167502

theorem sum_of_reciprocals_is_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x - y = 3 * x * y) : 
  (1 / x) + (1 / y) = 5 :=
sorry

end sum_of_reciprocals_is_five_l1675_167502


namespace volunteer_recommendations_l1675_167546

def num_recommendations (boys girls : ℕ) (total_choices chosen : ℕ) : ℕ :=
  let total_combinations := Nat.choose total_choices chosen
  let invalid_combinations := Nat.choose boys chosen
  total_combinations - invalid_combinations

theorem volunteer_recommendations : num_recommendations 4 3 7 4 = 34 := by
  sorry

end volunteer_recommendations_l1675_167546


namespace small_cubes_with_two_faces_painted_red_l1675_167528

theorem small_cubes_with_two_faces_painted_red (edge_length : ℕ) (small_cube_edge_length : ℕ)
  (h1 : edge_length = 4) (h2 : small_cube_edge_length = 1) :
  ∃ n, n = 24 :=
by
  -- Proof skipped
  sorry

end small_cubes_with_two_faces_painted_red_l1675_167528


namespace cookies_left_correct_l1675_167552

def cookies_left (cookies_per_dozen : ℕ) (flour_per_dozen_lb : ℕ) (bag_count : ℕ) (flour_per_bag_lb : ℕ) (cookies_eaten : ℕ) : ℕ :=
  let total_flour_lb := bag_count * flour_per_bag_lb
  let total_cookies := (total_flour_lb / flour_per_dozen_lb) * cookies_per_dozen
  total_cookies - cookies_eaten

theorem cookies_left_correct :
  cookies_left 12 2 4 5 15 = 105 :=
by sorry

end cookies_left_correct_l1675_167552


namespace gcd_g10_g13_l1675_167525

-- Define the polynomial function g
def g (x : ℤ) : ℤ := x^3 - 3 * x^2 + x + 2050

-- State the theorem to prove that gcd(g(10), g(13)) is 1
theorem gcd_g10_g13 : Int.gcd (g 10) (g 13) = 1 := by
  sorry

end gcd_g10_g13_l1675_167525


namespace compute_sum_pq_pr_qr_l1675_167573

theorem compute_sum_pq_pr_qr (p q r : ℝ) (h : 5 * (p + q + r) = p^2 + q^2 + r^2) : 
  let N := 150
  let n := -12.5
  N + 15 * n = -37.5 := 
by {
  sorry
}

end compute_sum_pq_pr_qr_l1675_167573


namespace frank_worked_days_l1675_167547

def total_hours : ℝ := 8.0
def hours_per_day : ℝ := 2.0

theorem frank_worked_days :
  (total_hours / hours_per_day = 4.0) :=
by sorry

end frank_worked_days_l1675_167547


namespace ratio_of_ticket_prices_l1675_167517

-- Given conditions
def num_adults := 400
def num_children := 200
def adult_ticket_price : ℕ := 32
def total_amount : ℕ := 16000
def child_ticket_price (C : ℕ) : Prop := num_adults * adult_ticket_price + num_children * C = total_amount

theorem ratio_of_ticket_prices (C : ℕ) (hC : child_ticket_price C) :
  adult_ticket_price / C = 2 :=
by
  sorry

end ratio_of_ticket_prices_l1675_167517


namespace hand_position_at_8PM_yesterday_l1675_167537

-- Define the conditions of the problem
def positions : ℕ := 20
def jump_interval_min : ℕ := 7
def jump_positions : ℕ := 9
def start_position : ℕ := 0
def end_position : ℕ := 8 -- At 8:00 AM, the hand is at position 9, hence moving forward 8 positions from position 0

-- Define the total time from 8:00 PM yesterday to 8:00 AM today
def total_minutes : ℕ := 720

-- Calculate the number of full jumps
def num_full_jumps : ℕ := total_minutes / jump_interval_min

-- Calculate the hand's final position from 8:00 PM yesterday
def final_hand_position : ℕ := (start_position + num_full_jumps * jump_positions) % positions

-- Prove that the final hand position is 2
theorem hand_position_at_8PM_yesterday : final_hand_position = 2 :=
by
  sorry

end hand_position_at_8PM_yesterday_l1675_167537


namespace min_f_x_gt_2_solve_inequality_l1675_167524

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 / (x + b)

theorem min_f_x_gt_2 (a b : ℝ) (h1 : ∀ x, f a b x = 2 * x + 3 → x = -2 ∨ x = 3) :
∃ c, ∀ x > 2, f a b x ≥ c ∧ (∀ y, y > 2 → f a b y = c → y = 4 ∧ c = 8) :=
sorry

theorem solve_inequality (a b k : ℝ) (x : ℝ) (h1 : ∀ x, f a b x = 2 * x + 3 → x = -2 ∨ x = 3) :
  f a b x < (k * (x - 1) + 1 - x^2) / (2 - x) ↔ 
  (x < 2 ∧ k = 0) ∨ 
  (-1 < k ∧ k < 0 ∧ 1 - 1 / k < x ∧ x < 2) ∨ 
  ((k > 0 ∨ k < -1) ∧ (1 - 1 / k < x ∧ x < 2) ∨ x > 2) ∨ 
  (k = -1 ∧ x ≠ 2) :=
sorry

end min_f_x_gt_2_solve_inequality_l1675_167524


namespace geometric_mean_of_4_and_9_l1675_167555

theorem geometric_mean_of_4_and_9 : ∃ G : ℝ, (4 / G = G / 9) ∧ (G = 6 ∨ G = -6) := 
by
  sorry

end geometric_mean_of_4_and_9_l1675_167555


namespace stream_speed_l1675_167585

-- Define the conditions
def still_water_speed : ℝ := 15
def upstream_time_factor : ℕ := 2

-- Define the theorem
theorem stream_speed (t v : ℝ) (h : (still_water_speed + v) * t = (still_water_speed - v) * (upstream_time_factor * t)) : v = 5 :=
by
  sorry

end stream_speed_l1675_167585


namespace belongs_to_one_progression_l1675_167522

-- Define the arithmetic progression and membership property
def is_arith_prog (P : ℕ → Prop) : Prop :=
  ∃ a d, ∀ n, P (a + n * d)

-- Define the given conditions
def condition (P1 P2 P3 : ℕ → Prop) : Prop :=
  is_arith_prog P1 ∧ is_arith_prog P2 ∧ is_arith_prog P3 ∧
  (P1 1 ∨ P2 1 ∨ P3 1) ∧
  (P1 2 ∨ P2 2 ∨ P3 2) ∧
  (P1 3 ∨ P2 3 ∨ P3 3) ∧
  (P1 4 ∨ P2 4 ∨ P3 4) ∧
  (P1 5 ∨ P2 5 ∨ P3 5) ∧
  (P1 6 ∨ P2 6 ∨ P3 6) ∧
  (P1 7 ∨ P2 7 ∨ P3 7) ∧
  (P1 8 ∨ P2 8 ∨ P3 8)

-- Statement to prove
theorem belongs_to_one_progression (P1 P2 P3 : ℕ → Prop) (h : condition P1 P2 P3) : 
  P1 1980 ∨ P2 1980 ∨ P3 1980 := 
by
sorry

end belongs_to_one_progression_l1675_167522


namespace find_missing_number_l1675_167527

theorem find_missing_number (x : ℕ) (h : 10111 - x * 2 * 5 = 10011) : x = 5 := 
sorry

end find_missing_number_l1675_167527


namespace memory_efficiency_problem_l1675_167576

theorem memory_efficiency_problem (x : ℝ) (hx : x ≠ 0) :
  (100 / x - 100 / (1.2 * x) = 5 / 12) ↔ (100 / x - 100 / ((1 + 0.20) * x) = 5 / 12) :=
by sorry

end memory_efficiency_problem_l1675_167576


namespace find_a_c_l1675_167532

theorem find_a_c (a b c : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_c_neg : c < 0)
    (h_max : c + a = 3) (h_min : c - a = -5) :
  a = 4 ∧ c = -1 := 
sorry

end find_a_c_l1675_167532


namespace overall_percent_change_l1675_167577

theorem overall_percent_change (x : ℝ) : 
  (0.85 * x * 1.25 * 0.9 / x - 1) * 100 = -4.375 := 
by 
  sorry

end overall_percent_change_l1675_167577


namespace hours_of_work_l1675_167559

variables (M W X : ℝ)

noncomputable def work_rate := 
  (2 * M + 3 * W) * X * 5 = 1 ∧ 
  (4 * M + 4 * W) * 3 * 7 = 1 ∧ 
  7 * M * 4 * 5.000000000000001 = 1

theorem hours_of_work (M W : ℝ) (h : work_rate M W 7) : X = 7 :=
sorry

end hours_of_work_l1675_167559


namespace percent_problem_l1675_167599

theorem percent_problem (x : ℝ) (hx : 0.60 * 600 = 0.50 * x) : x = 720 :=
by
  sorry

end percent_problem_l1675_167599


namespace acuteAngleAt725_l1675_167554

noncomputable def hourHandPosition (h : ℝ) (m : ℝ) : ℝ :=
  h * 30 + m / 60 * 30

noncomputable def minuteHandPosition (m : ℝ) : ℝ :=
  m / 60 * 360

noncomputable def angleBetweenHands (h m : ℝ) : ℝ :=
  abs (hourHandPosition h m - minuteHandPosition m)

theorem acuteAngleAt725 : angleBetweenHands 7 25 = 72.5 :=
  sorry

end acuteAngleAt725_l1675_167554


namespace charity_tickets_solution_l1675_167583

theorem charity_tickets_solution (f h d p : ℕ) (ticket_count : f + h + d = 200)
  (revenue : f * p + h * (p / 2) + d * (2 * p) = 3600) : f = 80 := by
  sorry

end charity_tickets_solution_l1675_167583


namespace number_cooking_and_weaving_l1675_167542

section CurriculumProblem

variables {total_yoga total_cooking total_weaving : ℕ}
variables {cooking_only cooking_and_yoga all_curriculums CW : ℕ}

-- Given conditions
def yoga (total_yoga : ℕ) := total_yoga = 35
def cooking (total_cooking : ℕ) := total_cooking = 20
def weaving (total_weaving : ℕ) := total_weaving = 15
def cookingOnly (cooking_only : ℕ) := cooking_only = 7
def cookingAndYoga (cooking_and_yoga : ℕ) := cooking_and_yoga = 5
def allCurriculums (all_curriculums : ℕ) := all_curriculums = 3

-- Prove that CW (number of people studying both cooking and weaving) is 8
theorem number_cooking_and_weaving : 
  yoga total_yoga → cooking total_cooking → weaving total_weaving → 
  cookingOnly cooking_only → cookingAndYoga cooking_and_yoga → 
  allCurriculums all_curriculums → CW = 8 := 
by 
  intros h_yoga h_cooking h_weaving h_cookingOnly h_cookingAndYoga h_allCurriculums
  -- Placeholder for the actual proof
  sorry

end CurriculumProblem

end number_cooking_and_weaving_l1675_167542


namespace max_possible_median_l1675_167507

theorem max_possible_median (total_cups : ℕ) (total_customers : ℕ) (min_cups_per_customer : ℕ)
  (h1 : total_cups = 310) (h2 : total_customers = 120) (h3 : min_cups_per_customer = 1) :
  ∃ median : ℕ, median = 4 :=
by {
  sorry
}

end max_possible_median_l1675_167507


namespace amelia_wins_probability_l1675_167520

def amelia_prob_heads : ℚ := 1 / 4
def blaine_prob_heads : ℚ := 3 / 7

def probability_blaine_wins_first_turn : ℚ := blaine_prob_heads

def probability_amelia_wins_first_turn : ℚ :=
  (1 - blaine_prob_heads) * amelia_prob_heads

def probability_amelia_wins_second_turn : ℚ :=
  (1 - blaine_prob_heads) * (1 - amelia_prob_heads) * (1 - blaine_prob_heads) * amelia_prob_heads

def probability_amelia_wins_third_turn : ℚ :=
  (1 - blaine_prob_heads) * (1 - amelia_prob_heads) * (1 - blaine_prob_heads) * 
  (1 - amelia_prob_heads) * (1 - blaine_prob_heads) * amelia_prob_heads

def probability_amelia_wins : ℚ :=
  probability_amelia_wins_first_turn + probability_amelia_wins_second_turn + probability_amelia_wins_third_turn

theorem amelia_wins_probability : probability_amelia_wins = 223 / 784 := by
  sorry

end amelia_wins_probability_l1675_167520


namespace solve_for_y_l1675_167523

theorem solve_for_y {y : ℕ} (h : (1000 : ℝ) = (10 : ℝ)^3) : (1000 : ℝ)^4 = (10 : ℝ)^y ↔ y = 12 :=
by
  sorry

end solve_for_y_l1675_167523


namespace average_score_of_juniors_l1675_167506

theorem average_score_of_juniors :
  ∀ (N : ℕ) (junior_percent senior_percent overall_avg senior_avg : ℚ),
  junior_percent = 0.20 →
  senior_percent = 0.80 →
  overall_avg = 86 →
  senior_avg = 85 →
  (N * overall_avg - (N * senior_percent * senior_avg)) / (N * junior_percent) = 90 := 
by
  intros N junior_percent senior_percent overall_avg senior_avg
  intros h1 h2 h3 h4
  sorry

end average_score_of_juniors_l1675_167506


namespace find_natural_number_l1675_167593

theorem find_natural_number (n : ℕ) (h : ∃ k : ℕ, n^2 - 19 * n + 95 = k^2) : n = 5 ∨ n = 14 := by
  sorry

end find_natural_number_l1675_167593


namespace age_of_new_person_l1675_167586

theorem age_of_new_person (T A : ℤ) (h : (T / 10 - 3) = (T - 40 + A) / 10) : A = 10 :=
sorry

end age_of_new_person_l1675_167586
