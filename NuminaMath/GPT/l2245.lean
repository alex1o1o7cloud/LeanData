import Mathlib

namespace NUMINAMATH_GPT_new_average_production_l2245_224552

theorem new_average_production (n : ℕ) (daily_avg : ℕ) (today_prod : ℕ) (new_avg : ℕ) 
  (h1 : daily_avg = 50) 
  (h2 : today_prod = 95) 
  (h3 : n = 8) 
  (h4 : new_avg = (daily_avg * n + today_prod) / (n + 1)) : 
  new_avg = 55 := 
sorry

end NUMINAMATH_GPT_new_average_production_l2245_224552


namespace NUMINAMATH_GPT_derivative_of_y_l2245_224529

variable (a b c x : ℝ)

def y : ℝ := (x - a) * (x - b) * (x - c)

theorem derivative_of_y :
  deriv (fun x:ℝ => (x - a) * (x - b) * (x - c)) x = 3 * x^2 - 2 * (a + b + c) * x + (a * b + a * c + b * c) :=
by
  sorry

end NUMINAMATH_GPT_derivative_of_y_l2245_224529


namespace NUMINAMATH_GPT_solve_system_of_equations_l2245_224574

theorem solve_system_of_equations (x y : ℝ) :
  16 * x^3 + 4 * x = 16 * y + 5 ∧ 16 * y^3 + 4 * y = 16 * x + 5 → x = y ∧ 16 * x^3 - 12 * x - 5 = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2245_224574


namespace NUMINAMATH_GPT_activity_popularity_order_l2245_224564

theorem activity_popularity_order
  (dodgeball : ℚ := 13 / 40)
  (picnic : ℚ := 9 / 30)
  (swimming : ℚ := 7 / 20)
  (crafts : ℚ := 3 / 15) :
  (swimming > dodgeball ∧ dodgeball > picnic ∧ picnic > crafts) :=
by 
  sorry

end NUMINAMATH_GPT_activity_popularity_order_l2245_224564


namespace NUMINAMATH_GPT_exists_odd_digit_div_by_five_power_l2245_224559

theorem exists_odd_digit_div_by_five_power (n : ℕ) (h : 0 < n) : ∃ (k : ℕ), 
  (∃ (m : ℕ), k = m * 5^n) ∧ 
  (∀ (d : ℕ), (d = (k / (10^(n-1))) % 10) → d % 2 = 1) :=
sorry

end NUMINAMATH_GPT_exists_odd_digit_div_by_five_power_l2245_224559


namespace NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l2245_224544

theorem tenth_term_arithmetic_sequence :
  let a_1 := (1 : ℝ) / 2
  let a_2 := (5 : ℝ) / 6
  let d := a_2 - a_1
  (a_1 + 9 * d) = 7 / 2 := 
by
  sorry

end NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l2245_224544


namespace NUMINAMATH_GPT_solve_equation_l2245_224547

noncomputable def equation (x : ℝ) : Prop :=
  -2 * x ^ 3 = (5 * x ^ 2 + 2) / (2 * x - 1)

theorem solve_equation (x : ℝ) :
  equation x ↔ (x = (1 + Real.sqrt 17) / 4 ∨ x = (1 - Real.sqrt 17) / 4) :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l2245_224547


namespace NUMINAMATH_GPT_value_of_expression_l2245_224521

theorem value_of_expression :
  (3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5) = 2000 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_expression_l2245_224521


namespace NUMINAMATH_GPT_water_content_in_boxes_l2245_224507

noncomputable def totalWaterInBoxes (num_boxes : ℕ) (bottles_per_box : ℕ) (capacity_per_bottle : ℚ) (fill_fraction : ℚ) : ℚ :=
  num_boxes * bottles_per_box * capacity_per_bottle * fill_fraction

theorem water_content_in_boxes :
  totalWaterInBoxes 10 50 12 (3 / 4) = 4500 := 
by
  sorry

end NUMINAMATH_GPT_water_content_in_boxes_l2245_224507


namespace NUMINAMATH_GPT_number_of_students_l2245_224583

theorem number_of_students
    (average_marks : ℕ)
    (wrong_mark : ℕ)
    (correct_mark : ℕ)
    (correct_average_marks : ℕ)
    (h1 : average_marks = 100)
    (h2 : wrong_mark = 50)
    (h3 : correct_mark = 10)
    (h4 : correct_average_marks = 96)
  : ∃ n : ℕ, (100 * n - 40) / n = 96 ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_students_l2245_224583


namespace NUMINAMATH_GPT_simplified_sum_l2245_224577

def exp1 := -( -1 ^ 2006 )
def exp2 := -( -1 ^ 2007 )
def exp3 := -( 1 ^ 2008 )
def exp4 := -( -1 ^ 2009 )

theorem simplified_sum : 
  exp1 + exp2 + exp3 + exp4 = 0 := 
by 
  sorry

end NUMINAMATH_GPT_simplified_sum_l2245_224577


namespace NUMINAMATH_GPT_unicorn_rope_problem_l2245_224580

theorem unicorn_rope_problem
  (d e f : ℕ)
  (h_prime_f : Prime f)
  (h_d : d = 75)
  (h_e : e = 450)
  (h_f : f = 3)
  : d + e + f = 528 := by
  sorry

end NUMINAMATH_GPT_unicorn_rope_problem_l2245_224580


namespace NUMINAMATH_GPT_sum_first_5n_l2245_224532

theorem sum_first_5n (n : ℕ) (h : (3 * n * (3 * n + 1)) / 2 = (n * (n + 1)) / 2 + 210) : 
  (5 * n * (5 * n + 1)) / 2 = 630 :=
sorry

end NUMINAMATH_GPT_sum_first_5n_l2245_224532


namespace NUMINAMATH_GPT_zero_in_interval_l2245_224575

def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem zero_in_interval :
  (f 0 < 0) → (f 0.5 > 0) → (f 0.25 < 0) → ∃ x, 0.25 < x ∧ x < 0.5 ∧ f x = 0 :=
by
  intro h0 h05 h025
  -- This is just the statement; the proof is not required as per instructions
  sorry

end NUMINAMATH_GPT_zero_in_interval_l2245_224575


namespace NUMINAMATH_GPT_inequality_satisfied_for_a_l2245_224557

theorem inequality_satisfied_for_a (a : ℝ) :
  (∀ x : ℝ, |2 * x - a| + |3 * x - 2 * a| ≥ a^2) ↔ -1/3 ≤ a ∧ a ≤ 1/3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_satisfied_for_a_l2245_224557


namespace NUMINAMATH_GPT_scientific_notation_of_taichulight_performance_l2245_224531

noncomputable def trillion := 10^12

def convert_to_scientific_notation (x : ℝ) (n : ℤ) : Prop :=
  1 ≤ x ∧ x < 10 ∧ x * 10^n = 12.5 * trillion

theorem scientific_notation_of_taichulight_performance :
  ∃ (x : ℝ) (n : ℤ), convert_to_scientific_notation x n ∧ x = 1.25 ∧ n = 13 :=
by
  unfold convert_to_scientific_notation
  use 1.25
  use 13
  sorry

end NUMINAMATH_GPT_scientific_notation_of_taichulight_performance_l2245_224531


namespace NUMINAMATH_GPT_range_of_y_l2245_224554

theorem range_of_y (y : ℝ) (h1 : y < 0) (h2 : Int.ceil y * Int.floor y = 72) : 
  -9 < y ∧ y < -8 :=
sorry

end NUMINAMATH_GPT_range_of_y_l2245_224554


namespace NUMINAMATH_GPT_evaluate_expression_l2245_224500

theorem evaluate_expression : 4 * (9 - 3)^2 - 8 = 136 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2245_224500


namespace NUMINAMATH_GPT_length_increase_percentage_l2245_224585

theorem length_increase_percentage
  (L W : ℝ)
  (A : ℝ := L * W)
  (A' : ℝ := 1.30000000000000004 * A)
  (new_length : ℝ := L * (1 + x / 100))
  (new_width : ℝ := W / 2)
  (area_equiv : new_length * new_width = A')
  (x : ℝ) :
  1 + x / 100 = 2.60000000000000008 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_length_increase_percentage_l2245_224585


namespace NUMINAMATH_GPT_average_length_of_strings_l2245_224542

theorem average_length_of_strings : 
  let length1 := 2
  let length2 := 5
  let length3 := 3
  let total_length := length1 + length2 + length3 
  let average_length := total_length / 3
  average_length = 10 / 3 :=
by
  let length1 := 2
  let length2 := 5
  let length3 := 3
  let total_length := length1 + length2 + length3
  let average_length := total_length / 3
  have h1 : total_length = 10 := by rfl
  have h2 : average_length = 10 / 3 := by rfl
  exact h2

end NUMINAMATH_GPT_average_length_of_strings_l2245_224542


namespace NUMINAMATH_GPT_range_of_a_l2245_224512

theorem range_of_a {
  a : ℝ
} :
  (∀ x ∈ Set.Ici (2 : ℝ), (x^2 + (2 - a) * x + 4 - 2 * a) > 0) ↔ a < 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2245_224512


namespace NUMINAMATH_GPT_integer_roots_count_l2245_224596

theorem integer_roots_count (b c d e f : ℚ) :
  ∃ (n : ℕ), (n = 0 ∨ n = 1 ∨ n = 2 ∨ n = 4 ∨ n = 5) ∧
  (∃ (r : ℕ → ℤ), ∀ i, i < n → (∀ z : ℤ, (∃ m, z = r m) → (z^5 + b * z^4 + c * z^3 + d * z^2 + e * z + f = 0))) :=
sorry

end NUMINAMATH_GPT_integer_roots_count_l2245_224596


namespace NUMINAMATH_GPT_no_same_last_four_digits_of_powers_of_five_and_six_l2245_224569

theorem no_same_last_four_digits_of_powers_of_five_and_six : 
  ¬ ∃ (n m : ℕ), 0 < n ∧ 0 < m ∧ (5 ^ n % 10000 = 6 ^ m % 10000) := 
by 
  sorry

end NUMINAMATH_GPT_no_same_last_four_digits_of_powers_of_five_and_six_l2245_224569


namespace NUMINAMATH_GPT_contrapositive_even_contrapositive_not_even_l2245_224584

theorem contrapositive_even (x y : ℤ) : 
  (∃ a b : ℤ, x = 2*a ∧ y = 2*b)  → (∃ c : ℤ, x + y = 2*c) :=
sorry

theorem contrapositive_not_even (x y : ℤ) :
  (¬ ∃ c : ℤ, x + y = 2*c) → (¬ ∃ a b : ℤ, x = 2*a ∧ y = 2*b) :=
sorry

end NUMINAMATH_GPT_contrapositive_even_contrapositive_not_even_l2245_224584


namespace NUMINAMATH_GPT_sum_of_distinct_integers_eq_36_l2245_224550

theorem sum_of_distinct_integers_eq_36
  (p q r s t : ℤ)
  (hpq : p ≠ q) (hpr : p ≠ r) (hps : p ≠ s) (hpt : p ≠ t)
  (hqr : q ≠ r) (hqs : q ≠ s) (hqt : q ≠ t)
  (hrs : r ≠ s) (hrt : r ≠ t)
  (hst : s ≠ t)
  (h : (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 80) :
  p + q + r + s + t = 36 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_distinct_integers_eq_36_l2245_224550


namespace NUMINAMATH_GPT_rectangle_area_l2245_224533

open Real

theorem rectangle_area (A : ℝ) (s l w : ℝ) (h1 : A = 9 * sqrt 3) (h2 : A = (sqrt 3 / 4) * s^2)
  (h3 : w = s) (h4 : l = 3 * w) : w * l = 108 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l2245_224533


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l2245_224539

theorem parabola_focus_coordinates :
  ∃ h k : ℝ, (y = -1/8 * x^2 + 2 * x - 1) ∧ (h = 8 ∧ k = 5) :=
sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l2245_224539


namespace NUMINAMATH_GPT_derivative_f_at_pi_l2245_224593

noncomputable def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem derivative_f_at_pi : (deriv f π) = -1 := 
by
  sorry

end NUMINAMATH_GPT_derivative_f_at_pi_l2245_224593


namespace NUMINAMATH_GPT_classroom_students_count_l2245_224505

theorem classroom_students_count (b g : ℕ) (hb : 3 * g = 5 * b) (hg : g = b + 4) : b + g = 16 :=
by
  sorry

end NUMINAMATH_GPT_classroom_students_count_l2245_224505


namespace NUMINAMATH_GPT_faye_money_left_is_30_l2245_224587

-- Definitions and conditions
def initial_money : ℝ := 20
def mother_gave (initial : ℝ) : ℝ := 2 * initial
def cost_of_cupcakes : ℝ := 10 * 1.5
def cost_of_cookies : ℝ := 5 * 3

-- Calculate the total money Faye has left
def total_money_left (initial : ℝ) (mother_gave_ : ℝ) (cost_cupcakes : ℝ) (cost_cookies : ℝ) : ℝ :=
  initial + mother_gave_ - (cost_cupcakes + cost_cookies)

-- Theorem stating the money left
theorem faye_money_left_is_30 :
  total_money_left initial_money (mother_gave initial_money) cost_of_cupcakes cost_of_cookies = 30 :=
by sorry

end NUMINAMATH_GPT_faye_money_left_is_30_l2245_224587


namespace NUMINAMATH_GPT_vector_dot_product_l2245_224517

-- Definitions based on the given conditions
variables (A B C M : ℝ)  -- points in 2D or 3D space can be generalized as real numbers for simplicity
variables (BA BC BM : ℝ) -- vector magnitudes
variables (AC : ℝ) -- magnitude of AC

-- Hypotheses from the problem conditions
variable (hM : 2 * BM = BA + BC)  -- M is the midpoint of AC
variable (hAC : AC = 4)
variable (hBM : BM = 3)

-- Theorem statement asserting the desired result
theorem vector_dot_product :
  BA * BC = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_vector_dot_product_l2245_224517


namespace NUMINAMATH_GPT_inequality_proof_l2245_224588

open Real

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_product : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_proof_l2245_224588


namespace NUMINAMATH_GPT_dividend_correct_l2245_224538

-- Given constants for the problem
def divisor := 19
def quotient := 7
def remainder := 6

-- Dividend formula
def dividend := (divisor * quotient) + remainder

-- The proof problem statement
theorem dividend_correct : dividend = 139 := by
  sorry

end NUMINAMATH_GPT_dividend_correct_l2245_224538


namespace NUMINAMATH_GPT_parallelogram_angle_sum_l2245_224582

theorem parallelogram_angle_sum (ABCD : Type) (A B C D : ABCD) 
  (angle : ABCD → ℝ) (h_parallelogram : true) (h_B : angle B = 60) :
  ¬ (angle C + angle A = 180) :=
sorry

end NUMINAMATH_GPT_parallelogram_angle_sum_l2245_224582


namespace NUMINAMATH_GPT_determine_placemat_length_l2245_224536

theorem determine_placemat_length :
  ∃ (y : ℝ), ∀ (r : ℝ), r = 5 →
  (∀ (n : ℕ), n = 8 →
  (∀ (w : ℝ), w = 1 →
  y = 10 * Real.sin (5 * Real.pi / 16))) :=
by
  sorry

end NUMINAMATH_GPT_determine_placemat_length_l2245_224536


namespace NUMINAMATH_GPT_solveExpression_l2245_224589

noncomputable def evaluateExpression : ℝ := (Real.sqrt 3) / Real.sin (Real.pi / 9) - 1 / Real.sin (7 * Real.pi / 18)

theorem solveExpression : evaluateExpression = 4 :=
by sorry

end NUMINAMATH_GPT_solveExpression_l2245_224589


namespace NUMINAMATH_GPT_inequality_solution_set_l2245_224522

   theorem inequality_solution_set : 
     {x : ℝ | (4 * x - 5)^2 + (3 * x - 2)^2 < (x - 3)^2} = {x : ℝ | (2 / 3 : ℝ) < x ∧ x < (5 / 4 : ℝ)} :=
   by
     sorry
   
end NUMINAMATH_GPT_inequality_solution_set_l2245_224522


namespace NUMINAMATH_GPT_total_calories_burned_l2245_224598

def base_distance : ℝ := 15
def records : List ℝ := [0.1, -0.8, 0.9, 16.5 - base_distance, 2.0, -1.5, 14.1 - base_distance, 1.0, 0.8, -1.1]
def calorie_burn_rate : ℝ := 20

theorem total_calories_burned :
  (base_distance * 10 + (List.sum records)) * calorie_burn_rate = 3040 :=
by
  sorry

end NUMINAMATH_GPT_total_calories_burned_l2245_224598


namespace NUMINAMATH_GPT_john_has_leftover_bulbs_l2245_224516

-- Definitions of the problem statements
def initial_bulbs : ℕ := 40
def used_bulbs : ℕ := 16
def remaining_bulbs_after_use : ℕ := initial_bulbs - used_bulbs
def given_to_friend : ℕ := remaining_bulbs_after_use / 2

-- Statement to prove
theorem john_has_leftover_bulbs :
  remaining_bulbs_after_use - given_to_friend = 12 :=
by
  sorry

end NUMINAMATH_GPT_john_has_leftover_bulbs_l2245_224516


namespace NUMINAMATH_GPT_tony_fish_after_ten_years_l2245_224590

theorem tony_fish_after_ten_years :
  let initial_fish := 6
  let x := [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
  let y := [4, 3, 2, 1, 2, 3, 4, 5, 6, 7]
  (List.foldl (fun acc ⟨add, die⟩ => acc + add - die) initial_fish (List.zip x y)) = 34 := 
by
  sorry

end NUMINAMATH_GPT_tony_fish_after_ten_years_l2245_224590


namespace NUMINAMATH_GPT_gcd_of_three_numbers_l2245_224571

-- Define the given numbers
def a := 72
def b := 120
def c := 168

-- Define the GCD function and prove the required statement
theorem gcd_of_three_numbers : Nat.gcd (Nat.gcd a b) c = 24 := by
  -- Intermediate steps and their justifications would go here in the proof, but we are putting sorry
  sorry

end NUMINAMATH_GPT_gcd_of_three_numbers_l2245_224571


namespace NUMINAMATH_GPT_number_of_blue_lights_l2245_224570

-- Conditions
def total_colored_lights : Nat := 95
def red_lights : Nat := 26
def yellow_lights : Nat := 37
def blue_lights : Nat := total_colored_lights - (red_lights + yellow_lights)

-- Statement we need to prove
theorem number_of_blue_lights : blue_lights = 32 := by
  sorry

end NUMINAMATH_GPT_number_of_blue_lights_l2245_224570


namespace NUMINAMATH_GPT_contractor_fired_two_people_l2245_224510

theorem contractor_fired_two_people
  (total_days : ℕ) (initial_people : ℕ) (days_worked : ℕ) (fraction_completed : ℚ)
  (remaining_days : ℕ) (people_fired : ℕ)
  (h1 : total_days = 100)
  (h2 : initial_people = 10)
  (h3 : days_worked = 20)
  (h4 : fraction_completed = 1/4)
  (h5 : remaining_days = 75)
  (h6 : remaining_days + days_worked = total_days)
  (h7 : people_fired = initial_people - 8) :
  people_fired = 2 :=
  sorry

end NUMINAMATH_GPT_contractor_fired_two_people_l2245_224510


namespace NUMINAMATH_GPT_crows_cannot_be_on_same_tree_l2245_224548

theorem crows_cannot_be_on_same_tree :
  (∀ (trees : ℕ) (crows : ℕ),
   trees = 22 ∧ crows = 22 →
   (∀ (positions : ℕ → ℕ),
    (∀ i, 1 ≤ positions i ∧ positions i ≤ 2) →
    ∀ (move : (ℕ → ℕ) → (ℕ → ℕ)),
    (∀ (pos : ℕ → ℕ) (i : ℕ),
     move pos i = pos i + positions (i + 1) ∨ move pos i = pos i - positions (i + 1)) →
    (∀ (pos : ℕ → ℕ) (i : ℕ),
     pos i % trees = (move pos i) % trees) →
    ¬ (∃ (final_pos : ℕ → ℕ),
      (∀ i, final_pos i = 0 ∨ final_pos i = 22) ∧
      (∀ i j, final_pos i = final_pos j)
    )
  )
) :=
sorry

end NUMINAMATH_GPT_crows_cannot_be_on_same_tree_l2245_224548


namespace NUMINAMATH_GPT_road_length_kopatych_to_losyash_l2245_224527

variable (T Krosh_dist Yozhik_dist : ℕ)
variable (d_k d_y r_k r_y : ℕ)

theorem road_length_kopatych_to_losyash : 
    (d_k = 20) → (d_y = 16) → (r_k = 30) → (r_y = 60) → 
    (Krosh_dist = 5 * T / 9) → (Yozhik_dist = 4 * T / 9) → 
    (T = Krosh_dist + r_k) →
    (T = Yozhik_dist + r_y) → 
    (T = 180) :=
by
  intros
  sorry

end NUMINAMATH_GPT_road_length_kopatych_to_losyash_l2245_224527


namespace NUMINAMATH_GPT_factorize_expression_l2245_224534

theorem factorize_expression (m n : ℤ) : m^2 * n - 9 * n = n * (m + 3) * (m - 3) := by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2245_224534


namespace NUMINAMATH_GPT_roots_quadratic_identity_l2245_224525

theorem roots_quadratic_identity :
  ∀ (r s : ℝ), (r^2 - 5 * r + 3 = 0) ∧ (s^2 - 5 * s + 3 = 0) → r^2 + s^2 = 19 :=
by
  intros r s h
  sorry

end NUMINAMATH_GPT_roots_quadratic_identity_l2245_224525


namespace NUMINAMATH_GPT_m_range_l2245_224586

/-- Given a point (x, y) on the circle x^2 + (y - 1)^2 = 2, show that the real number m,
such that x + y + m ≥ 0, must satisfy m ≥ 1. -/
theorem m_range (x y m : ℝ) (h₁ : x^2 + (y - 1)^2 = 2) (h₂ : x + y + m ≥ 0) : m ≥ 1 :=
sorry

end NUMINAMATH_GPT_m_range_l2245_224586


namespace NUMINAMATH_GPT_total_sum_of_grid_is_745_l2245_224508

theorem total_sum_of_grid_is_745 :
  let top_row := [12, 13, 15, 17, 19]
  let left_column := [12, 14, 16, 18]
  let total_sum := 360 + 375 + 10
  total_sum = 745 :=
by
  -- The theorem establishes the total sum calculation.
  sorry

end NUMINAMATH_GPT_total_sum_of_grid_is_745_l2245_224508


namespace NUMINAMATH_GPT_doll_cost_l2245_224514

theorem doll_cost (D : ℝ) (h : 4 * D = 60) : D = 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_doll_cost_l2245_224514


namespace NUMINAMATH_GPT_compute_f_sum_l2245_224556

noncomputable def f : ℝ → ℝ := sorry -- placeholder for f(x)

variables (x : ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 2) = f x
axiom f_definition : ∀ x, 0 < x ∧ x < 1 → f x = x^2

-- Prove the main statement
theorem compute_f_sum : f (-3 / 2) + f 1 = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_compute_f_sum_l2245_224556


namespace NUMINAMATH_GPT_cake_shop_problem_l2245_224579

theorem cake_shop_problem :
  ∃ (N n K : ℕ), (N - n * K = 6) ∧ (N = (n - 1) * 8 + 1) ∧ (N = 97) :=
by
  sorry

end NUMINAMATH_GPT_cake_shop_problem_l2245_224579


namespace NUMINAMATH_GPT_intersection_of_sets_union_of_complement_and_set_l2245_224565

def set1 := { x : ℝ | -1 < x ∧ x < 2 }
def set2 := { x : ℝ | x > 0 }
def complement_set2 := { x : ℝ | x ≤ 0 }
def intersection_set := { x : ℝ | 0 < x ∧ x < 2 }
def union_set := { x : ℝ | x < 2 }

theorem intersection_of_sets : 
  { x : ℝ | x ∈ set1 ∧ x ∈ set2 } = intersection_set := 
by 
  sorry

theorem union_of_complement_and_set : 
  { x : ℝ | x ∈ complement_set2 ∨ x ∈ set1 } = union_set := 
by 
  sorry

end NUMINAMATH_GPT_intersection_of_sets_union_of_complement_and_set_l2245_224565


namespace NUMINAMATH_GPT_fish_weight_l2245_224597

variables (H T X : ℝ)
-- Given conditions
def tail_weight : Prop := X = 1
def head_weight : Prop := H = X + 0.5 * T
def torso_weight : Prop := T = H + X

theorem fish_weight (H T X : ℝ) 
  (h_tail : tail_weight X)
  (h_head : head_weight H T X)
  (h_torso : torso_weight H T X) : 
  H + T + X = 8 :=
sorry

end NUMINAMATH_GPT_fish_weight_l2245_224597


namespace NUMINAMATH_GPT_box_one_contains_at_least_one_ball_l2245_224518

-- Define the conditions
def boxes : List ℕ := [1, 2, 3, 4]
def balls : List ℕ := [1, 2, 3]

-- Define the problem
def count_ways_box_one_contains_ball :=
  let total_ways := (boxes.length)^(balls.length)
  let ways_box_one_empty := (boxes.length - 1)^(balls.length)
  total_ways - ways_box_one_empty

-- The proof problem statement
theorem box_one_contains_at_least_one_ball : count_ways_box_one_contains_ball = 37 := by
  sorry

end NUMINAMATH_GPT_box_one_contains_at_least_one_ball_l2245_224518


namespace NUMINAMATH_GPT_problem_I_problem_II_l2245_224530

-- Definitions
def p (x : ℝ) : Prop := (x + 2) * (x - 3) ≤ 0
def q (m : ℝ) (x : ℝ) : Prop := 1 - m ≤ x ∧ x ≤ 1 + m

-- Problem (I)
theorem problem_I (m : ℝ) : m > 0 → (∀ x : ℝ, q m x → p x) → 0 < m ∧ m ≤ 2 := by
  sorry

-- Problem (II)
theorem problem_II (x : ℝ) : 7 > 0 → 
  (p x ∨ q 7 x) ∧ ¬(p x ∧ q 7 x) → 
  (-6 ≤ x ∧ x < -2) ∨ (3 < x ∧ x ≤ 8) := by
  sorry

end NUMINAMATH_GPT_problem_I_problem_II_l2245_224530


namespace NUMINAMATH_GPT_total_games_across_leagues_l2245_224549

-- Defining the conditions for the leagues
def leagueA_teams := 20
def leagueB_teams := 25
def leagueC_teams := 30

-- Function to calculate the number of games in a round-robin tournament
def number_of_games (n : ℕ) := n * (n - 1) / 2

-- Proposition to prove total games across all leagues
theorem total_games_across_leagues :
  number_of_games leagueA_teams + number_of_games leagueB_teams + number_of_games leagueC_teams = 925 := by
  sorry

end NUMINAMATH_GPT_total_games_across_leagues_l2245_224549


namespace NUMINAMATH_GPT_compare_magnitudes_proof_l2245_224528

noncomputable def compare_magnitudes (a b c : ℝ) (ha : 0 < a) (hbc : b * c > a^2) (heq : a^2 - 2 * a * b + c^2 = 0) : Prop :=
  b > c ∧ c > a ∧ b > a

theorem compare_magnitudes_proof (a b c : ℝ) (ha : 0 < a) (hbc : b * c > a^2) (heq : a^2 - 2 * a * b + c^2 = 0) :
  compare_magnitudes a b c ha hbc heq :=
sorry

end NUMINAMATH_GPT_compare_magnitudes_proof_l2245_224528


namespace NUMINAMATH_GPT_find_pairs_1984_l2245_224555

theorem find_pairs_1984 (m n : ℕ) :
  19 * m + 84 * n = 1984 ↔ (m = 100 ∧ n = 1) ∨ (m = 16 ∧ n = 20) :=
by
  sorry

end NUMINAMATH_GPT_find_pairs_1984_l2245_224555


namespace NUMINAMATH_GPT_trig_identity_l2245_224524

theorem trig_identity (α : ℝ) 
  (h : Real.tan (α - Real.pi / 4) = 1 / 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 2 := 
sorry

end NUMINAMATH_GPT_trig_identity_l2245_224524


namespace NUMINAMATH_GPT_area_of_circle_l2245_224599

def circle_area (x y : ℝ) : Prop := x^2 + y^2 - 8 * x + 18 * y = -45

theorem area_of_circle :
  (∃ x y : ℝ, circle_area x y) → ∃ A : ℝ, A = 52 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_area_of_circle_l2245_224599


namespace NUMINAMATH_GPT_floor_plus_ceil_eq_seven_l2245_224578

theorem floor_plus_ceil_eq_seven (x : ℝ) :
  (⌊x⌋ + ⌈x⌉ = 7) ↔ (3 < x ∧ x < 4) :=
sorry

end NUMINAMATH_GPT_floor_plus_ceil_eq_seven_l2245_224578


namespace NUMINAMATH_GPT_no_real_solution_for_inequality_l2245_224526

theorem no_real_solution_for_inequality :
  ∀ x : ℝ, ¬(3 * x^2 - x + 2 < 0) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_for_inequality_l2245_224526


namespace NUMINAMATH_GPT_no_common_points_l2245_224591

theorem no_common_points : 
  ∀ (x y : ℝ), ¬(x^2 + y^2 = 9 ∧ x^2 + y^2 = 4) := 
by
  sorry

end NUMINAMATH_GPT_no_common_points_l2245_224591


namespace NUMINAMATH_GPT_max_m_sq_plus_n_sq_l2245_224535

theorem max_m_sq_plus_n_sq (m n : ℤ) (h1 : 1 ≤ m ∧ m ≤ 1981) (h2 : 1 ≤ n ∧ n ≤ 1981) (h3 : (n^2 - m*n - m^2)^2 = 1) : m^2 + n^2 ≤ 3524578 :=
sorry

end NUMINAMATH_GPT_max_m_sq_plus_n_sq_l2245_224535


namespace NUMINAMATH_GPT_dot_product_property_l2245_224509

-- Definitions based on conditions
def vec_a : ℝ × ℝ := (2, -1)
def vec_b : ℝ × ℝ := (-1, 2)
def scalar_mult (c : ℝ) (v : ℝ × ℝ) := (c * v.1, c * v.2)
def vec_add (v1 v2 : ℝ × ℝ) := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2

-- Required property
theorem dot_product_property : dot_product (vec_add (scalar_mult 2 vec_a) vec_b) vec_a = 6 :=
by sorry

end NUMINAMATH_GPT_dot_product_property_l2245_224509


namespace NUMINAMATH_GPT_return_trip_time_l2245_224513

theorem return_trip_time (d p w : ℝ) (h1 : d = 84 * (p - w)) (h2 : d / (p + w) = d / p - 9) :
  (d / (p + w) = 63) ∨ (d / (p + w) = 12) :=
by
  sorry

end NUMINAMATH_GPT_return_trip_time_l2245_224513


namespace NUMINAMATH_GPT_mean_proportional_of_segments_l2245_224563

theorem mean_proportional_of_segments (a b c : ℝ) (a_val : a = 2) (b_val : b = 6) :
  c = 2 * Real.sqrt 3 ↔ c*c = a * b := by
  sorry

end NUMINAMATH_GPT_mean_proportional_of_segments_l2245_224563


namespace NUMINAMATH_GPT_rectangle_area_is_12_l2245_224558

noncomputable def rectangle_area_proof (w l y : ℝ) : Prop :=
  l = 3 * w ∧ 2 * (l + w) = 16 ∧ (l^2 + w^2 = y^2) → l * w = 12

theorem rectangle_area_is_12 (y : ℝ) : ∃ (w l : ℝ), rectangle_area_proof w l y :=
by
  -- Introducing variables
  exists 2
  exists 6
  -- Constructing proof steps (skipped here with sorry)
  sorry

end NUMINAMATH_GPT_rectangle_area_is_12_l2245_224558


namespace NUMINAMATH_GPT_correct_age_equation_l2245_224566

variable (x : ℕ)

def age_older_brother (x : ℕ) : ℕ := 2 * x
def age_younger_brother_six_years_ago (x : ℕ) : ℕ := x - 6
def age_older_brother_six_years_ago (x : ℕ) : ℕ := 2 * x - 6

theorem correct_age_equation (h1 : age_younger_brother_six_years_ago x + age_older_brother_six_years_ago x = 15) :
  (x - 6) + (2 * x - 6) = 15 :=
by
  sorry

end NUMINAMATH_GPT_correct_age_equation_l2245_224566


namespace NUMINAMATH_GPT_inequality_x_y_z_squares_l2245_224506

theorem inequality_x_y_z_squares (x y z m : ℝ) (h : x + y + z = m) : x^2 + y^2 + z^2 ≥ (m^2) / 3 := by
  sorry

end NUMINAMATH_GPT_inequality_x_y_z_squares_l2245_224506


namespace NUMINAMATH_GPT_heptagon_angle_sum_l2245_224560

theorem heptagon_angle_sum 
  (angle_A angle_B angle_C angle_D angle_E angle_F angle_G : ℝ) 
  (h : angle_A + angle_B + angle_C + angle_D + angle_E + angle_F + angle_G = 540) :
  angle_A + angle_B + angle_C + angle_D + angle_E + angle_F + angle_G = 540 :=
by
  sorry

end NUMINAMATH_GPT_heptagon_angle_sum_l2245_224560


namespace NUMINAMATH_GPT_minimum_value_expression_l2245_224572

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^2 + 5*x + 2) * (y^2 + 5*y + 2) * (z^2 + 5*z + 2) / (x * y * z) ≥ 343 :=
sorry

end NUMINAMATH_GPT_minimum_value_expression_l2245_224572


namespace NUMINAMATH_GPT_perpendicular_lines_m_value_l2245_224501

def is_perpendicular (m : ℝ) : Prop :=
    let slope1 := 1 / 2
    let slope2 := -2 / m
    slope1 * slope2 = -1

theorem perpendicular_lines_m_value (m : ℝ) (h : is_perpendicular m) : m = 1 := by
    sorry

end NUMINAMATH_GPT_perpendicular_lines_m_value_l2245_224501


namespace NUMINAMATH_GPT_students_catching_up_on_homework_l2245_224592

def total_students : ℕ := 24
def silent_reading_students : ℕ := total_students / 2
def board_games_students : ℕ := total_students / 3

theorem students_catching_up_on_homework : 
  total_students - (silent_reading_students + board_games_students) = 4 := by
  sorry

end NUMINAMATH_GPT_students_catching_up_on_homework_l2245_224592


namespace NUMINAMATH_GPT_tim_minus_tom_l2245_224553

def sales_tax_rate : ℝ := 0.07
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def city_tax_rate : ℝ := 0.05

noncomputable def tim_total : ℝ :=
  let price_with_tax := original_price * (1 + sales_tax_rate)
  price_with_tax * (1 - discount_rate)

noncomputable def tom_total : ℝ :=
  let discounted_price := original_price * (1 - discount_rate)
  let price_with_sales_tax := discounted_price * (1 + sales_tax_rate)
  price_with_sales_tax * (1 + city_tax_rate)

theorem tim_minus_tom : tim_total - tom_total = -4.82 := 
by sorry

end NUMINAMATH_GPT_tim_minus_tom_l2245_224553


namespace NUMINAMATH_GPT_robot_cost_l2245_224573

theorem robot_cost (num_friends : ℕ) (total_tax change start_money : ℝ) (h_friends : num_friends = 7) (h_tax : total_tax = 7.22) (h_change : change = 11.53) (h_start : start_money = 80) :
  let spent_money := start_money - change
  let cost_robots := spent_money - total_tax
  let cost_per_robot := cost_robots / num_friends
  cost_per_robot = 8.75 :=
by
  sorry

end NUMINAMATH_GPT_robot_cost_l2245_224573


namespace NUMINAMATH_GPT_solve_system_of_equations_l2245_224546

theorem solve_system_of_equations (x y z : ℝ) :
  (x * y + 1 = 2 * z) →
  (y * z + 1 = 2 * x) →
  (z * x + 1 = 2 * y) →
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  ((x = -2 ∧ y = -2 ∧ z = 5/2) ∨
   (x = 5/2 ∧ y = -2 ∧ z = -2) ∨ 
   (x = -2 ∧ y = 5/2 ∧ z = -2)) :=
sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2245_224546


namespace NUMINAMATH_GPT_min_sum_sequence_n_l2245_224545

theorem min_sum_sequence_n (S : ℕ → ℤ) (h : ∀ n, S n = n * n - 48 * n) : 
  ∃ n, n = 24 ∧ ∀ m, S n ≤ S m :=
by
  sorry

end NUMINAMATH_GPT_min_sum_sequence_n_l2245_224545


namespace NUMINAMATH_GPT_pick_theorem_l2245_224595

def lattice_polygon (vertices : List (ℤ × ℤ)) : Prop :=
  ∀ v ∈ vertices, ∃ i j : ℤ, v = (i, j)

variables {n m : ℕ}
variables {A : ℤ}
variables {vertices : List (ℤ × ℤ)}

def lattice_point_count_inside (vertices : List (ℤ × ℤ)) : ℕ :=
  -- Placeholder for the actual logic to count inside points
  sorry

def lattice_point_count_boundary (vertices : List (ℤ × ℤ)) : ℕ :=
  -- Placeholder for the actual logic to count boundary points
  sorry

theorem pick_theorem (h : lattice_polygon vertices) :
  lattice_point_count_inside vertices = n → 
  lattice_point_count_boundary vertices = m → 
  A = n + m / 2 - 1 :=
sorry

end NUMINAMATH_GPT_pick_theorem_l2245_224595


namespace NUMINAMATH_GPT_sum_of_coordinates_of_D_l2245_224561

/--
Given points A = (4,8), B = (2,4), C = (6,6), and D = (a,b) in the first quadrant, if the quadrilateral formed by joining the midpoints of the segments AB, BC, CD, and DA is a square with sides inclined at 45 degrees to the x-axis, then the sum of the coordinates of point D is 6.
-/
theorem sum_of_coordinates_of_D 
  (a b : ℝ)
  (h_quadrilateral : ∃ A B C D : Prod ℝ ℝ, 
    A = (4, 8) ∧ B = (2, 4) ∧ C = (6, 6) ∧ D = (a, b) ∧ 
    ∃ M1 M2 M3 M4 : Prod ℝ ℝ,
    M1 = ((4 + 2) / 2, (8 + 4) / 2) ∧ M2 = ((2 + 6) / 2, (4 + 6) / 2) ∧ 
    M3 = (M2.1 + 1, M2.2 - 1) ∧ M4 = (M3.1 + 1, M3.2 + 1) ∧ 
    M3 = ((a + 6) / 2, (b + 6) / 2) ∧ M4 = ((a + 4) / 2, (b + 8) / 2)
  ) : 
  a + b = 6 := sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_D_l2245_224561


namespace NUMINAMATH_GPT_principal_amount_correct_l2245_224576

-- Define the given conditions and quantities
def P : ℝ := 1054.76
def final_amount : ℝ := 1232.0
def rate1 : ℝ := 0.05
def rate2 : ℝ := 0.07
def rate3 : ℝ := 0.04

-- Define the statement we want to prove
theorem principal_amount_correct :
  final_amount = P * (1 + rate1) * (1 + rate2) * (1 + rate3) :=
sorry

end NUMINAMATH_GPT_principal_amount_correct_l2245_224576


namespace NUMINAMATH_GPT_ratio_sub_div_a_l2245_224511

theorem ratio_sub_div_a (a b : ℝ) (h : a / b = 5 / 8) : (b - a) / a = 3 / 5 :=
sorry

end NUMINAMATH_GPT_ratio_sub_div_a_l2245_224511


namespace NUMINAMATH_GPT_number_is_48_l2245_224581

theorem number_is_48 (x : ℝ) (h : (1/4) * x + 15 = 27) : x = 48 :=
by sorry

end NUMINAMATH_GPT_number_is_48_l2245_224581


namespace NUMINAMATH_GPT_grid_points_circumference_l2245_224567

def numGridPointsOnCircumference (R : ℝ) : ℕ := sorry

def isInteger (x : ℝ) : Prop := ∃ (n : ℤ), x = n

theorem grid_points_circumference (R : ℝ) (h : numGridPointsOnCircumference R = 1988) : 
  isInteger R ∨ isInteger (Real.sqrt 2 * R) :=
by
  sorry

end NUMINAMATH_GPT_grid_points_circumference_l2245_224567


namespace NUMINAMATH_GPT_books_in_library_final_l2245_224504

variable (initial_books : ℕ) (books_taken_out_tuesday : ℕ) 
          (books_returned_wednesday : ℕ) (books_taken_out_thursday : ℕ)

def books_left_in_library (initial_books books_taken_out_tuesday 
                          books_returned_wednesday books_taken_out_thursday : ℕ) : ℕ :=
  initial_books - books_taken_out_tuesday + books_returned_wednesday - books_taken_out_thursday

theorem books_in_library_final 
  (initial_books := 250) 
  (books_taken_out_tuesday := 120) 
  (books_returned_wednesday := 35) 
  (books_taken_out_thursday := 15) :
  books_left_in_library initial_books books_taken_out_tuesday 
                        books_returned_wednesday books_taken_out_thursday = 150 :=
by 
  sorry

end NUMINAMATH_GPT_books_in_library_final_l2245_224504


namespace NUMINAMATH_GPT_isosceles_trapezoid_area_l2245_224523

theorem isosceles_trapezoid_area (m h : ℝ) (hg : h = 3) (mg : m = 15) : 
  (m * h = 45) :=
by
  simp [hg, mg]
  sorry

end NUMINAMATH_GPT_isosceles_trapezoid_area_l2245_224523


namespace NUMINAMATH_GPT_single_elimination_games_l2245_224520

theorem single_elimination_games (n : ℕ) (h : n = 128) : (n - 1) = 127 :=
by
  sorry

end NUMINAMATH_GPT_single_elimination_games_l2245_224520


namespace NUMINAMATH_GPT_hash_triple_l2245_224562

def hash (N : ℝ) : ℝ := 0.5 * (N^2) + 1

theorem hash_triple  : hash (hash (hash 4)) = 862.125 :=
by {
  sorry
}

end NUMINAMATH_GPT_hash_triple_l2245_224562


namespace NUMINAMATH_GPT_abel_arrival_earlier_l2245_224519

variable (distance : ℕ) (speed_abel : ℕ) (speed_alice : ℕ) (start_delay_alice : ℕ)

theorem abel_arrival_earlier (h_dist : distance = 1000) 
                             (h_speed_abel : speed_abel = 50) 
                             (h_speed_alice : speed_alice = 40) 
                             (h_start_delay : start_delay_alice = 1) : 
                             (start_delay_alice + distance / speed_alice) * 60 - (distance / speed_abel) * 60 = 360 :=
by
  sorry

end NUMINAMATH_GPT_abel_arrival_earlier_l2245_224519


namespace NUMINAMATH_GPT_first_tier_price_level_is_10000_l2245_224543

noncomputable def first_tier_price_level (P : ℝ) : Prop :=
  ∀ (car_price : ℝ), car_price = 30000 → (P ≤ car_price ∧ 
    (0.25 * P + 0.15 * (car_price - P)) = 5500)

theorem first_tier_price_level_is_10000 :
  first_tier_price_level 10000 :=
by
  sorry

end NUMINAMATH_GPT_first_tier_price_level_is_10000_l2245_224543


namespace NUMINAMATH_GPT_find_a_l2245_224503

theorem find_a (a : ℝ) :
  (∀ (x y : ℝ), (x - a)^2 + y^2 = (x^2 + (y-1)^2)) ∧ (¬ ∃ x y : ℝ, y = x + 1) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2245_224503


namespace NUMINAMATH_GPT_monica_problem_l2245_224515

open Real

noncomputable def completingSquare : Prop :=
  ∃ (b c : ℤ), (∀ x : ℝ, (x - 4) ^ 2 = x^2 - 8 * x + 16) ∧ b = -4 ∧ c = 8 ∧ (b + c = 4)

theorem monica_problem : completingSquare := by
  sorry

end NUMINAMATH_GPT_monica_problem_l2245_224515


namespace NUMINAMATH_GPT_exam_questions_count_l2245_224502

theorem exam_questions_count (Q S : ℕ) 
    (hS : S = (4 * Q) / 5)
    (sergio_correct : Q - 4 = S + 6) : 
    Q = 50 :=
by 
  sorry

end NUMINAMATH_GPT_exam_questions_count_l2245_224502


namespace NUMINAMATH_GPT_combined_alloy_force_l2245_224594

-- Define the masses and forces exerted by Alloy A and Alloy B
def mass_A : ℝ := 6
def force_A : ℝ := 30
def mass_B : ℝ := 3
def force_B : ℝ := 10

-- Define the combined mass and force
def combined_mass : ℝ := mass_A + mass_B
def combined_force : ℝ := force_A + force_B

-- Theorem statement
theorem combined_alloy_force :
  combined_force = 40 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_combined_alloy_force_l2245_224594


namespace NUMINAMATH_GPT_find_a4_l2245_224537

variable (a : ℕ → ℤ)

def S (n : ℕ) : ℤ := (n * (a 1 + a n)) / 2

theorem find_a4 (h₁ : S 5 = 25) (h₂ : a 2 = 3) : a 4 = 7 := by
  sorry

end NUMINAMATH_GPT_find_a4_l2245_224537


namespace NUMINAMATH_GPT_min_val_x_2y_l2245_224551

noncomputable def min_x_2y (x y : ℝ) : ℝ :=
  x + 2 * y

theorem min_val_x_2y : 
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (1 / (x + 2) + 1 / (y + 2) = 1 / 3) → 
  min_x_2y x y ≥ 3 + 6 * Real.sqrt 2 :=
by
  intros x y x_pos y_pos eqn
  sorry

end NUMINAMATH_GPT_min_val_x_2y_l2245_224551


namespace NUMINAMATH_GPT_martha_total_clothes_l2245_224540

-- Define the conditions
def jackets_bought : ℕ := 4
def t_shirts_bought : ℕ := 9
def free_jacket_condition : ℕ := 2
def free_t_shirt_condition : ℕ := 3

-- Define calculations based on conditions
def free_jackets : ℕ := jackets_bought / free_jacket_condition
def free_t_shirts : ℕ := t_shirts_bought / free_t_shirt_condition
def total_jackets := jackets_bought + free_jackets
def total_t_shirts := t_shirts_bought + free_t_shirts
def total_clothes := total_jackets + total_t_shirts

-- Prove the total number of clothes
theorem martha_total_clothes : total_clothes = 18 :=
by
    sorry

end NUMINAMATH_GPT_martha_total_clothes_l2245_224540


namespace NUMINAMATH_GPT_ring_area_l2245_224541

theorem ring_area (r1 r2 : ℝ) (h1 : r1 = 12) (h2 : r2 = 5) : 
  (π * r1^2) - (π * r2^2) = 119 * π := 
by simp [h1, h2]; sorry

end NUMINAMATH_GPT_ring_area_l2245_224541


namespace NUMINAMATH_GPT_shaded_area_isosceles_right_triangle_l2245_224568

theorem shaded_area_isosceles_right_triangle (y : ℝ) :
  (∃ (x : ℝ), 2 * x^2 = y^2) ∧
  (∃ (A : ℝ), A = (1 / 2) * (y^2 / 2)) ∧
  (∃ (shaded_area : ℝ), shaded_area = (1 / 2) * (y^2 / 4)) →
  (shaded_area = y^2 / 8) :=
sorry

end NUMINAMATH_GPT_shaded_area_isosceles_right_triangle_l2245_224568
