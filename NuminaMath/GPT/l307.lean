import Mathlib

namespace ram_initial_deposit_l307_30732

theorem ram_initial_deposit :
  ∃ P: ℝ, P + 100 = 1100 ∧ 1.20 * 1100 = 1320 ∧ P * 1.32 = 1320 ∧ P = 1000 :=
by
  existsi (1000 : ℝ)
  sorry

end ram_initial_deposit_l307_30732


namespace sum_of_integers_l307_30711

theorem sum_of_integers (x y : ℤ) (h1 : x ^ 2 + y ^ 2 = 130) (h2 : x * y = 36) (h3 : x - y = 4) : x + y = 4 := 
by sorry

end sum_of_integers_l307_30711


namespace largest_consecutive_sum_55_l307_30788

theorem largest_consecutive_sum_55 : 
  ∃ k n : ℕ, k > 0 ∧ (∃ n : ℕ, 55 = k * n + (k * (k - 1)) / 2 ∧ k = 5) :=
sorry

end largest_consecutive_sum_55_l307_30788


namespace range_of_m_l307_30774

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 9 * x - m

theorem range_of_m (H : ∃ (x_0 : ℝ), x_0 ≠ 0 ∧ f 0 x_0 = f 0 x_0) : 0 < m ∧ m < 1 / 2 :=
sorry

end range_of_m_l307_30774


namespace find_divisor_l307_30759

theorem find_divisor {x y : ℤ} (h1 : (x - 5) / y = 7) (h2 : (x - 24) / 10 = 3) : y = 7 :=
by
  sorry

end find_divisor_l307_30759


namespace clothes_prices_l307_30792

theorem clothes_prices (total_cost : ℕ) (shirt_more : ℕ) (trousers_price : ℕ) (shirt_price : ℕ)
  (h1 : total_cost = 185)
  (h2 : shirt_more = 5)
  (h3 : shirt_price = 2 * trousers_price + shirt_more)
  (h4 : total_cost = shirt_price + trousers_price) : 
  trousers_price = 60 ∧ shirt_price = 125 :=
  by sorry

end clothes_prices_l307_30792


namespace sum_of_remaining_digit_is_correct_l307_30731

-- Define the local value calculation function for a particular digit with its place value
def local_value (digit place_value : ℕ) : ℕ := digit * place_value

-- Define the number in question
def number : ℕ := 2345

-- Define the local values for each digit in their respective place values
def local_value_2 : ℕ := local_value 2 1000
def local_value_3 : ℕ := local_value 3 100
def local_value_4 : ℕ := local_value 4 10
def local_value_5 : ℕ := local_value 5 1

-- Define the given sum of the local values
def given_sum : ℕ := 2345

-- Define the sum of the local values of the digits 2, 3, and 5
def sum_of_other_digits : ℕ := local_value_2 + local_value_3 + local_value_5

-- Define the target sum which is the sum of the local value of the remaining digit
def target_sum : ℕ := given_sum - sum_of_other_digits

-- Prove that the sum of the local value of the remaining digit is equal to 40
theorem sum_of_remaining_digit_is_correct : target_sum = 40 := 
by
  -- The proof will be provided here
  sorry

end sum_of_remaining_digit_is_correct_l307_30731


namespace find_x_l307_30786

theorem find_x (a b x : ℝ) (h : ∀ a b, a * b = a + 2 * b) (H : 3 * (4 * x) = 6) : x = -5 / 4 :=
by
  sorry

end find_x_l307_30786


namespace find_f_5pi_div_3_l307_30721

variable (f : ℝ → ℝ)

-- Define the conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_periodic_function (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

theorem find_f_5pi_div_3
  (h_odd : is_odd_function f)
  (h_periodic : is_periodic_function f π)
  (h_def : ∀ x, 0 ≤ x → x ≤ π/2 → f x = Real.sin x) :
  f (5 * π / 3) = - (Real.sqrt 3 / 2) := by
  sorry

end find_f_5pi_div_3_l307_30721


namespace two_points_same_color_at_distance_one_l307_30797

theorem two_points_same_color_at_distance_one (color : ℝ × ℝ → ℕ) (h : ∀p : ℝ × ℝ, color p < 3) :
  ∃ (p q : ℝ × ℝ), dist p q = 1 ∧ color p = color q :=
sorry

end two_points_same_color_at_distance_one_l307_30797


namespace prod_eq_of_eqs_l307_30702

variable (a : ℝ) (m n p q : ℕ)
variable (h1 : a ≠ 0) (h2 : a ≠ 1) (h3 : a ≠ -1)
variable (h4 : a^m + a^n = a^p + a^q) (h5 : a^{3*m} + a^{3*n} = a^{3*p} + a^{3*q})

theorem prod_eq_of_eqs : m * n = p * q := by
  sorry

end prod_eq_of_eqs_l307_30702


namespace ellipse_equation_l307_30706

theorem ellipse_equation {a b : ℝ} 
  (center_origin : ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 → x + y = 0)
  (foci_on_x : ∀ c : ℝ, c = a / 2)
  (perimeter_triangle : ∀ A B : ℝ, A + B + 2 * c = 16) :
  a = 4 ∧ b^2 = 12 → (∀ x y : ℝ, x^2/16 + y^2/12 = 1) :=
by
  sorry

end ellipse_equation_l307_30706


namespace bus_speed_excluding_stoppages_l307_30744

theorem bus_speed_excluding_stoppages (v : Real) 
  (h1 : ∀ x, x = 41) 
  (h2 : ∀ y, y = 14.444444444444443 / 60) : 
  v = 54 := 
by
  -- Proving the statement. Proof steps are skipped.
  sorry

end bus_speed_excluding_stoppages_l307_30744


namespace max_value_harmonic_series_l307_30791

theorem max_value_harmonic_series (k l m : ℕ) (hk : 0 < k) (hl : 0 < l) (hm : 0 < m)
  (h : 1/k + 1/l + 1/m < 1) : 
  (1/2 + 1/3 + 1/7) = 41/42 := 
sorry

end max_value_harmonic_series_l307_30791


namespace quadratic_has_solutions_l307_30701

theorem quadratic_has_solutions :
  (1 + Real.sqrt 2)^2 - 2 * (1 + Real.sqrt 2) - 1 = 0 ∧ 
  (1 - Real.sqrt 2)^2 - 2 * (1 - Real.sqrt 2) - 1 = 0 :=
by
  sorry

end quadratic_has_solutions_l307_30701


namespace range_of_a_l307_30726

noncomputable def f : ℝ → ℝ := sorry

variables (a : ℝ)
variable (is_even : ∀ x : ℝ, f (x) = f (-x)) -- f is even
variable (monotonic_incr : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) -- f is monotonically increasing in [0, +∞)

theorem range_of_a
  (h : f (Real.log a / Real.log 2) + f (Real.log (1/a) / Real.log 2) ≤ 2 * f 1) : 
  1 / 2 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l307_30726


namespace tangent_product_value_l307_30752

theorem tangent_product_value (A B : ℝ) (hA : A = 20) (hB : B = 25) :
    (1 + Real.tan (A * Real.pi / 180)) * (1 + Real.tan (B * Real.pi / 180)) = 2 :=
sorry

end tangent_product_value_l307_30752


namespace VerifyMultiplicationProperties_l307_30772

theorem VerifyMultiplicationProperties (α : Type) [Semiring α] :
  ((∀ x y z : α, (x * y) * z = x * (y * z)) ∧
   (∀ x y : α, x * y = y * x) ∧
   (∀ x y z : α, x * (y + z) = x * y + x * z) ∧
   (∃ e : α, ∀ x : α, x * e = x)) := by
  sorry

end VerifyMultiplicationProperties_l307_30772


namespace remainder_1394_mod_2535_l307_30716

-- Definition of the least number satisfying the given conditions
def L : ℕ := 1394

-- Proof statement: proving the remainder of division
theorem remainder_1394_mod_2535 : (1394 % 2535) = 1394 :=
by sorry

end remainder_1394_mod_2535_l307_30716


namespace joan_dimes_spent_l307_30713

theorem joan_dimes_spent (initial_dimes remaining_dimes spent_dimes : ℕ) 
    (h_initial: initial_dimes = 5) 
    (h_remaining: remaining_dimes = 3) : 
    spent_dimes = initial_dimes - remaining_dimes := 
by 
    sorry

end joan_dimes_spent_l307_30713


namespace sum_of_roots_l307_30782

theorem sum_of_roots (f : ℝ → ℝ) :
  (∀ x : ℝ, f (3 + x) = f (3 - x)) →
  (∃ (S : Finset ℝ), S.card = 6 ∧ ∀ x ∈ S, f x = 0) →
  (∃ (S : Finset ℝ), S.sum id = 18) :=
by
  sorry

end sum_of_roots_l307_30782


namespace solution_set_abs_inequality_l307_30717

theorem solution_set_abs_inequality (x : ℝ) : |3 - x| + |x - 7| ≤ 8 ↔ 1 ≤ x ∧ x ≤ 9 :=
sorry

end solution_set_abs_inequality_l307_30717


namespace product_of_all_possible_values_l307_30750

theorem product_of_all_possible_values (x : ℝ) :
  (|16 / x + 4| = 3) → ((x = -16 ∨ x = -16 / 7) →
  (x_1 = -16 ∧ x_2 = -16 / 7) →
  (x_1 * x_2 = 256 / 7)) :=
sorry

end product_of_all_possible_values_l307_30750


namespace steve_speed_ratio_l307_30760

/-- Define the distance from Steve's house to work. -/
def distance_to_work := 30

/-- Define the total time spent on the road by Steve. -/
def total_time_on_road := 6

/-- Define Steve's speed on the way back from work. -/
def speed_back := 15

/-- Calculate the ratio of Steve's speed on the way back to his speed on the way to work. -/
theorem steve_speed_ratio (v : ℝ) (h_v_pos : v > 0) 
    (h1 : distance_to_work / v + distance_to_work / speed_back = total_time_on_road) :
    speed_back / v = 2 := 
by
  -- We will provide the proof here
  sorry

end steve_speed_ratio_l307_30760


namespace right_triangle_eqn_roots_indeterminate_l307_30739

theorem right_triangle_eqn_roots_indeterminate 
  (a b c : ℝ) (h : a^2 + c^2 = b^2) : 
  ¬(∃ Δ, Δ = 4 - 4 * c^2 ∧ (Δ > 0 ∨ Δ = 0 ∨ Δ < 0)) →
  (¬∃ x, a * (x^2 - 1) - 2 * x + b * (x^2 + 1) = 0 ∨
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ a * (x₁^2 - 1) - 2 * x₁ + b * (x₁^2 + 1) = 0 ∧ a * (x₂^2 - 1) - 2 * x₂ + b * (x₂^2 + 1) = 0) :=
by
  sorry

end right_triangle_eqn_roots_indeterminate_l307_30739


namespace number_of_action_figures_bought_l307_30798

-- Definitions of conditions
def cost_of_board_game : ℕ := 2
def cost_per_action_figure : ℕ := 7
def total_spent : ℕ := 30

-- The problem to prove
theorem number_of_action_figures_bought : 
  ∃ (n : ℕ), total_spent - cost_of_board_game = n * cost_per_action_figure ∧ n = 4 :=
by
  sorry

end number_of_action_figures_bought_l307_30798


namespace greatest_integer_leq_l307_30771

theorem greatest_integer_leq (a b : ℝ) (ha : a = 5^150) (hb : b = 3^150) (c d : ℝ) (hc : c = 5^147) (hd : d = 3^147):
  ⌊ (a + b) / (c + d) ⌋ = 124 := 
sorry

end greatest_integer_leq_l307_30771


namespace arithmetic_sequence_n_value_l307_30724

theorem arithmetic_sequence_n_value
  (a : ℕ → ℚ)
  (h1 : a 1 = 1 / 3)
  (h2 : a 2 + a 5 = 4)
  (h3 : a n = 33)
  : n = 50 :=
sorry

end arithmetic_sequence_n_value_l307_30724


namespace pipe_length_difference_l307_30733

theorem pipe_length_difference 
  (total_length shorter_piece : ℝ)
  (h1: total_length = 68) 
  (h2: shorter_piece = 28) : 
  ∃ longer_piece diff : ℝ, longer_piece = total_length - shorter_piece ∧ diff = longer_piece - shorter_piece ∧ diff = 12 :=
by
  sorry

end pipe_length_difference_l307_30733


namespace present_age_of_son_l307_30793

variable (S M : ℕ)

-- Conditions
def condition1 : Prop := M = S + 32
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- Theorem stating the required proof
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 30 := by
  sorry

end present_age_of_son_l307_30793


namespace largest_number_is_y_l307_30778

def x := 8.1235
def y := 8.12355555555555 -- 8.123\overline{5}
def z := 8.12345454545454 -- 8.123\overline{45}
def w := 8.12345345345345 -- 8.12\overline{345}
def v := 8.12345234523452 -- 8.1\overline{2345}

theorem largest_number_is_y : y > x ∧ y > z ∧ y > w ∧ y > v :=
by
-- Proof steps would go here.
sorry

end largest_number_is_y_l307_30778


namespace alcohol_water_ratio_l307_30729

theorem alcohol_water_ratio 
  (P_alcohol_pct : ℝ) (Q_alcohol_pct : ℝ) 
  (P_volume : ℝ) (Q_volume : ℝ) 
  (mixture_alcohol : ℝ) (mixture_water : ℝ)
  (h1 : P_alcohol_pct = 62.5)
  (h2 : Q_alcohol_pct = 87.5)
  (h3 : P_volume = 4)
  (h4 : Q_volume = 4)
  (ha : mixture_alcohol = (P_volume * (P_alcohol_pct / 100)) + (Q_volume * (Q_alcohol_pct / 100)))
  (hm : mixture_water = (P_volume + Q_volume) - mixture_alcohol) :
  mixture_alcohol / mixture_water = 3 :=
by
  sorry

end alcohol_water_ratio_l307_30729


namespace equal_share_of_marbles_l307_30715

-- Define the number of marbles bought by each friend based on the conditions
def wolfgang_marbles : ℕ := 16
def ludo_marbles : ℕ := wolfgang_marbles + (wolfgang_marbles / 4)
def michael_marbles : ℕ := 2 * (wolfgang_marbles + ludo_marbles) / 3
def shania_marbles : ℕ := 2 * ludo_marbles
def gabriel_marbles : ℕ := (wolfgang_marbles + ludo_marbles + michael_marbles + shania_marbles) - 1
def total_marbles : ℕ := wolfgang_marbles + ludo_marbles + michael_marbles + shania_marbles + gabriel_marbles
def marbles_per_friend : ℕ := total_marbles / 5

-- Mathematical equivalent proof problem
theorem equal_share_of_marbles : marbles_per_friend = 39 := by
  sorry

end equal_share_of_marbles_l307_30715


namespace seq_geom_prog_l307_30765

theorem seq_geom_prog (a : ℕ → ℝ) (b : ℝ) (h_pos_b : 0 < b)
  (h_pos_a : ∀ n, 0 < a n)
  (h_recurrence : ∀ n, a (n + 2) = (b + 1) * a n * a (n + 1)) :
  (∃ r, ∀ n, a (n + 1) = r * a n) ↔ a 0 = a 1 :=
sorry

end seq_geom_prog_l307_30765


namespace gcf_of_36_and_54_l307_30704

theorem gcf_of_36_and_54 : Nat.gcd 36 54 = 18 := 
by
  sorry

end gcf_of_36_and_54_l307_30704


namespace smallest_five_digit_multiple_of_18_l307_30754

theorem smallest_five_digit_multiple_of_18 : ∃ n : ℕ, 10000 ≤ n ∧ n ≤ 99999 ∧ n % 18 = 0 ∧ ∀ m : ℕ, 10000 ≤ m ∧ m ≤ 99999 ∧ m % 18 = 0 → n ≤ m :=
  sorry

end smallest_five_digit_multiple_of_18_l307_30754


namespace harry_less_than_half_selena_l307_30734

-- Definitions of the conditions
def selena_book_pages := 400
def harry_book_pages := 180
def half (n : ℕ) := n / 2

-- The theorem to prove that Harry's book is 20 pages less than half of Selena's book.
theorem harry_less_than_half_selena :
  harry_book_pages = half selena_book_pages - 20 := 
by
  sorry

end harry_less_than_half_selena_l307_30734


namespace total_volume_stacked_dice_l307_30777

def die_volume (width length height : ℕ) : ℕ := 
  width * length * height

def total_dice (horizontal vertical layers : ℕ) : ℕ := 
  horizontal * vertical * layers

theorem total_volume_stacked_dice :
  let width := 1
  let length := 1
  let height := 1
  let horizontal := 7
  let vertical := 5
  let layers := 3
  let single_die_volume := die_volume width length height
  let num_dice := total_dice horizontal vertical layers
  single_die_volume * num_dice = 105 :=
by
  sorry  -- proof to be provided

end total_volume_stacked_dice_l307_30777


namespace percentage_of_loss_l307_30747

theorem percentage_of_loss (CP SP : ℕ) (h1 : CP = 1750) (h2 : SP = 1610) : 
  (CP - SP) * 100 / CP = 8 := by
  sorry

end percentage_of_loss_l307_30747


namespace arc_length_parametric_curve_l307_30753

noncomputable def arcLength (x y : ℝ → ℝ) (t1 t2 : ℝ) : ℝ :=
  ∫ t in t1..t2, Real.sqrt ((deriv x t)^2 + (deriv y t)^2)

theorem arc_length_parametric_curve :
    (∫ t in (0 : ℝ)..(3 * Real.pi), 
        Real.sqrt ((deriv (fun t => (t ^ 2 - 2) * Real.sin t + 2 * t * Real.cos t) t) ^ 2 +
                   (deriv (fun t => (2 - t ^ 2) * Real.cos t + 2 * t * Real.sin t) t) ^ 2)) =
    9 * Real.pi ^ 3 :=
by
  -- The proof is omitted
  sorry

end arc_length_parametric_curve_l307_30753


namespace distinct_solutions_of_transformed_eq_l307_30705

open Function

variable {R : Type} [Field R]

def cubic_func (a b c d : R) (x : R) : R := a*x^3 + b*x^2 + c*x + d

noncomputable def three_distinct_roots {a b c d : R} (f : R → R)
  (h : ∀ x, f x = a*x^3 + b*x^2 + c*x + d) : Prop :=
∃ α β γ, α ≠ β ∧ β ≠ γ ∧ γ ≠ α ∧ f α = 0 ∧ f β = 0 ∧ f γ = 0

theorem distinct_solutions_of_transformed_eq
  {a b c d : R} (h : ∃ α β γ, α ≠ β ∧ β ≠ γ ∧ γ ≠ α ∧ (cubic_func a b c d α) = 0 ∧ (cubic_func a b c d β) = 0 ∧ (cubic_func a b c d γ) = 0) :
  ∃ p q, p ≠ q ∧ (4 * (cubic_func a b c d p) * (3 * a * p + b) = (3 * a * p^2 + 2 * b * p + c)^2) ∧ 
              (4 * (cubic_func a b c d q) * (3 * a * q + b) = (3 * a * q^2 + 2 * b * q + c)^2) := sorry

end distinct_solutions_of_transformed_eq_l307_30705


namespace find_chord_points_l307_30743

/-
Define a parabola and check if the points given form a chord that intersects 
the point (8,4) in the ratio 1:4.
-/

def parabola (P : ℝ × ℝ) : Prop :=
  P.snd^2 = 4 * P.fst

def divides_in_ratio (C A B : ℝ × ℝ) (m n : ℝ) : Prop :=
  (A.fst * n + B.fst * m = C.fst * (m + n)) ∧ 
  (A.snd * n + B.snd * m = C.snd * (m + n))

theorem find_chord_points :
  ∃ (P1 P2 : ℝ × ℝ),
  parabola P1 ∧
  parabola P2 ∧
  divides_in_ratio (8, 4) P1 P2 1 4 ∧ 
  ((P1 = (1, 2) ∧ P2 = (36, 12)) ∨ (P1 = (9, 6) ∧ P2 = (4, -4))) :=
sorry

end find_chord_points_l307_30743


namespace smallest_k_multiple_of_180_l307_30764

def sum_of_squares (k : ℕ) : ℕ :=
  (k * (k + 1) * (2 * k + 1)) / 6

def divisible_by_180 (n : ℕ) : Prop :=
  n % 180 = 0

theorem smallest_k_multiple_of_180 :
  ∃ k : ℕ, k > 0 ∧ divisible_by_180 (sum_of_squares k) ∧ ∀ m : ℕ, m > 0 ∧ divisible_by_180 (sum_of_squares m) → k ≤ m :=
sorry

end smallest_k_multiple_of_180_l307_30764


namespace range_a_for_false_proposition_l307_30787

theorem range_a_for_false_proposition :
  {a : ℝ | ¬ ∃ x : ℝ, x^2 + (1 - a) * x < 0} = {1} :=
sorry

end range_a_for_false_proposition_l307_30787


namespace crop_yield_solution_l307_30710

variable (x y : ℝ)

axiom h1 : 3 * x + 6 * y = 4.7
axiom h2 : 5 * x + 3 * y = 5.5

theorem crop_yield_solution :
  x = 0.9 ∧ y = 1/3 :=
by
  sorry

end crop_yield_solution_l307_30710


namespace perpendicular_vectors_k_value_l307_30794

theorem perpendicular_vectors_k_value (k : ℝ) (a b: ℝ × ℝ)
  (h_a : a = (-1, 3)) (h_b : b = (1, k)) (h_perp : (a.1 * b.1 + a.2 * b.2) = 0) :
  k = 1 / 3 :=
by
  sorry

end perpendicular_vectors_k_value_l307_30794


namespace find_two_digit_number_l307_30783

theorem find_two_digit_number : 
  ∃ (a b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ (b = 0 ∨ b = 5) ∧ (10 * a + b = 5 * (a + b)) ∧ (10 * a + b = 45) :=
by
  sorry

end find_two_digit_number_l307_30783


namespace ancient_china_pentatonic_scale_l307_30768

theorem ancient_china_pentatonic_scale (a : ℝ) (h : a * (2/3) * (4/3) * (2/3) = 32) : a = 54 :=
by
  sorry

end ancient_china_pentatonic_scale_l307_30768


namespace range_of_a_l307_30746

open Set

def A (a : ℝ) : Set ℝ := { x | a - 1 ≤ x ∧ x ≤ 2 * a + 1 }
def B : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }

theorem range_of_a (a : ℝ) (h : A a ∪ B = B) : a ∈ Iio (-2) ∪ Icc (-1) (3 / 2) :=
by
  sorry

end range_of_a_l307_30746


namespace red_pairs_count_l307_30727

def num_green_students : Nat := 63
def num_red_students : Nat := 69
def total_pairs : Nat := 66
def num_green_pairs : Nat := 27

theorem red_pairs_count : 
  (num_red_students - (num_green_students - num_green_pairs * 2)) / 2 = 30 := 
by sorry

end red_pairs_count_l307_30727


namespace solve_Diamond_l307_30784

theorem solve_Diamond :
  ∀ (Diamond : ℕ), (Diamond * 7 + 4 = Diamond * 8 + 1) → Diamond = 3 :=
by
  intros Diamond h
  sorry

end solve_Diamond_l307_30784


namespace part1_solution_set_part2_range_a_l307_30766

def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

theorem part1_solution_set (x : ℝ) (h : f x 2 ≥ 4) : 
  x ≤ 3/2 ∨ x ≥ 11/2 :=
sorry

theorem part2_range_a (a : ℝ) (h : ∃ x, f x a ≥ 4) : 
  a ≤ -1 ∨ a ≥ 3 :=
sorry

end part1_solution_set_part2_range_a_l307_30766


namespace part_one_part_two_l307_30742

def M (n : ℤ) : ℤ := n - 3
def M_frac (n : ℚ) : ℚ := - (1 / n^2)

theorem part_one 
    : M 28 * M_frac (1/5) = -1 :=
by {
  sorry
}

theorem part_two 
    : -1 / M 39 / (- M_frac (1/6)) = -1 :=
by {
  sorry
}

end part_one_part_two_l307_30742


namespace sandcastle_ratio_l307_30720

-- Definitions based on conditions in a)
def sandcastles_on_marks_beach : ℕ := 20
def towers_per_sandcastle_marks_beach : ℕ := 10
def towers_per_sandcastle_jeffs_beach : ℕ := 5
def total_combined_sandcastles_and_towers : ℕ := 580

-- The main statement to prove
theorem sandcastle_ratio : 
  ∃ (J : ℕ), 
  (sandcastles_on_marks_beach + (towers_per_sandcastle_marks_beach * sandcastles_on_marks_beach) + J + (towers_per_sandcastle_jeffs_beach * J) = total_combined_sandcastles_and_towers) ∧ 
  (J / sandcastles_on_marks_beach = 3) :=
by 
  sorry

end sandcastle_ratio_l307_30720


namespace cards_per_layer_l307_30799

theorem cards_per_layer (total_decks : ℕ) (cards_per_deck : ℕ) (layers : ℕ) (h_decks : total_decks = 16) (h_cards_per_deck : cards_per_deck = 52) (h_layers : layers = 32) :
  total_decks * cards_per_deck / layers = 26 :=
by {
  -- To skip the proof
  sorry
}

end cards_per_layer_l307_30799


namespace find_amount_with_r_l307_30737

variable (p q r : ℝ)

-- Condition 1: p, q, and r have Rs. 6000 among themselves.
def total_amount : Prop := p + q + r = 6000

-- Condition 2: r has two-thirds of the total amount with p and q.
def r_amount : Prop := r = (2 / 3) * (p + q)

theorem find_amount_with_r (h1 : total_amount p q r) (h2 : r_amount p q r) : r = 2400 := by
  sorry

end find_amount_with_r_l307_30737


namespace minimum_distance_from_circle_to_line_l307_30725

noncomputable def point_on_circle (θ : ℝ) : ℝ × ℝ :=
  (1 + 2 * Real.cos θ, 1 + 2 * Real.sin θ)

def line_eq (p : ℝ × ℝ) : ℝ :=
  p.1 - p.2 + 4

noncomputable def distance_from_point_to_line (p : ℝ × ℝ) : ℝ :=
  |p.1 - p.2 + 4| / Real.sqrt (1^2 + 1^2)

theorem minimum_distance_from_circle_to_line :
  ∀ θ : ℝ, (∃ θ, distance_from_point_to_line (point_on_circle θ) = 2 * Real.sqrt 2 - 2) :=
by
  sorry

end minimum_distance_from_circle_to_line_l307_30725


namespace price_of_brand_X_pen_l307_30779

variable (P : ℝ)

theorem price_of_brand_X_pen :
  (∀ (n : ℕ), n = 12 → 6 * P + 6 * 2.20 = 42 - 13.20) →
  P = 4.80 :=
by
  intro h₁
  have h₂ := h₁ 12 rfl
  sorry

end price_of_brand_X_pen_l307_30779


namespace each_child_plays_for_90_minutes_l307_30709

-- Definitions based on the conditions
def total_playing_time : ℕ := 180
def children_playing_at_a_time : ℕ := 3
def total_children : ℕ := 6

-- The proof problem statement
theorem each_child_plays_for_90_minutes :
  (children_playing_at_a_time * total_playing_time) / total_children = 90 := by
  sorry

end each_child_plays_for_90_minutes_l307_30709


namespace no_negative_roots_but_at_least_one_positive_root_l307_30767

def f (x : ℝ) : ℝ := x^6 - 3 * x^5 - 6 * x^3 - x + 8

theorem no_negative_roots_but_at_least_one_positive_root :
  (∀ x : ℝ, x < 0 → f x ≠ 0) ∧ (∃ x : ℝ, x > 0 ∧ f x = 0) :=
by {
  sorry
}

end no_negative_roots_but_at_least_one_positive_root_l307_30767


namespace jim_caught_fish_l307_30718

variable (ben judy billy susie jim caught_back total_filets : ℕ)

def caught_fish : ℕ :=
  ben + judy + billy + susie + jim - caught_back

theorem jim_caught_fish (h_ben : ben = 4)
                        (h_judy : judy = 1)
                        (h_billy : billy = 3)
                        (h_susie : susie = 5)
                        (h_caught_back : caught_back = 3)
                        (h_total_filets : total_filets = 24)
                        (h_filets_per_fish : ∀ f : ℕ, total_filets = f * 2 → caught_fish ben judy billy susie jim caught_back = f) :
  jim = 2 :=
by
  -- Proof goes here
  sorry

end jim_caught_fish_l307_30718


namespace arithmetic_expression_l307_30757

theorem arithmetic_expression :
  7 / 2 - 3 - 5 + 3 * 4 = 7.5 :=
by {
  -- We state the main equivalence to be proven
  sorry
}

end arithmetic_expression_l307_30757


namespace angle_E_in_quadrilateral_EFGH_l307_30770

theorem angle_E_in_quadrilateral_EFGH 
  (angle_E angle_F angle_G angle_H : ℝ) 
  (h1 : angle_E = 2 * angle_F)
  (h2 : angle_E = 3 * angle_G)
  (h3 : angle_E = 6 * angle_H)
  (sum_angles : angle_E + angle_F + angle_G + angle_H = 360) : 
  angle_E = 180 :=
by
  sorry

end angle_E_in_quadrilateral_EFGH_l307_30770


namespace liter_kerosene_cost_friday_l307_30735

-- Define initial conditions.
def cost_pound_rice_monday : ℚ := 0.36
def cost_dozen_eggs_monday : ℚ := cost_pound_rice_monday
def cost_half_liter_kerosene_monday : ℚ := (8 / 12) * cost_dozen_eggs_monday

-- Define the Wednesday price increase.
def percent_increase_rice : ℚ := 0.20
def cost_pound_rice_wednesday : ℚ := cost_pound_rice_monday * (1 + percent_increase_rice)
def cost_half_liter_kerosene_wednesday : ℚ := cost_half_liter_kerosene_monday * (1 + percent_increase_rice)

-- Define the Friday discount on eggs.
def percent_discount_eggs : ℚ := 0.10
def cost_dozen_eggs_friday : ℚ := cost_dozen_eggs_monday * (1 - percent_discount_eggs)
def cost_per_egg_friday : ℚ := cost_dozen_eggs_friday / 12

-- Define the price calculation for a liter of kerosene on Wednesday.
def cost_liter_kerosene_wednesday : ℚ := 2 * cost_half_liter_kerosene_wednesday

-- Define the final goal.
def cost_liter_kerosene_friday := cost_liter_kerosene_wednesday

theorem liter_kerosene_cost_friday : cost_liter_kerosene_friday = 0.576 := by
  sorry

end liter_kerosene_cost_friday_l307_30735


namespace transformation_maps_segment_l307_30795

variables (C D : ℝ × ℝ) (C' D' : ℝ × ℝ)

def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem transformation_maps_segment :
  reflect_x (reflect_y (3, -2)) = (-3, 2) ∧ reflect_x (reflect_y (4, -5)) = (-4, 5) :=
by {
  sorry
}

end transformation_maps_segment_l307_30795


namespace find_divisor_nearest_to_3105_l307_30740

def nearest_divisible_number (n : ℕ) (d : ℕ) : ℕ :=
  if n % d = 0 then n else n + d - (n % d)

theorem find_divisor_nearest_to_3105 (d : ℕ) (h : nearest_divisible_number 3105 d = 3108) : d = 3 :=
by
  sorry

end find_divisor_nearest_to_3105_l307_30740


namespace smithtown_left_handed_women_percentage_l307_30758

theorem smithtown_left_handed_women_percentage
    (x y : ℕ)
    (H1 : 3 * x + x = 4 * x)
    (H2 : 3 * y + 2 * y = 5 * y)
    (H3 : 4 * x = 5 * y) :
    (x / (4 * x)) * 100 = 25 :=
by sorry

end smithtown_left_handed_women_percentage_l307_30758


namespace g_of_f_three_l307_30723

def f (x : ℝ) : ℝ := x^3 - 2
def g (x : ℝ) : ℝ := 3 * x^2 + x + 2

theorem g_of_f_three : g (f 3) = 1902 :=
by
  sorry

end g_of_f_three_l307_30723


namespace equivalent_modulo_l307_30780

theorem equivalent_modulo:
  123^2 * 947 % 60 = 3 :=
by
  sorry

end equivalent_modulo_l307_30780


namespace product_of_numbers_l307_30722

variable (x y : ℕ)

theorem product_of_numbers : x + y = 120 ∧ x - y = 6 → x * y = 3591 := by
  sorry

end product_of_numbers_l307_30722


namespace find_percentage_loss_l307_30749

theorem find_percentage_loss 
  (P : ℝ)
  (initial_marbles remaining_marbles : ℝ)
  (h1 : initial_marbles = 100)
  (h2 : remaining_marbles = 20)
  (h3 : (initial_marbles - initial_marbles * P / 100) / 2 = remaining_marbles) :
  P = 60 :=
by
  sorry

end find_percentage_loss_l307_30749


namespace quadratic_inequality_solution_l307_30755

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, k * x^2 + k * x - (3 / 4) < 0) ↔ -3 < k ∧ k ≤ 0 :=
by
  sorry

end quadratic_inequality_solution_l307_30755


namespace abs_minus_five_plus_three_l307_30707

theorem abs_minus_five_plus_three : |(-5 + 3)| = 2 := 
by
  sorry

end abs_minus_five_plus_three_l307_30707


namespace least_multiple_25_gt_500_l307_30738

theorem least_multiple_25_gt_500 : ∃ (k : ℕ), 25 * k > 500 ∧ (∀ m : ℕ, (25 * m > 500 → 25 * k ≤ 25 * m)) :=
by
  use 21
  sorry

end least_multiple_25_gt_500_l307_30738


namespace solve_for_x_l307_30775

theorem solve_for_x (x : ℝ) (h : x + 2 = 7) : x = 5 := 
by
  sorry

end solve_for_x_l307_30775


namespace total_cookies_l307_30785

-- Conditions
def Paul_cookies : ℕ := 45
def Paula_cookies : ℕ := Paul_cookies - 3

-- Question and Answer
theorem total_cookies : Paul_cookies + Paula_cookies = 87 := by
  sorry

end total_cookies_l307_30785


namespace isosceles_triangle_angle_between_vectors_l307_30762

theorem isosceles_triangle_angle_between_vectors 
  (α β γ : ℝ) 
  (h1: α + β + γ = 180)
  (h2: α = 120) 
  (h3: β = γ):
  180 - β = 150 :=
sorry

end isosceles_triangle_angle_between_vectors_l307_30762


namespace composite_function_properties_l307_30769

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem composite_function_properties
  (a b c : ℝ)
  (h_a_nonzero : a ≠ 0)
  (h_no_real_roots : ∀ x : ℝ, f a b c x ≠ x) :
  (∀ x : ℝ, f a b c (f a b c x) ≠ x) ∧
  (a > 0 → ∀ x : ℝ, f a b c (f a b c x) > x) :=
by sorry

end composite_function_properties_l307_30769


namespace charcoal_drawings_correct_l307_30761

-- Define the constants based on the problem conditions
def total_drawings : ℕ := 120
def colored_pencils : ℕ := 35
def blending_markers : ℕ := 22
def pastels : ℕ := 15
def watercolors : ℕ := 12

-- Calculate the total number of charcoal drawings
def charcoal_drawings : ℕ := total_drawings - (colored_pencils + blending_markers + pastels + watercolors)

-- The theorem we want to prove is that the number of charcoal drawings is 36
theorem charcoal_drawings_correct : charcoal_drawings = 36 :=
by
  -- The proof goes here (we skip it with 'sorry')
  sorry

end charcoal_drawings_correct_l307_30761


namespace problem1_solution_problem2_solution_l307_30796

-- Problem 1: Prove that x = 1 given 6x - 7 = 4x - 5
theorem problem1_solution (x : ℝ) (h : 6 * x - 7 = 4 * x - 5) : x = 1 := by
  sorry


-- Problem 2: Prove that x = -1 given (3x - 1) / 4 - 1 = (5x - 7) / 6
theorem problem2_solution (x : ℝ) (h : (3 * x - 1) / 4 - 1 = (5 * x - 7) / 6) : x = -1 := by
  sorry

end problem1_solution_problem2_solution_l307_30796


namespace number_of_students_in_both_ball_and_track_l307_30781

variable (total studentsSwim studentsTrack studentsBall bothSwimTrack bothSwimBall bothTrackBall : ℕ)
variable (noAllThree : Prop)

theorem number_of_students_in_both_ball_and_track
  (h_total : total = 26)
  (h_swim : studentsSwim = 15)
  (h_track : studentsTrack = 8)
  (h_ball : studentsBall = 14)
  (h_both_swim_track : bothSwimTrack = 3)
  (h_both_swim_ball : bothSwimBall = 3)
  (h_no_all_three : noAllThree) :
  bothTrackBall = 5 := by
  sorry

end number_of_students_in_both_ball_and_track_l307_30781


namespace prob_simultaneous_sequences_l307_30751

-- Definitions for coin probabilities
def prob_heads_A : ℝ := 0.3
def prob_tails_A : ℝ := 0.7
def prob_heads_B : ℝ := 0.4
def prob_tails_B : ℝ := 0.6

-- Definitions for required sequences
def seq_TTH_A : ℝ := prob_tails_A * prob_tails_A * prob_heads_A
def seq_HTT_B : ℝ := prob_heads_B * prob_tails_B * prob_tails_B

-- Main assertion
theorem prob_simultaneous_sequences :
  seq_TTH_A * seq_HTT_B = 0.021168 :=
by
  sorry

end prob_simultaneous_sequences_l307_30751


namespace find_a_l307_30741

theorem find_a 
  (x y a : ℝ)
  (h₁ : x - 3 ≤ 0)
  (h₂ : y - a ≤ 0)
  (h₃ : x + y ≥ 0)
  (h₄ : ∃ (x y : ℝ), 2*x + y = 10): a = 4 :=
sorry

end find_a_l307_30741


namespace expand_and_simplify_l307_30745

theorem expand_and_simplify (x : ℝ) : (x + 6) * (x - 11) = x^2 - 5 * x - 66 :=
by
  sorry

end expand_and_simplify_l307_30745


namespace find_ac_bc_val_l307_30756

variable (a b c d : ℚ)
variable (h_neq : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
variable (h1 : (a + c) * (a + d) = 1)
variable (h2 : (b + c) * (b + d) = 1)

theorem find_ac_bc_val : (a + c) * (b + c) = -1 := 
by 
  sorry

end find_ac_bc_val_l307_30756


namespace tank_emptying_time_correct_l307_30703

noncomputable def tank_emptying_time : ℝ :=
  let initial_volume := 1 / 5
  let fill_rate := 1 / 15
  let empty_rate := 1 / 6
  let combined_rate := fill_rate - empty_rate
  initial_volume / combined_rate

theorem tank_emptying_time_correct :
  tank_emptying_time = 2 :=
by
  -- Proof will be provided here
  sorry

end tank_emptying_time_correct_l307_30703


namespace simplify_cbrt_expr_l307_30714

-- Define the cube root function.
def cbrt (x : ℝ) : ℝ := x^(1/3)

-- Define the original expression under the cube root.
def original_expr : ℝ := 40^3 + 70^3 + 100^3

-- Define the simplified expression.
def simplified_expr : ℝ := 10 * cbrt 1407

theorem simplify_cbrt_expr : cbrt original_expr = simplified_expr := by
  -- Declaration that proof is not provided to ensure Lean statement is complete.
  sorry

end simplify_cbrt_expr_l307_30714


namespace incorrect_equation_l307_30736

noncomputable def x : ℂ := (-1 + Real.sqrt 3 * Complex.I) / 2
noncomputable def y : ℂ := (-1 - Real.sqrt 3 * Complex.I) / 2

theorem incorrect_equation : x^9 + y^9 ≠ -1 := sorry

end incorrect_equation_l307_30736


namespace range_of_m_l307_30719

-- Definitions based on the given conditions
def setA : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def setB (m : ℝ) : Set ℝ := {x | 2 * m - 1 < x ∧ x < m + 1}

-- Lean statement of the problem
theorem range_of_m (m : ℝ) (h : setB m ⊆ setA) : m ≥ -1 :=
sorry  -- proof is not required

end range_of_m_l307_30719


namespace volume_relation_l307_30773

-- Definitions for points and geometry structures
structure Point3D :=
(x : ℝ) (y : ℝ) (z : ℝ)

structure Tetrahedron :=
(A B C D : Point3D)

-- Volume function for Tetrahedron
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

-- Given conditions
variable {A B C D D1 A1 B1 C1 : Point3D} 

-- D_1 is the centroid of triangle ABC
axiom centroid_D1 (A B C D1 : Point3D) : D1 = Point3D.mk ((A.x + B.x + C.x) / 3) ((A.y + B.y + C.y) / 3) ((A.z + B.z + C.z) / 3)

-- Line through A parallel to DD_1 intersects plane BCD at A1
axiom A1_condition (A B C D D1 A1 : Point3D) : sorry
-- Line through B parallel to DD_1 intersects plane ACD at B1
axiom B1_condition (A B C D D1 B1 : Point3D) : sorry
-- Line through C parallel to DD_1 intersects plane ABD at C1
axiom C1_condition (A B C D D1 C1 : Point3D) : sorry

-- Volume relation to be proven
theorem volume_relation (t1 t2 : Tetrahedron) (h : t1.A = A ∧ t1.B = B ∧ t1.C = C ∧ t1.D = D ∧
                                                t2.A = A1 ∧ t2.B = B1 ∧ t2.C = C1 ∧ t2.D = D1) :
  volume t1 = 2 * volume t2 := 
sorry

end volume_relation_l307_30773


namespace nat_forms_6n_plus_1_or_5_prod_6n_plus_1_prod_6n_plus_5_prod_6n_plus_1_and_5_l307_30728

theorem nat_forms_6n_plus_1_or_5 (x : ℕ) (h1 : ¬ (x % 2 = 0) ∧ ¬ (x % 3 = 0)) :
  ∃ n : ℕ, x = 6 * n + 1 ∨ x = 6 * n + 5 := 
sorry

theorem prod_6n_plus_1 (m n : ℕ) :
  (6 * m + 1) * (6 * n + 1) = 6 * (6 * m * n + m + n) + 1 :=
sorry

theorem prod_6n_plus_5 (m n : ℕ) :
  (6 * m + 5) * (6 * n + 5) = 6 * (6 * m * n + 5 * m + 5 * n + 4) + 1 :=
sorry

theorem prod_6n_plus_1_and_5 (m n : ℕ) :
  (6 * m + 1) * (6 * n + 5) = 6 * (6 * m * n + 5 * m + n) + 5 :=
sorry

end nat_forms_6n_plus_1_or_5_prod_6n_plus_1_prod_6n_plus_5_prod_6n_plus_1_and_5_l307_30728


namespace simplify_expression_l307_30748

variable {x y z : ℝ} 
variable (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0)

theorem simplify_expression :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = (x * y * z)⁻¹ * (x + y + z)⁻¹ :=
sorry

end simplify_expression_l307_30748


namespace remainder_3a_plus_b_l307_30789

theorem remainder_3a_plus_b (p q : ℤ) (a b : ℤ)
  (h1 : a = 98 * p + 92)
  (h2 : b = 147 * q + 135) :
  ((3 * a + b) % 49) = 19 := by
sorry

end remainder_3a_plus_b_l307_30789


namespace sum_of_positive_odd_divisors_of_90_l307_30763

-- Define the conditions: the odd divisors of 45
def odd_divisors_45 : List ℕ := [1, 3, 5, 9, 15, 45]

-- Define a function to sum the elements of a list
def sum_list (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Now, state the theorem
theorem sum_of_positive_odd_divisors_of_90 : sum_list odd_divisors_45 = 78 := by
  sorry

end sum_of_positive_odd_divisors_of_90_l307_30763


namespace range_of_f_l307_30776

noncomputable def f (x : ℝ) : ℝ := if x < 1 then 3 * x - 1 else 2 * x ^ 2

theorem range_of_f (a : ℝ) : (f (f a) = 2 * (f a) ^ 2) ↔ (a ≥ 2 / 3 ∨ a = 1 / 2) := 
  sorry

end range_of_f_l307_30776


namespace eval_p_positive_int_l307_30790

theorem eval_p_positive_int (p : ℕ) : 
  (∃ n : ℕ, n > 0 ∧ (4 * p + 20) = n * (3 * p - 6)) ↔ p = 3 ∨ p = 4 ∨ p = 15 ∨ p = 28 := 
by sorry

end eval_p_positive_int_l307_30790


namespace ratio_fifth_terms_l307_30712

variable (a_n b_n S_n T_n : ℕ → ℚ)

-- Conditions
variable (h : ∀ n, S_n n / T_n n = (9 * n + 2) / (n + 7))

-- Define the 5th term
def a_5 (S_n : ℕ → ℚ) : ℚ := S_n 9 / 9
def b_5 (T_n : ℕ → ℚ) : ℚ := T_n 9 / 9

-- Prove that the ratio of the 5th terms is 83 / 16
theorem ratio_fifth_terms :
  (a_5 S_n) / (b_5 T_n) = 83 / 16 :=
by
  sorry

end ratio_fifth_terms_l307_30712


namespace trapezoid_area_l307_30708

variable (x y : ℝ)

def condition1 : Prop := abs (y - 3 * x) ≥ abs (2 * y + x) ∧ -1 ≤ y - 3 ∧ y - 3 ≤ 1

def condition2 : Prop := (2 * y + y - y + 3 * x) * (2 * y + x + y - 3 * x) ≤ 0 ∧ 2 ≤ y ∧ y ≤ 4

theorem trapezoid_area (h1 : condition1 x y) (h2 : condition2 x y) :
  let A := (3, 2)
  let B := (-1/2, 2)
  let C := (-1, 4)
  let D := (6, 4)
  let S := (1/2) * (2 * (7 + 3.5))
  S = 10.5 :=
sorry

end trapezoid_area_l307_30708


namespace sum_x_y_is_4_l307_30730

theorem sum_x_y_is_4 {x y : ℝ} (h : x / (1 - (I : ℂ)) + y / (1 - 2 * I) = 5 / (1 - 3 * I)) : x + y = 4 :=
sorry

end sum_x_y_is_4_l307_30730


namespace proof_complex_ratio_l307_30700

noncomputable def condition1 (x y : ℂ) (k : ℝ) : Prop :=
  (x + k * y) / (x - k * y) + (x - k * y) / (x + k * y) = 1

theorem proof_complex_ratio (x y : ℂ) (k : ℝ) (h : condition1 x y k) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = (41 / 20 : ℂ) :=
by 
  sorry

end proof_complex_ratio_l307_30700
